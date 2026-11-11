from __future__ import annotations

import pytest
from app.infra.cache import MemoryCache


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """A monotonic clock the test moves by hand."""
    now = [1000.0]
    monkeypatch.setattr("app.infra.cache.time.monotonic", lambda: now[0])
    return now


async def test_window_hit_allows_exactly_the_limit(clock: list[float]) -> None:
    cache = MemoryCache()

    assert (await cache.window_hit("k", limit=2, window_s=60))[0] is True
    assert (await cache.window_hit("k", limit=2, window_s=60))[0] is True

    allowed, retry = await cache.window_hit("k", limit=2, window_s=60)
    assert allowed is False
    assert 59.0 <= retry <= 60.0
    await cache.close()


async def test_a_sliding_window_has_no_fixed_boundary_burst(clock: list[float]) -> None:
    cache = MemoryCache()
    clock[0] = 59.0
    for _ in range(3):
        assert (await cache.window_hit("k", limit=3, window_s=60))[0] is True

    # A fixed 60s bucket would reset at t=60 and allow a fresh burst. A sliding
    # window still remembers the three events from one second ago.
    clock[0] = 60.0
    assert (await cache.window_hit("k", limit=3, window_s=60))[0] is False
    await cache.close()


async def test_a_window_slot_is_freed_when_its_event_ages_out(clock: list[float]) -> None:
    cache = MemoryCache()
    assert (await cache.window_hit("k", limit=1, window_s=60))[0] is True
    assert (await cache.window_hit("k", limit=1, window_s=60))[0] is False

    clock[0] += 60.5
    assert (await cache.window_hit("k", limit=1, window_s=60))[0] is True
    await cache.close()


async def test_window_sum_tracks_amounts_and_expires(clock: list[float]) -> None:
    cache = MemoryCache()
    await cache.window_add("tokens", amount=40, window_s=60)
    clock[0] += 10
    await cache.window_add("tokens", amount=35, window_s=60)

    assert await cache.window_sum("tokens", window_s=60) == 75

    clock[0] += 51  # the first event is now older than the window
    assert await cache.window_sum("tokens", window_s=60) == 35
    await cache.close()


async def test_an_empty_window_sums_to_zero(clock: list[float]) -> None:
    cache = MemoryCache()
    assert await cache.window_sum("never-used", window_s=60) == 0.0
    await cache.close()
