"""Redis integration: the Lua paths that the in-memory tests cannot prove.

Gated on `CUEPILOT_TEST_REDIS_URL`. CI supplies a real Redis service; locally,
`make up` provides one on the default port.

The token bucket and sliding window live in Lua so replicas agree even with
skewed clocks. Only a real Redis executes those scripts, which is why this file
exists instead of another MemoryCache test.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import redis.asyncio as aioredis
from app.infra.cache import RedisCache, build_cache
from tests.conftest import make_settings

REDIS_URL = os.environ.get("CUEPILOT_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not REDIS_URL, reason="set CUEPILOT_TEST_REDIS_URL to run Redis integration tests"
)

PREFIX = "cuepilot-test:"


def _key(name: str) -> str:
    return f"{PREFIX}{uuid.uuid4().hex}:{name}"


@pytest.fixture()
async def cache():
    client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    redis_cache = RedisCache(client)
    yield redis_cache
    keys = await redis_cache.scan(PREFIX)
    if keys:
        await redis_cache.delete(*keys)
    await redis_cache.close()


async def test_token_bucket_bursts_then_refills(cache: RedisCache) -> None:
    key = _key("bucket")

    assert (await cache.token_bucket(key, capacity=2, refill_per_s=1))[0] is True
    assert (await cache.token_bucket(key, capacity=2, refill_per_s=1))[0] is True

    allowed, retry = await cache.token_bucket(key, capacity=2, refill_per_s=1)
    assert allowed is False
    assert 0.9 <= retry <= 1.1, retry

    # One second per token: half of one is not enough, and the denied attempt
    # must not have consumed anything.
    await asyncio.sleep(0.3)
    assert (await cache.token_bucket(key, capacity=2, refill_per_s=1))[0] is False

    await asyncio.sleep(0.8)
    assert (await cache.token_bucket(key, capacity=2, refill_per_s=1))[0] is True


async def test_token_bucket_is_shared_between_replicas(cache: RedisCache) -> None:
    key = _key("shared-bucket")
    other = RedisCache(
        aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    )
    try:
        assert (await cache.token_bucket(key, capacity=1, refill_per_s=0.01))[0] is True
        # The second replica sees the first replica's spend, so the limiter is
        # fleet-wide rather than per process.
        assert (await other.token_bucket(key, capacity=1, refill_per_s=0.01))[0] is False
    finally:
        await other.close()


async def test_sliding_window_is_shared_and_slides(cache: RedisCache) -> None:
    key = _key("window")
    other = RedisCache(
        aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    )
    try:
        assert (await cache.window_hit(key, limit=2, window_s=1))[0] is True
        assert (await cache.window_hit(key, limit=2, window_s=1))[0] is True

        allowed, retry = await other.window_hit(key, limit=2, window_s=1)
        assert allowed is False
        assert 0 < retry <= 1.0, retry

        await asyncio.sleep(1.05)
        assert (await other.window_hit(key, limit=2, window_s=1))[0] is True
    finally:
        await other.close()


async def test_window_sum_tracks_amounts_and_expires(cache: RedisCache) -> None:
    key = _key("sum")
    await cache.window_add(key, amount=40, window_s=0.3)
    await cache.window_add(key, amount=35, window_s=0.3)
    assert await cache.window_sum(key, window_s=0.3) == 75

    await asyncio.sleep(0.35)
    assert await cache.window_sum(key, window_s=0.3) == 0


async def test_idle_bucket_keys_expire(cache: RedisCache) -> None:
    key = _key("ttl")
    assert (await cache.token_bucket(key, capacity=5, refill_per_s=1))[0] is True

    client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    try:
        ttl_ms = await client.pttl(key)
        # capacity / refill = 5s, so an idle bucket cannot leak forever.
        assert 0 < ttl_ms <= 5000
    finally:
        await client.aclose()


async def test_build_cache_reports_redis_when_reachable() -> None:
    settings = make_settings(redis_url=REDIS_URL, infra_probe_timeout_s=2.0)
    built, reason = await build_cache(settings)
    try:
        assert built.backend == "redis"
        assert reason == ""
        assert await built.incr(_key("incr")) == 1
    finally:
        await built.close()
