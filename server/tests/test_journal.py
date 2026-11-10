from __future__ import annotations

import os
from typing import Any

import pytest
from app.infra.cache import MemoryCache
from app.telemetry.metrics import Metrics
from app.ws.journal import TurnJournal, TurnRecord
from app.ws.session import Session, SessionManager
from tests.conftest import make_settings

PG_URL = os.environ.get("CUEPILOT_TEST_DATABASE_URL")
pytestmark_pg = pytest.mark.skipif(
    not PG_URL, reason="set CUEPILOT_TEST_DATABASE_URL to run journal integration tests"
)


# ── fakes: prove the queue semantics without a database ────────────────────
class _NullTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.row: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self.fail = False

    async def execute(self, sql: str, *args: Any) -> None:
        if self.fail:
            raise ConnectionError("postgres went away")
        self.executed.append((sql.split()[0].upper(), args))

    def transaction(self) -> _NullTransaction:
        return _NullTransaction()

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        return self.row

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return self.turns


class FakeAcquire:
    def __init__(self, pool: FakePool) -> None:
        self._pool = pool

    async def __aenter__(self) -> FakeConnection:
        return self._pool.conn

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakePool:
    def __init__(self) -> None:
        self.conn = FakeConnection()

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self)


def make_journal(**overrides: Any) -> tuple[TurnJournal, FakePool]:
    pool = FakePool()
    journal = TurnJournal(pool, make_settings(**overrides), Metrics())  # type: ignore[arg-type]
    return journal, pool


def turn(**overrides: Any) -> TurnRecord:
    base: dict[str, Any] = {
        "session_id": "s1",
        "turn_id": "t1",
        "prompt": "why do clients reconnect all at once",
        "answer": "thundering herd",
        "status": "done",
    }
    base.update(overrides)
    return TurnRecord(**base)


# ── queue semantics ────────────────────────────────────────────────────────
async def test_submit_never_blocks_and_drops_when_full() -> None:
    journal, _ = make_journal(session_journal_queue=1)

    assert journal.submit(turn(turn_id="t1")) is True
    assert journal.submit(turn(turn_id="t2")) is False  # queue is full
    assert journal.dropped == 1


async def test_a_dropped_write_is_counted_not_raised() -> None:
    journal, _ = make_journal(session_journal_queue=1)
    journal.submit(turn(turn_id="t1"))
    journal.submit(turn(turn_id="t2"))

    stats = journal.stats()
    assert stats["dropped"] == 1
    assert stats["queued"] == 1
    assert stats["backend"] == "postgres"


async def test_the_worker_writes_queued_jobs() -> None:
    journal, pool = make_journal()
    await journal.start()
    try:
        journal.save_session(
            Session(id="s1", created_at=0.0, updated_at=0.0, turns=1, client={"app": "test"})
        )
        journal.record_turn(turn())
        await journal.stop()  # drains what is queued
    finally:
        await journal.stop()

    statements = [name for name, _ in pool.conn.executed]
    assert statements.count("INSERT") >= 2
    assert journal.written == 2
    assert journal.failed == 0


async def test_a_write_failure_is_counted_and_does_not_kill_the_worker() -> None:
    journal, pool = make_journal()
    pool.conn.fail = True
    await journal.start()
    try:
        journal.record_turn(turn())
        await journal.stop()
    finally:
        await journal.stop()

    assert journal.failed == 1
    assert journal.written == 0


async def test_invalid_status_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="invalid turn status"):
        turn(status="finished")


# ── reads ──────────────────────────────────────────────────────────────────
async def test_load_session_rebuilds_the_transcript() -> None:
    journal, pool = make_journal(turn_history_limit=2)
    pool.conn.row = {
        "id": "s1",
        "created_at": _dt(100.0),
        "updated_at": _dt(200.0),
        "client": {"app": "test"},
        "kb_version": 3,
        "turns": 4,
        "scratchpad": {},
    }
    pool.conn.turns = [
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p2", "answer": "a2"},
        {"prompt": "p3", "answer": ""},
    ]

    record = await journal.load_session("s1")

    assert record is not None
    assert record["kb_version"] == 3
    assert record["client"] == {"app": "test"}
    # Only the newest `turn_history_limit` turns, and an empty answer is skipped.
    assert record["history"] == [
        {"role": "user", "content": "p2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "p3"},
    ]
    assert record["created_at"] == 100.0


async def test_load_session_returns_none_for_an_unknown_session() -> None:
    journal, pool = make_journal()
    pool.conn.row = None
    assert await journal.load_session("nope") is None


def _dt(epoch: float):
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC)


# ── SessionManager integration ─────────────────────────────────────────────
class FakeJournal:
    backend = "postgres"

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []
        self.turns: list[TurnRecord] = []
        self.record: dict[str, Any] | None = None
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def save_session(self, session: Any) -> bool:
        self.saved.append(session.to_record())
        return True

    def record_turn(self, item: TurnRecord) -> bool:
        self.turns.append(item)
        return True

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        return self.record

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend}


async def test_save_mirrors_and_enqueues() -> None:
    cache = MemoryCache()
    journal = FakeJournal()
    manager = SessionManager(cache, make_settings(), journal)  # type: ignore[arg-type]
    session = await manager.create()

    assert await cache.get(f"cuepilot:session:{session.id}") is not None
    assert journal.saved and journal.saved[-1]["id"] == session.id


async def test_a_cache_miss_falls_back_to_the_journal() -> None:
    cache = MemoryCache()
    journal = FakeJournal()
    manager = SessionManager(cache, make_settings(), journal)  # type: ignore[arg-type]
    session = await manager.create()
    manager._hot.clear()
    await cache.delete(f"cuepilot:session:{session.id}")
    journal.record = {
        "id": session.id,
        "created_at": 1.0,
        "updated_at": 2.0,
        "history": [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}],
        "turns": 1,
    }

    recovered = await manager.get(session.id)

    assert recovered is not None
    assert recovered.history[0]["content"] == "earlier"


async def test_a_journal_read_failure_degrades_to_a_new_session() -> None:
    class Broken(FakeJournal):
        async def load_session(self, session_id: str) -> dict[str, Any] | None:
            raise ConnectionError("postgres went away")

    manager = SessionManager(MemoryCache(), make_settings(), Broken())  # type: ignore[arg-type]
    assert await manager.get("anything") is None


async def test_record_turn_without_a_journal_is_a_no_op() -> None:
    manager = SessionManager(MemoryCache(), make_settings())
    session = await manager.create()

    assert manager.record_turn(session, turn_id="t1", prompt="p", answer="a", status="done") is False


async def test_manager_starts_and_stops_the_journal() -> None:
    journal = FakeJournal()
    manager = SessionManager(MemoryCache(), make_settings(), journal)  # type: ignore[arg-type]
    await manager.start()
    assert journal.started
    await manager.stop()
    assert journal.stopped


# ── real Postgres ──────────────────────────────────────────────────────────
@pytestmark_pg
async def test_journal_round_trips_through_postgres() -> None:
    import asyncpg
    from app.infra.migrations import run_migrations

    pool = await asyncpg.create_pool(PG_URL, min_size=1, max_size=2)
    assert pool is not None
    try:
        await run_migrations(pool, params={"embedding_dim": 256})
        await pool.execute("DELETE FROM turns; DELETE FROM sessions")

        journal = TurnJournal(pool, make_settings(), Metrics())
        await journal.start()
        manager = SessionManager(MemoryCache(), make_settings(), journal)
        session = await manager.create(client={"app": "test"})
        manager.record_turn(
            session,
            turn_id="t1",
            prompt="why do clients reconnect all at once",
            answer="thundering herd",
            status="done",
            finish_reason="stop",
            ttft_ms=120.0,
            total_ms=400.0,
            tool_iterations=1,
            usage={"total_tokens": 42},
            model="mock/echo-1",
        )
        await journal.stop()

        record = await journal.load_session(session.id)
        assert record is not None
        assert record["client"] == {"app": "test"}
        assert record["history"] == [
            {"role": "user", "content": "why do clients reconnect all at once"},
            {"role": "assistant", "content": "thundering herd"},
        ]

        turns = await journal.load_turns(session.id)
        assert len(turns) == 1
        assert turns[0]["status"] == "done"
        assert turns[0]["usage"] == {"total_tokens": 42}
        assert turns[0]["model"] == "mock/echo-1"
    finally:
        await pool.execute("DELETE FROM turns; DELETE FROM sessions")
        await pool.close()


@pytestmark_pg
async def test_a_turn_can_land_before_its_session_row() -> None:
    # Under backpressure the session upsert can be dropped while the turn write
    # survives. The FK must not reject a transcript line in that case.
    import asyncpg
    from app.infra.migrations import run_migrations

    pool = await asyncpg.create_pool(PG_URL, min_size=1, max_size=2)
    assert pool is not None
    try:
        await run_migrations(pool, params={"embedding_dim": 256})
        await pool.execute("DELETE FROM turns; DELETE FROM sessions")

        journal = TurnJournal(pool, make_settings(), Metrics())
        await journal.start()
        journal.record_turn(turn(session_id="orphan", turn_id="t9"))
        await journal.stop()

        assert await pool.fetchval("SELECT count(*) FROM turns WHERE session_id = 'orphan'") == 1
        assert await pool.fetchval("SELECT count(*) FROM sessions WHERE id = 'orphan'") == 1
    finally:
        await pool.execute("DELETE FROM turns; DELETE FROM sessions")
        await pool.close()


async def test_concurrent_submits_do_not_raise() -> None:
    journal, _ = make_journal(session_journal_queue=8)
    await journal.start()
    try:
        results = [journal.submit(turn(turn_id=f"t{i}")) for i in range(50)]
        await journal.stop()
    finally:
        await journal.stop()

    assert sum(results) == 8  # queue capacity
    assert journal.dropped == 42
    assert journal.written == 8


async def test_asyncio_gather_style_burst_is_ordered() -> None:
    journal, pool = make_journal()
    await journal.start()
    try:
        for index in range(5):
            journal.submit(turn(turn_id=f"t{index}"))
        await journal.stop()
    finally:
        await journal.stop()

    written = [args[1] for name, args in pool.conn.executed if name == "INSERT" and len(args) > 2]
    assert written == ["t0", "t1", "t2", "t3", "t4"]
