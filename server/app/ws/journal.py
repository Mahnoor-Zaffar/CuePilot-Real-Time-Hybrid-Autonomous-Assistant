"""Durable session and turn journal.

Redis is the fast path: a hot session is one GET, and a resume usually never
touches Postgres. But a cache is not a record. Before this, eviction or a Redis
restart silently ended resume and lost the transcript — the socket kept working
and quietly started a new session, which is the worst kind of failure.

Now the write path is write-behind:

* `save()` still writes Redis (synchronous, so the fast path stays coherent) and
  *enqueues* the Postgres upsert.
* A finished turn is enqueued on `done` and on `abort`, the same two points
  where the terminal frame is emitted.
* One worker drains a **bounded** queue. If Postgres stalls, the queue fills and
  jobs are dropped with a counter — the token path never waits on durability.
  That trade is deliberate: a dropped journal write costs a transcript line, a
  blocked write would cost the stream.

Reads go Redis, then Postgres, then "new session". A miss is a cache miss, not
data loss.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import asyncpg

from app.config import Settings
from app.telemetry.metrics import Metrics

logger = logging.getLogger(__name__)

_UPSERT_SESSION = """
INSERT INTO sessions (id, created_at, updated_at, client, kb_version, turns, scratchpad)
VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::jsonb)
ON CONFLICT (id) DO UPDATE SET
    updated_at = EXCLUDED.updated_at,
    client = EXCLUDED.client,
    kb_version = EXCLUDED.kb_version,
    -- A session that resumed from the journal can legitimately have a lower
    -- in-memory count than the row (its earlier turns were pruned), so never
    -- walk the number backwards.
    turns = GREATEST(sessions.turns, EXCLUDED.turns),
    scratchpad = EXCLUDED.scratchpad
"""

_ENSURE_SESSION = "INSERT INTO sessions (id) VALUES ($1) ON CONFLICT (id) DO NOTHING"

_UPSERT_TURN = """
INSERT INTO turns (
    session_id, turn_id, prompt, answer, status, finish_reason,
    ttft_ms, total_ms, tool_iterations, usage, model
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
ON CONFLICT (session_id, turn_id) DO UPDATE SET
    answer = EXCLUDED.answer,
    status = EXCLUDED.status,
    finish_reason = EXCLUDED.finish_reason,
    ttft_ms = EXCLUDED.ttft_ms,
    total_ms = EXCLUDED.total_ms,
    tool_iterations = EXCLUDED.tool_iterations,
    usage = EXCLUDED.usage,
    model = EXCLUDED.model
"""


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One finished turn. `status` is terminal: done, aborted or error."""

    session_id: str
    turn_id: str
    prompt: str
    answer: str
    status: str
    finish_reason: str | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    tool_iterations: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("done", "aborted", "error"):
            raise ValueError(f"invalid turn status {self.status!r}")


@dataclass(frozen=True, slots=True)
class _SessionWrite:
    record: dict[str, Any]


Job = _SessionWrite | TurnRecord


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _timestamp(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=UTC)


class TurnJournal:
    """Bounded write-behind queue in front of the Postgres journal tables."""

    backend = "postgres"

    def __init__(self, pool: asyncpg.Pool, settings: Settings, metrics: Metrics) -> None:
        self._pool = pool
        self._settings = settings
        self._metrics = metrics
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=settings.session_journal_queue)
        self._worker: asyncio.Task | None = None
        self.written = 0
        self.dropped = 0
        self.failed = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._drain(), name="session-journal")

    async def stop(self) -> None:
        if self._worker is None:
            return
        # Drain what is already queued so a graceful shutdown does not lose the
        # last turn; a hard kill loses at most the queue, which is the point of
        # write-behind.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._queue.join(), timeout=self._settings.session_journal_drain_s)
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None

    # ── writes ─────────────────────────────────────────────────────────────
    def submit(self, job: Job) -> bool:
        """Never blocks. Returns False when the queue is full and the job is dropped."""
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            self._metrics.incr("session_journal_dropped")
            return False

    def save_session(self, session: Any) -> bool:
        return self.submit(_SessionWrite(record=session.to_record()))

    def record_turn(self, turn: TurnRecord) -> bool:
        return self.submit(turn)

    async def _drain(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._write_with_retry(job)
                self.written += 1
                self._metrics.incr("session_journal_written")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the journal must never kill the worker
                self.failed += 1
                self._metrics.incr("session_journal_failed")
                logger.warning("session journal write failed: %s", exc)
            finally:
                self._queue.task_done()

    async def _write_with_retry(self, job: Job, attempts: int = 2) -> None:
        for attempt in range(1, attempts + 1):
            try:
                await asyncio.wait_for(
                    self._write(job), timeout=self._settings.session_journal_timeout_s
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt == attempts:
                    raise
                await asyncio.sleep(0.05 * attempt)

    async def _write(self, job: Job) -> None:
        async with self._pool.acquire() as conn:
            if isinstance(job, _SessionWrite):
                record = job.record
                await conn.execute(
                    _UPSERT_SESSION,
                    record["id"],
                    _timestamp(float(record["created_at"])),
                    _timestamp(float(record["updated_at"])),
                    _json(record.get("client") or {}),
                    int(record.get("kb_version", 1)),
                    int(record.get("turns", 0)),
                    _json(record.get("scratchpad") or {}),
                )
                return
            async with conn.transaction():
                # A turn can be written before its session row if the session
                # upsert was dropped under backpressure. The FK would reject it,
                # and the transcript line is worth more than the ordering.
                await conn.execute(_ENSURE_SESSION, job.session_id)
                await conn.execute(
                    _UPSERT_TURN,
                    job.session_id,
                    job.turn_id,
                    job.prompt,
                    job.answer,
                    job.status,
                    job.finish_reason,
                    job.ttft_ms,
                    job.total_ms,
                    job.tool_iterations,
                    _json(job.usage),
                    job.model,
                )

    # ── reads ──────────────────────────────────────────────────────────────
    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Rebuild a session record (header + recent transcript) from Postgres."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, created_at, updated_at, client, kb_version, turns, scratchpad "
                "FROM sessions WHERE id = $1",
                session_id,
            )
            if row is None:
                return None
            turns = await conn.fetch(
                "SELECT prompt, answer FROM turns WHERE session_id = $1 "
                "ORDER BY created_at ASC, turn_id ASC",
                session_id,
            )
        history: list[dict[str, Any]] = []
        for turn in turns[-self._settings.turn_history_limit :]:
            history.append({"role": "user", "content": turn["prompt"]})
            if turn["answer"]:
                history.append({"role": "assistant", "content": turn["answer"]})
        return {
            "id": row["id"],
            "created_at": row["created_at"].timestamp(),
            "updated_at": row["updated_at"].timestamp(),
            "history": history,
            "scratchpad": _loads(row["scratchpad"]),
            "kb_version": int(row["kb_version"]),
            "turns": int(row["turns"]),
            "client": _loads(row["client"]),
        }

    async def load_turns(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """The transcript, newest last. Used by the API and by tests."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT turn_id, prompt, answer, status, finish_reason, ttft_ms, total_ms, "
                "tool_iterations, usage, model, created_at FROM turns "
                "WHERE session_id = $1 ORDER BY created_at DESC, turn_id DESC LIMIT $2",
                session_id,
                limit,
            )
        return [
            {
                "turn_id": row["turn_id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "status": row["status"],
                "finish_reason": row["finish_reason"],
                "ttft_ms": row["ttft_ms"],
                "total_ms": row["total_ms"],
                "tool_iterations": row["tool_iterations"],
                "usage": _loads(row["usage"]),
                "model": row["model"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in reversed(rows)
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "queued": self._queue.qsize(),
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
        }


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}
