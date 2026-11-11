"""Durable session and turn journal.

Redis is the fast path: a hot session is one GET, and a resume usually never
touches Postgres. But a cache is not a record. Before this, eviction or a Redis
restart silently ended resume and lost the transcript — the socket kept working
and quietly started a new session, which is the worst kind of failure.

Now the write path is write-behind:

* `save()` still writes Redis (synchronous, so the fast path stays coherent) and
  *enqueues* the Postgres upsert.
* A streaming turn is checkpointed every `turn_checkpoint_interval_ms`, so a
  hard crash loses at most one interval of text instead of the whole answer.
  Checkpoints are best-effort and may be dropped under backpressure.
* A **terminal** turn (`done`, `aborted`, `error`) is enqueued on the same points
  where the terminal frame is emitted. It is the transcript, so it is never
  silently dropped: if the bounded queue is full or the write fails, the record
  is appended to a small on-disk spool and replayed later.
* One worker drains the queue. If Postgres stalls, the token path still never
  waits on durability; checkpoints are shed and terminal turns spool.

Reads go Redis, then Postgres, then "new session". A miss is a cache miss, not
data loss. Stale `streaming` rows (a turn whose replica stopped checkpointing)
are marked `aborted`/`interrupted` when the session is read, and history replay
flags the partial turn so the client can show it honestly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import asyncpg

from app.config import Settings
from app.telemetry.metrics import Metrics

if TYPE_CHECKING:
    from app.infra.cache import Cache

logger = logging.getLogger(__name__)

# Cache-backed spool and in-flight checkpoint keys. Both live beside the session
# mirror so every replica sees the same fallback and the same partial turn.
CACHE_SPOOL_PREFIX = "cuepilot:spool:turn:"
INFLIGHT_PREFIX = "cuepilot:inflight:"

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
    model = EXCLUDED.model,
    -- Checkpoints move forward; the terminal write is the last update.
    updated_at = now()
"""

_MARK_STALE_TURNS = """
UPDATE turns
SET status = 'aborted', finish_reason = 'interrupted', updated_at = now()
WHERE session_id = $1 AND status = 'streaming' AND updated_at < $2
"""


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One turn write.

    `streaming` is a checkpoint: the answer so far, best-effort and droppable.
    `done`, `aborted` and `error` are terminal, and a terminal record is the
    transcript — it spools rather than disappearing when Postgres is unavailable.
    """

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
        if self.status not in ("streaming", "done", "aborted", "error"):
            raise ValueError(f"invalid turn status {self.status!r}")

    @property
    def terminal(self) -> bool:
        return self.status != "streaming"


@dataclass(frozen=True, slots=True)
class _SessionWrite:
    record: dict[str, Any]


Job = _SessionWrite | TurnRecord


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _timestamp(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=UTC)


class JournalSpool:
    """Append-only NDJSON spill for terminal turn writes.

    The queue is bounded so a slow database cannot stall the token path. That is
    the right trade for checkpoints and session headers, but a terminal turn is
    the transcript: dropping it is data loss. When the queue is full or a write
    fails, the record lands here and is replayed in the background.

    The spool is local disk, not a second database. It survives a process crash
    and a Postgres outage; it does not survive losing the container filesystem.
    It is bounded (`session_journal_spool_max_bytes`) so a long outage cannot
    fill the disk. When the local spool is full or unwritable, `TurnJournal`
    falls back to the shared cache (`session_journal_spool_cache_ttl_s`), so a
    lost container filesystem does not automatically mean a lost transcript.
    Both limits are reported in health and the failure docs.
    """

    def __init__(self, path: str | Path | None, settings: Settings, metrics: Metrics) -> None:
        self.path = Path(path) if path else None
        self.replay_path = (
            self.path.with_name(f"{self.path.name}.replay") if self.path is not None else None
        )
        self.max_bytes = settings.session_journal_spool_max_bytes
        self._metrics = metrics
        self.spooled = 0
        self.replayed = 0
        self.errors = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def append(self, turn: TurnRecord) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(turn), separators=(",", ":"), default=str) + "\n"
            with self._lock:
                if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                    self._metrics.incr("session_journal_spool_full")
                    return False
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
            self.spooled += 1
            self._metrics.incr("session_journal_spooled")
            return True
        except OSError as exc:
            self.errors += 1
            self._metrics.incr("session_journal_spool_errors")
            logger.warning("journal spool append failed: %s", exc)
            return False

    def rotate(self) -> Path | None:
        """Hand the current spool to the replayer without racing new appends.

        New records keep going to `path`; the replayer owns `.replay`. If a
        previous replay was interrupted, its file is returned first so it is
        not orphaned.
        """
        if self.path is None or self.replay_path is None:
            return None
        try:
            with self._lock:
                if self.replay_path.exists():
                    return self.replay_path
                if not self.path.exists() or self.path.stat().st_size == 0:
                    return None
                os.replace(self.path, self.replay_path)
                return self.replay_path
        except OSError as exc:
            self.errors += 1
            self._metrics.incr("session_journal_spool_errors")
            logger.warning("journal spool rotate failed: %s", exc)
            return None

    def read(self, path: Path) -> list[TurnRecord]:
        records: list[TurnRecord] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(TurnRecord(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        self.errors += 1
                        self._metrics.incr("session_journal_spool_errors")
                        logger.warning("skipping unreadable spool record: %s", exc)
        except OSError as exc:
            self.errors += 1
            self._metrics.incr("session_journal_spool_errors")
            logger.warning("journal spool read failed: %s", exc)
        return records

    def remove(self, path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path is not None else None,
            "spooled": self.spooled,
            "replayed": self.replayed,
            "errors": self.errors,
        }


class TurnJournal:
    """Bounded write-behind queue in front of the Postgres journal tables."""

    backend = "postgres"

    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        metrics: Metrics,
        cache: Cache | None = None,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._metrics = metrics
        self._cache = cache
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=settings.session_journal_queue)
        self._worker: asyncio.Task | None = None
        self._spool_task: asyncio.Task | None = None
        self._replay_lock = asyncio.Lock()
        self._spool = JournalSpool(settings.session_journal_spool_path or None, settings, metrics)
        self._durability_degraded = False
        self.written = 0
        self.dropped = 0
        self.failed = 0
        self.cache_spooled = 0
        self.cache_replayed = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._drain(), name="session-journal")
        if (self._spool.enabled or self._cache is not None) and self._spool_task is None:
            self._spool_task = asyncio.create_task(self._spool_loop(), name="journal-spool")

    async def stop(self) -> None:
        if self._worker is None:
            return
        # Drain what is already queued so a graceful shutdown does not lose the
        # last turn; a hard kill loses at most the queue, which is the point of
        # write-behind.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._queue.join(), timeout=self._settings.session_journal_drain_s)
        # Give the spool one bounded chance too: a graceful stop is the last
        # point at which a terminal write can be replayed without a restart.
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(
                self.replay_spool(), timeout=self._settings.session_journal_drain_s
            )
        if self._spool_task is not None:
            self._spool_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._spool_task
            self._spool_task = None
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None

    # ── writes ─────────────────────────────────────────────────────────────
    @property
    def durability_degraded(self) -> bool:
        """True when a terminal write could neither be queued nor spooled.

        New prompts are refused with a retryable `journal_unavailable` until a
        write or replay succeeds. Refusing is more honest than accepting a turn
        whose transcript is guaranteed to be lost.
        """
        return self._durability_degraded

    def submit(self, job: Job) -> bool:
        """Never blocks. Terminal turns spool to local disk when the queue is full."""
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            if isinstance(job, TurnRecord) and job.terminal and self._spool.append(job):
                return True
            self.dropped += 1
            self._metrics.incr("session_journal_dropped")
            return False

    def save_session(self, session: Any) -> bool:
        return self.submit(_SessionWrite(record=session.to_record()))

    def record_turn(self, turn: TurnRecord) -> bool:
        """Synchronous enqueue, used for droppable checkpoints."""
        return self.submit(turn)

    async def record_terminal(self, turn: TurnRecord) -> bool:
        """Enqueue a terminal turn, falling back to disk then the shared cache.

        The local spool is tried first because it is cheap and survives a process
        crash. The cache is the second line: it survives losing the container
        filesystem, at the cost of depending on the cache's own persistence.
        """
        try:
            self._queue.put_nowait(turn)
            return True
        except asyncio.QueueFull:
            return await self._spool_terminal(turn)

    async def _spool_terminal(self, turn: TurnRecord) -> bool:
        if await asyncio.to_thread(self._spool.append, turn):
            return True
        if await self._cache_append(turn):
            return True
        self._mark_degraded("terminal turn could not be spooled to disk or cache")
        self.dropped += 1
        self._metrics.incr("session_journal_dropped")
        return False

    async def _cache_append(self, turn: TurnRecord) -> bool:
        if self._cache is None:
            return False
        key = f"{CACHE_SPOOL_PREFIX}{time.time_ns()}:{uuid.uuid4().hex[:8]}"
        try:
            await self._cache.set(
                key,
                json.dumps(asdict(turn), separators=(",", ":"), default=str),
                self._settings.session_journal_spool_cache_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - cache spool is a fallback
            self._metrics.incr("session_journal_cache_spool_errors")
            logger.warning("cache spool append failed: %s", exc)
            return False
        self.cache_spooled += 1
        self._metrics.incr("session_journal_cache_spooled")
        return True

    async def _clear_inflight(self, session_id: str) -> None:
        if self._cache is None:
            return
        with contextlib.suppress(Exception):
            await self._cache.delete(f"{INFLIGHT_PREFIX}{session_id}")

    def _mark_degraded(self, reason: str) -> None:
        if not self._durability_degraded:
            logger.error("journal durability degraded: %s", reason)
        self._durability_degraded = True
        self._metrics.incr("session_journal_durability_degraded")

    def _note_durable(self) -> None:
        self._durability_degraded = False

    async def _drain(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._write_with_retry(job)
                self.written += 1
                self._metrics.incr("session_journal_written")
                self._note_durable()
                if isinstance(job, TurnRecord) and job.terminal:
                    # The journal now holds the terminal turn, so the partial
                    # checkpoint is no longer needed for recovery.
                    await self._clear_inflight(job.session_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the journal must never kill the worker
                self.failed += 1
                self._metrics.incr("session_journal_failed")
                if isinstance(job, TurnRecord) and job.terminal:
                    if await self._spool_terminal(job):
                        logger.warning(
                            "session journal write failed; terminal turn spooled: %s", exc
                        )
                    else:
                        logger.error(
                            "session journal write failed and no spool is available: %s", exc
                        )
                else:
                    logger.warning("session journal write failed: %s", exc)
            finally:
                self._queue.task_done()

    async def replay_spool(self) -> int:
        """Try every spooled terminal turn once. Returns how many were written."""
        async with self._replay_lock:
            written = await self._replay_spool_locked()
            written += await self._replay_cache_spool_locked()
            if written:
                self._note_durable()
            return written

    async def _replay_spool_locked(self) -> int:
        replay_path = await asyncio.to_thread(self._spool.rotate)
        if replay_path is None:
            return 0
        records = await asyncio.to_thread(self._spool.read, replay_path)
        remaining: list[TurnRecord] = []
        written = 0
        for index, record in enumerate(records):
            try:
                await self._write_with_retry(record)
            except asyncio.CancelledError:
                # Keep the unprocessed tail for the next replay, then let the
                # cancellation unwind. The rotated file is not removed.
                remaining.extend(records[index:])
                for item in remaining:
                    self._spool.append(item)
                await asyncio.to_thread(self._spool.remove, replay_path)
                raise
            except Exception as exc:  # noqa: BLE001 - replay is best effort
                remaining.append(record)
                logger.warning("spooled turn replay failed: %s", exc)
            else:
                written += 1
                self.written += 1
                self._spool.replayed += 1
                self._metrics.incr("session_journal_written")
                self._metrics.incr("session_journal_replayed")
                await self._clear_inflight(record.session_id)
        for record in remaining:
            self._spool.append(record)
        await asyncio.to_thread(self._spool.remove, replay_path)
        return written

    async def _replay_cache_spool_locked(self) -> int:
        if self._cache is None:
            return 0
        try:
            keys = sorted(await self._cache.scan(CACHE_SPOOL_PREFIX))
        except Exception as exc:  # noqa: BLE001 - replay is best effort
            self._metrics.incr("session_journal_cache_spool_errors")
            logger.warning("cache spool scan failed: %s", exc)
            return 0
        written = 0
        for key in keys:
            try:
                raw = await self._cache.get(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache spool read failed: %s", exc)
                continue
            if not raw:
                continue
            try:
                record = TurnRecord(**json.loads(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                # A poison record would otherwise be retried forever.
                self._metrics.incr("session_journal_cache_spool_errors")
                logger.warning("dropping unreadable cache spool record: %s", exc)
                await self._cache.delete(key)
                continue
            try:
                await self._write_with_retry(record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - leave the key for the next pass
                logger.warning("cache-spooled turn replay failed: %s", exc)
                continue
            written += 1
            self.written += 1
            self.cache_replayed += 1
            self._metrics.incr("session_journal_written")
            self._metrics.incr("session_journal_cache_replayed")
            await self._cache.delete(key)
            await self._clear_inflight(record.session_id)
        return written

    async def _spool_loop(self) -> None:
        while True:
            try:
                await self.replay_spool()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - replay must not kill the worker
                logger.exception("journal spool replay failed")
            await asyncio.sleep(max(1.0, self._settings.session_journal_spool_retry_s))

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
        """Rebuild a session record (header + recent transcript) from Postgres.

        A `streaming` row older than `turn_checkpoint_stale_s` is a turn whose
        replica stopped checkpointing: it is marked interrupted here, on the read
        path, so recovery needs no separate janitor. A fresh streaming row is
        left alone in case another replica is still producing it.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                _MARK_STALE_TURNS,
                session_id,
                _timestamp(time.time() - self._settings.turn_checkpoint_stale_s),
            )
            row = await conn.fetchrow(
                "SELECT id, created_at, updated_at, client, kb_version, turns, scratchpad "
                "FROM sessions WHERE id = $1",
                session_id,
            )
            if row is None:
                return None
            turns = await conn.fetch(
                "SELECT turn_id, prompt, answer, status, finish_reason, updated_at FROM turns "
                "WHERE session_id = $1 ORDER BY created_at ASC, turn_id ASC",
                session_id,
            )
        history: list[dict[str, Any]] = []
        for turn in turns[-self._settings.turn_history_limit :]:
            interrupted = turn["status"] == "streaming" or turn["finish_reason"] == "interrupted"
            user_message: dict[str, Any] = {"role": "user", "content": turn["prompt"]}
            if interrupted:
                user_message["interrupted"] = True
            history.append(user_message)
            if turn["answer"]:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn["answer"],
                }
                if interrupted:
                    assistant_message["interrupted"] = True
                    assistant_message["partial"] = True
                    turn_id = turn.get("turn_id")
                    if turn_id:
                        assistant_message["turn_id"] = turn_id
                    updated_at = turn.get("updated_at")
                    if updated_at is not None:
                        assistant_message["checkpointed_at"] = updated_at.timestamp()
                history.append(assistant_message)
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
                "tool_iterations, usage, model, created_at, updated_at FROM turns "
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
                "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
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
            "spool": self._spool.stats(),
            "durability": {
                "degraded": self._durability_degraded,
                "cache_spool": {
                    "backend": self._cache.backend if self._cache is not None else None,
                    "shared": bool(self._cache is not None and self._cache.backend == "redis"),
                    "spooled": self.cache_spooled,
                    "replayed": self.cache_replayed,
                    "ttl_s": self._settings.session_journal_spool_cache_ttl_s,
                },
                "note": (
                    "terminal writes try the local spool first, then the shared cache; "
                    "cache durability depends on Redis persistence"
                ),
            },
        }


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}
