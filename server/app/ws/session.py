"""Session state: hot in memory, mirrored to Redis, journaled to Postgres."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from app.config import Settings
from app.infra.cache import Cache, MemoryCache

if TYPE_CHECKING:
    from app.ws.journal import TurnJournal

logger = logging.getLogger(__name__)


@dataclass
class AudioBuffer:
    sample_rate: int = 16000
    max_ms: int = 3000
    pcm: bytearray = field(default_factory=bytearray)
    last_seq: int = -1
    dropped_frames: int = 0

    @property
    def max_bytes(self) -> int:
        return int(self.sample_rate * 2 * self.max_ms / 1000)

    def append(self, seq: int, data: bytes, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.last_seq = seq
        self.pcm.extend(data)
        overflow = len(self.pcm) - self.max_bytes
        if overflow > 0:
            del self.pcm[:overflow]
        if seq and seq % 25 == 0:
            self.dropped_frames += 1  # monotonic marker for the level meter

    @property
    def buffered_ms(self) -> int:
        return int(len(self.pcm) / (self.sample_rate * 2) * 1000)

    def rms(self) -> float:
        if not self.pcm:
            return 0.0
        samples = np.frombuffer(bytes(self.pcm), dtype="<i2").astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0

    def take(self) -> bytes:
        data = bytes(self.pcm)
        keep = self.max_bytes // 3  # trailing overlap so words are not cut
        self.pcm = bytearray(self.pcm[-keep:]) if keep else bytearray()
        return data

    def clear(self) -> None:
        self.pcm.clear()
        self.last_seq = -1


@dataclass
class TurnHandle:
    turn_id: str
    started_at: float = field(default_factory=time.perf_counter)
    task: asyncio.Task | None = None


@dataclass
class Session:
    id: str
    created_at: float
    updated_at: float
    history: list[dict[str, Any]] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    kb_version: int = 1
    turns: int = 0
    client: dict[str, Any] = field(default_factory=dict)
    audio: AudioBuffer = field(default_factory=AudioBuffer)
    active_turn: TurnHandle | None = None
    # Set while a turn is streaming so a resume knows the Redis mirror may be
    # behind the journal and reads Postgres instead. Cleared on the terminal path.
    pending_turn: str | None = None

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history[-64:],
            "scratchpad": self.scratchpad,
            "kb_version": self.kb_version,
            "turns": self.turns,
            "client": self.client,
            "pending_turn": self.pending_turn,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Session:
        return cls(
            id=record["id"],
            created_at=float(record.get("created_at", time.time())),
            updated_at=float(record.get("updated_at", time.time())),
            history=list(record.get("history") or []),
            scratchpad=dict(record.get("scratchpad") or {}),
            kb_version=int(record.get("kb_version", 1)),
            turns=int(record.get("turns", 0)),
            client=dict(record.get("client") or {}),
            pending_turn=record.get("pending_turn"),
        )


class SessionManager:
    """Hot dict -> Redis -> Postgres.

    Redis is the fast path and Postgres is the record. The journal is optional:
    with no Postgres there is nothing durable to write to, and the manager
    behaves exactly as before (mirror-only).
    """

    def __init__(self, cache: Cache, settings: Settings, journal: TurnJournal | None = None) -> None:
        self.cache = cache
        self.settings = settings
        self.journal = journal
        self._hot: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._sweeper: asyncio.Task | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self.journal is not None:
            await self.journal.start()
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="session-sweeper")

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        if self.journal is not None:
            await self.journal.stop()

    @property
    def active_sessions(self) -> int:
        return len(self._hot)

    # ── access ─────────────────────────────────────────────────────────────
    async def create(self, client: dict[str, Any] | None = None) -> Session:
        session = Session(
            id=uuid.uuid4().hex,
            created_at=time.time(),
            updated_at=time.time(),
            client=client or {},
        )
        async with self._lock:
            self._hot[session.id] = session
        await self.save(session)
        return session

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            session = self._hot.get(session_id)
        if session is not None:
            return session

        record = await self._load_record(session_id)
        if record is None:
            return None
        try:
            session = Session.from_record(record)
        except (KeyError, TypeError, ValueError):
            logger.warning("ignoring unreadable session record for %s", session_id)
            return None
        async with self._lock:
            self._hot[session.id] = session
        return session

    async def _load_record(self, session_id: str) -> dict[str, Any] | None:
        """Redis first (hot mirror), then the journal (the record).

        A mirror with `pending_turn` was written while a turn was streaming, so
        it may be behind the checkpointed journal. In that one case the journal
        is authoritative and is read even though Redis has a hit.

        The in-flight key is merged last: it carries the newest partial text even
        when the journal checkpoint is one interval behind, and it is timestamped
        so the client can say how old the partial is.
        """
        raw = await self.cache.get(self._key(session_id))
        record: dict[str, Any] | None = None
        if raw:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("corrupt session mirror for %s, trying the journal", session_id)
                record = None
            if isinstance(record, dict) and record.get("pending_turn") and self.journal is not None:
                recovered = await self._load_journal(session_id)
                if recovered is not None:
                    logger.info(
                        "session %s recovered from the journal (in-flight turn)", session_id
                    )
                    record = recovered
        if not isinstance(record, dict):
            record = await self._load_journal(session_id)
        if record is not None:
            inflight = await self._load_inflight(session_id)
            if inflight is not None:
                record = self._merge_inflight(record, inflight)
        return record

    async def _load_inflight(self, session_id: str) -> dict[str, Any] | None:
        try:
            raw = await self.cache.get(self._inflight_key(session_id))
        except Exception as exc:  # noqa: BLE001 - recovery is best effort
            logger.warning("inflight checkpoint read failed for %s: %s", session_id, exc)
            return None
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _merge_inflight(
        record: dict[str, Any], inflight: dict[str, Any]
    ) -> dict[str, Any]:
        """Append a partial assistant turn unless the journal already has it.

        The journal clears the in-flight key after the terminal write, but a crash
        between the write and the clear can leave both. A completed turn always
        ends with an assistant message answering the same prompt, so that is the
        duplicate check.
        """
        prompt = str(inflight.get("prompt") or "")
        answer = str(inflight.get("answer") or "")
        if not prompt:
            return record
        history = list(record.get("history") or [])
        last_user = next(
            (message for message in reversed(history) if message.get("role") == "user"), None
        )
        if (
            history
            and history[-1].get("role") == "assistant"
            and last_user is not None
            and last_user.get("content") == prompt
        ):
            return record
        if answer:
            history.append(
                {
                    "role": "user",
                    "content": prompt,
                    "interrupted": True,
                    "turn_id": inflight.get("turn_id"),
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "interrupted": True,
                    "partial": True,
                    "turn_id": inflight.get("turn_id"),
                    "checkpointed_at": inflight.get("checkpointed_at"),
                }
            )
        return {**record, "history": history, "pending_turn": None}

    async def save_inflight(
        self, session_id: str, *, turn_id: str | None = None, **fields: Any
    ) -> None:
        """Persist the newest partial answer for crash recovery.

        This is a cache write, not a database write: it is cheap enough to run on
        the checkpoint cadence and it survives losing the replica that was
        streaming. It is deleted by the journal after the terminal write lands.
        """
        if not turn_id:
            return
        payload = {
            "turn_id": turn_id,
            "prompt": fields.get("prompt") or "",
            "answer": fields.get("answer") or "",
            "checkpointed_at": time.time(),
        }
        try:
            await self.cache.set(
                self._inflight_key(session_id),
                json.dumps(payload, separators=(",", ":")),
                self.settings.session_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - a missed checkpoint is not fatal
            logger.warning("inflight checkpoint write failed for %s: %s", session_id, exc)

    async def clear_inflight(self, session_id: str) -> None:
        with contextlib.suppress(Exception):
            await self.cache.delete(self._inflight_key(session_id))

    async def _load_journal(self, session_id: str) -> dict[str, Any] | None:
        if self.journal is None:
            return None
        try:
            record = await self.journal.load_session(session_id)
        except Exception as exc:  # noqa: BLE001 - a read failure must not kill the socket
            logger.warning("session journal read failed for %s: %s", session_id, exc)
            return None
        if record is not None:
            logger.info("session %s recovered from the journal", session_id)
        return record

    async def resume_or_create(
        self, session_id: str | None, client: dict[str, Any] | None = None
    ) -> tuple[Session, bool]:
        if session_id:
            existing = await self.get(session_id)
            if existing is not None:
                existing.touch()
                existing.client = client or existing.client
                return existing, True
        return await self.create(client), False

    async def save(self, session: Session) -> None:
        """Write Redis synchronously (fast path), enqueue Postgres (the record)."""
        session.touch()
        await self.cache.set(
            self._key(session.id),
            json.dumps(session.to_record(), separators=(",", ":")),
            self.settings.session_ttl_s,
        )
        if self.journal is not None:
            self.journal.save_session(session)

    async def record_turn(self, session: Session, **fields: Any) -> bool:
        """Enqueue one finished turn. Returns False if there is nowhere durable to put it.

        Called from the turn's terminal path, so it must not block on Postgres:
        a slow journal may cost a transcript line, never the stream. Terminal
        turns try the local spool, then the shared cache, before they are lost.
        """
        if self.journal is None:
            return False
        from app.ws.journal import TurnRecord

        return await self.journal.record_terminal(TurnRecord(session_id=session.id, **fields))

    async def checkpoint_turn(self, session: Session, **fields: Any) -> bool:
        """Best-effort partial-turn checkpoint while tokens are streaming.

        Two destinations: the journal (durable, droppable) and the in-flight key
        (cache, one write per interval, visible to every replica). The next
        checkpoint carries the whole answer so far, so a dropped journal write
        costs at most one interval of crash protection, never the stream.
        """
        queued = False
        if self.journal is not None:
            from app.ws.journal import TurnRecord

            queued = self.journal.record_turn(
                TurnRecord(session_id=session.id, status="streaming", **fields)
            )
        await self.save_inflight(session.id, **fields)
        return queued

    async def drop(self, session_id: str) -> None:
        async with self._lock:
            self._hot.pop(session_id, None)
        await self.cache.delete(self._key(session_id))
        await self.clear_inflight(session_id)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"cuepilot:session:{session_id}"

    @staticmethod
    def _inflight_key(session_id: str) -> str:
        return f"cuepilot:inflight:{session_id}"

    # ── housekeeping ───────────────────────────────────────────────────────
    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self.sweep()
            except Exception:  # noqa: BLE001 - a sweeper must never die
                logger.exception("session sweep failed")

    async def sweep(self) -> int:
        cutoff = time.time() - self.settings.session_ttl_s
        async with self._lock:
            stale = [
                sid
                for sid, session in self._hot.items()
                if session.updated_at < cutoff and session.active_turn is None
            ]
            for sid in stale:
                self._hot.pop(sid, None)
        if isinstance(self.cache, MemoryCache):
            await self.cache.sweep()
        return len(stale)
