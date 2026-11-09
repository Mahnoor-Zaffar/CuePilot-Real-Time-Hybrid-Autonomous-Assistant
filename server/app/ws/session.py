"""Session state: hot in memory, mirrored to Redis for cross-replica resume."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import Settings
from app.infra.cache import Cache, MemoryCache

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
        )


class SessionManager:
    def __init__(self, cache: Cache, settings: Settings) -> None:
        self.cache = cache
        self.settings = settings
        self._hot: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._sweeper: asyncio.Task | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="session-sweeper")

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None

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
        raw = await self.cache.get(self._key(session_id))
        if not raw:
            return None
        try:
            session = Session.from_record(json.loads(raw))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        async with self._lock:
            self._hot[session.id] = session
        return session

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
        session.touch()
        await self.cache.set(
            self._key(session.id),
            json.dumps(session.to_record(), separators=(",", ":")),
            self.settings.session_ttl_s,
        )

    async def drop(self, session_id: str) -> None:
        async with self._lock:
            self._hot.pop(session_id, None)
        await self.cache.delete(self._key(session_id))

    @staticmethod
    def _key(session_id: str) -> str:
        return f"cuepilot:session:{session_id}"

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
