"""Cache abstraction: Redis when available, TTL-aware in-process dict otherwise.

The in-process fallback is not a toy: it implements expiry, atomic-ish counters
and JSON helpers so tests, CI and the offline demo exercise identical code
paths to production.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

import redis.asyncio as aioredis

from app.config import Settings


class Cache(Protocol):
    backend: str

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...
    async def delete(self, *keys: str) -> None: ...
    async def incr(self, key: str, ttl_s: int | None = None) -> int: ...
    async def close(self) -> None: ...


class MemoryCache:
    backend = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    def _expired(self, key: str) -> bool:
        entry = self._data.get(key)
        if entry is None:
            return True
        _, expires_at = entry
        if expires_at is not None and expires_at <= time.monotonic():
            self._data.pop(key, None)
            return True
        return False

    async def get(self, key: str) -> str | None:
        async with self._lock:
            if self._expired(key):
                return None
            return self._data[key][0]

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        async with self._lock:
            expires_at = time.monotonic() + ttl_s if ttl_s else None
            self._data[key] = (value, expires_at)

    async def delete(self, *keys: str) -> None:
        async with self._lock:
            for key in keys:
                self._data.pop(key, None)

    async def incr(self, key: str, ttl_s: int | None = None) -> int:
        async with self._lock:
            current = 0 if self._expired(key) else int(self._data[key][0])
            current += 1
            expires_at = self._data.get(key, (None, None))[1]
            if key not in self._data or expires_at is None:
                expires_at = time.monotonic() + ttl_s if ttl_s else None
            self._data[key] = (str(current), expires_at)
            return current

    async def close(self) -> None:
        self._data.clear()

    async def sweep(self) -> int:
        """Drop expired keys; called periodically by the session sweeper."""
        async with self._lock:
            stale = [k for k in list(self._data) if self._expired(k)]
            for key in stale:
                self._data.pop(key, None)
            return len(stale)


class RedisCache:
    backend = "redis"

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        value = await self._client.get(key)
        return value if isinstance(value, str) else value.decode() if value else None

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        if ttl_s:
            await self._client.set(key, value, ex=ttl_s)
        else:
            await self._client.set(key, value)

    async def delete(self, *keys: str) -> None:
        if keys:
            await self._client.delete(*keys)

    async def incr(self, key: str, ttl_s: int | None = None) -> int:
        value = await self._client.incr(key)
        if ttl_s and value == 1:
            await self._client.expire(key, ttl_s)
        return int(value)

    async def close(self) -> None:
        await self._client.aclose()


async def build_cache(settings: Settings) -> tuple[Cache, str]:
    """Return (cache, degraded_reason). Never raises when degraded mode is on."""
    try:
        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=settings.infra_probe_timeout_s,
            socket_timeout=settings.infra_probe_timeout_s,
        )
        await asyncio.wait_for(client.ping(), timeout=settings.infra_probe_timeout_s)
        return RedisCache(client), ""
    except Exception as exc:  # noqa: BLE001 - any connect failure is degradable
        if not settings.allow_degraded_infra:
            raise
        return MemoryCache(), f"redis unavailable ({type(exc).__name__}: {exc})"


async def cache_get_json(cache: Cache, key: str) -> Any | None:
    raw = await cache.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def cache_set_json(cache: Cache, key: str, value: Any, ttl_s: int | None = None) -> None:
    await cache.set(key, json.dumps(value, separators=(",", ":")), ttl_s)
