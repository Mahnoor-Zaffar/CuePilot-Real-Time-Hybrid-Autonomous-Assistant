"""Cache abstraction: Redis when available, TTL-aware in-process dict otherwise.

The in-process fallback is not a toy: it implements expiry, atomic-ish counters
and JSON helpers so tests, CI and the offline demo exercise identical code
paths to production.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from typing import Any, Protocol

import redis.asyncio as aioredis

from app.config import Settings

# Sliding-window hit: a sorted set of event timestamps. Redis time is used
# inside the script so replicas with skewed clocks still agree on the window.
_WINDOW_HIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', key)
if limit > 0 and count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = 0
  if oldest[2] then retry = tonumber(oldest[2]) + window_ms - now_ms end
  if retry < 0 then retry = 0 end
  return {0, retry}
end
redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms)
return {1, 0}
"""

# Sliding-window sum: members carry `<amount>:<unique>` so the value survives
# the ZSET's single numeric score, which is reserved for the timestamp.
_WINDOW_SUM_LUA = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local items = redis.call('ZRANGE', key, 0, -1)
local total = 0
for _, item in ipairs(items) do
  local amount = string.match(item, '^([^:]+):')
  if amount then total = total + tonumber(amount) end
end
return tostring(total)
"""

_WINDOW_ADD_LUA = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local member = ARGV[3]
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms)
return amount
"""


class Cache(Protocol):
    backend: str

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...
    async def delete(self, *keys: str) -> None: ...
    async def incr(self, key: str, ttl_s: int | None = None) -> int: ...
    async def scan(self, prefix: str) -> list[str]: ...
    async def window_hit(self, key: str, *, limit: int, window_s: float) -> tuple[bool, float]:
        """Record one event in a sliding window.

        Returns `(allowed, retry_after_s)`. A sliding window has no fixed
        boundary to burst across, which is why the limiter uses it instead of
        `incr` on a bucket key.
        """
        ...

    async def window_sum(self, key: str, *, window_s: float) -> float: ...
    async def window_add(self, key: str, *, amount: float, window_s: float) -> None: ...
    async def close(self) -> None: ...


class MemoryCache:
    backend = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}
        # Sliding windows are kept separately: entries are (monotonic, amount)
        # and are pruned on read, so an idle key costs nothing after expiry.
        self._windows: dict[str, deque[tuple[float, float]]] = {}
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

    async def scan(self, prefix: str) -> list[str]:
        async with self._lock:
            return [key for key in self._data if key.startswith(prefix) and not self._expired(key)]

    # sliding windows ------------------------------------------------------
    def _prune(self, key: str, window_s: float, now: float) -> deque[tuple[float, float]]:
        entries = self._windows.get(key)
        if entries is None:
            entries = deque()
            self._windows[key] = entries
        cutoff = now - window_s
        while entries and entries[0][0] < cutoff:
            entries.popleft()
        return entries

    async def window_hit(self, key: str, *, limit: int, window_s: float) -> tuple[bool, float]:
        async with self._lock:
            now = time.monotonic()
            entries = self._prune(key, window_s, now)
            if limit > 0 and len(entries) >= limit:
                retry = max(0.0, entries[0][0] + window_s - now) if entries else window_s
                return False, retry
            entries.append((now, 1.0))
            return True, 0.0

    async def window_sum(self, key: str, *, window_s: float) -> float:
        async with self._lock:
            entries = self._prune(key, window_s, time.monotonic())
            return float(sum(amount for _, amount in entries))

    async def window_add(self, key: str, *, amount: float, window_s: float) -> None:
        async with self._lock:
            now = time.monotonic()
            entries = self._prune(key, window_s, now)
            entries.append((now, float(amount)))

    async def close(self) -> None:
        self._data.clear()
        self._windows.clear()

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

    async def scan(self, prefix: str) -> list[str]:
        keys: list[str] = []
        async for key in self._client.scan_iter(match=f"{prefix}*", count=100):
            keys.append(key if isinstance(key, str) else key.decode())
        return keys

    async def window_hit(self, key: str, *, limit: int, window_s: float) -> tuple[bool, float]:
        allowed, retry_ms = await self._client.eval(
            _WINDOW_HIT_LUA,
            1,
            key,
            str(limit),
            str(max(1, int(window_s * 1000))),
            uuid.uuid4().hex,
        )
        return bool(int(allowed)), float(retry_ms) / 1000

    async def window_sum(self, key: str, *, window_s: float) -> float:
        total = await self._client.eval(
            _WINDOW_SUM_LUA, 1, key, str(max(1, int(window_s * 1000)))
        )
        return float(total)

    async def window_add(self, key: str, *, amount: float, window_s: float) -> None:
        await self._client.eval(
            _WINDOW_ADD_LUA,
            1,
            key,
            str(max(1, int(window_s * 1000))),
            str(amount),
            f"{amount}:{uuid.uuid4().hex}",
        )

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
