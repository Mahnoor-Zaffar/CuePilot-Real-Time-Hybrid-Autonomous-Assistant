"""Per-client connection limits.

Two independent limits, because they fail in different ways:

* A **shared attempt counter** in the cache caps how often one client may open a
  socket in a window. Redis makes the counter cross-replica; the in-process
  fallback makes it per-replica, which is still useful and is reported by
  `cache.backend`.
* A **per-replica connection cap** protects this process's memory and task
  count. `ws_active` is a process gauge, so this cap is intentionally local.

The limiter fails **open** if the cache raises: a broken limiter must not take
the gateway down with it. The failure is counted (`rate_limit_cache_errors`) and
health still reports the configured limits, so it is visible rather than silent.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.infra.cache import Cache
from app.telemetry.metrics import Metrics

if TYPE_CHECKING:
    from fastapi import WebSocket

    from app.config import Settings

logger = logging.getLogger(__name__)

CLOSE_RATE_LIMITED = 4429


@dataclass(frozen=True, slots=True)
class LimitDecision:
    allowed: bool
    code: str | None = None
    message: str = ""
    retry_after_ms: int | None = None
    # True when the limiter reserved a connection slot. The caller must hand it
    # to `release_reservation()` once the socket is accepted (or on failure).
    reserved: bool = False


ALLOW = LimitDecision(True)


def client_identity(websocket: WebSocket, settings: Settings) -> str:
    """Stable-ish client key for the shared attempt counter.

    The socket peer is the default. Behind a trusted proxy the left-most
    `X-Forwarded-For` hop is the client, but it is only read when the operator
    says the proxy is trusted: an untrusted header is just a spoofable key.
    """
    if settings.rate_limit_trust_forwarded_for:
        forwarded = websocket.headers.get("x-forwarded-for")
        if forwarded:
            hop = forwarded.split(",")[0].strip()
            if hop:
                return hop[:128]
    client = websocket.client
    return (client.host if client else "unknown")[:128]


class ClientLimiter:
    backend = "cache"

    def __init__(self, cache: Cache, settings: Settings, metrics: Metrics) -> None:
        self.cache = cache
        self.settings = settings
        self.metrics = metrics
        # Admitted connections whose socket has not been accepted yet. Reserving
        # synchronously closes the race between the cap check and the gauge
        # increment, which both live behind an await.
        self._reserved = 0

    @property
    def reserved(self) -> int:
        return self._reserved

    def release_reservation(self) -> None:
        if self._reserved > 0:
            self._reserved -= 1

    async def admit(self, identity: str) -> LimitDecision:
        if not self.settings.rate_limit_enabled:
            return ALLOW

        cap = self.settings.rate_limit_max_connections
        if cap > 0 and self.metrics.gauges.get("ws_active", 0) + self._reserved >= cap:
            self.metrics.incr("ws_rejected_connection_cap")
            return LimitDecision(
                False,
                code="connection_cap",
                message=f"this replica is at its {cap}-connection cap",
                retry_after_ms=1000,
            )
        # Take the slot before the first await so a burst cannot over-admit.
        self._reserved += 1

        window_s = max(1, self.settings.rate_limit_window_s)
        attempts = max(1, self.settings.rate_limit_connect_attempts)
        now = time.time()
        bucket = int(now // window_s)
        key = f"cuepilot:ratelimit:connect:{identity}:{bucket}"
        try:
            count = await self.cache.incr(key, ttl_s=window_s * 2)
        except Exception as exc:  # noqa: BLE001 - fail open, but count it
            self.metrics.incr("rate_limit_cache_errors")
            logger.warning("rate limiter cache failed, allowing connection: %s", exc)
            # Fail open means keep the reservation: the caller still owns the slot.
            return LimitDecision(True, reserved=True)

        if count > attempts:
            self.release_reservation()
            retry_s = max(0.0, (bucket + 1) * window_s - now)
            self.metrics.incr("ws_rejected_rate_limit")
            return LimitDecision(
                False,
                code="client_rate_limited",
                message=f"more than {attempts} connections in {window_s}s",
                retry_after_ms=max(100, int(math.ceil(retry_s * 1000))),
            )
        return LimitDecision(True, reserved=True)

    def stats(self) -> dict[str, object]:
        return {
            "enabled": self.settings.rate_limit_enabled,
            "backend": self.cache.backend,
            "window_s": self.settings.rate_limit_window_s,
            "connect_attempts": self.settings.rate_limit_connect_attempts,
            "max_connections": self.settings.rate_limit_max_connections,
            "trust_forwarded_for": self.settings.rate_limit_trust_forwarded_for,
        }
