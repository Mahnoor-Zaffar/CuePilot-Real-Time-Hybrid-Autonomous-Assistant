"""Per-client connection limits and per-session turn/token budgets.

Three independent limits, because they fail in different ways:

* A **shared connection counter** in the cache caps how often one client may open
  a socket. It is a *sliding* window (sorted set / monotonic deque), not a fixed
  bucket, so there is no window boundary to burst across. Redis makes it
  cross-replica; the in-process fallback makes it per-replica, which is still
  useful and is reported by `cache.backend`.
* A **per-replica connection cap** protects this process's memory and task
  count. `ws_active` is a process gauge, so this cap is intentionally local.
* A **per-session turn and token budget** bounds how much one conversation may
  spend. Turns are checked at admission, before the provider is called. Tokens
  are recorded from the completed turn's usage, so the token budget is post-hoc
  and a turn may overshoot by at most one `llm_max_tokens`.

Identity defaults to the socket peer. Behind a trusted proxy, either trust the
left-most `X-Forwarded-For` hop or name a header the proxy rewrites; an
untrusted header is a spoofable key, not identity.

Every limiter fails **open** if the cache raises: a broken limiter must not take
the gateway down with it. The failure is counted (`rate_limit_cache_errors`) and
health still reports the configured limits, so it is visible rather than silent.
"""

from __future__ import annotations

import logging
import math
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
    """Stable-ish client key for the shared connection counter.

    Precedence: a trusted identity header, then a trusted proxy hop, then the
    socket peer. The header is only read when the operator names it, because a
    header the client can set itself is just a spoofable key.
    """
    header = settings.rate_limit_identity_header
    if header:
        supplied = websocket.headers.get(header.lower())
        if supplied:
            return supplied.strip()[:128]
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
        key = f"cuepilot:ratelimit:connect:{identity}"
        try:
            allowed, retry_s = await self.cache.window_hit(
                key, limit=attempts, window_s=window_s
            )
        except Exception as exc:  # noqa: BLE001 - fail open, but count it
            self.metrics.incr("rate_limit_cache_errors")
            logger.warning("rate limiter cache failed, allowing connection: %s", exc)
            # Fail open means keep the reservation: the caller still owns the slot.
            return LimitDecision(True, reserved=True)

        if not allowed:
            self.release_reservation()
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
            "algorithm": "sliding_window",
            "window_s": self.settings.rate_limit_window_s,
            "connect_attempts": self.settings.rate_limit_connect_attempts,
            "max_connections": self.settings.rate_limit_max_connections,
            "trust_forwarded_for": self.settings.rate_limit_trust_forwarded_for,
            "identity_header": self.settings.rate_limit_identity_header,
        }


class TurnBudget:
    """Per-session turn and token budgets over the same sliding window.

    The turn budget is enforced at admission: cheap, and it stops a loop before
    it reaches the provider. The token budget is enforced post-hoc because a
    turn's cost is only known after the provider reports usage. The overshoot is
    therefore bounded by one turn's `max_tokens`, which is stated in the docs
    rather than papered over.
    """

    backend = "cache"

    def __init__(self, cache: Cache, settings: Settings, metrics: Metrics) -> None:
        self.cache = cache
        self.settings = settings
        self.metrics = metrics

    @property
    def enabled(self) -> bool:
        return self.settings.rate_limit_enabled and (
            self.settings.rate_limit_session_turns > 0
            or self.settings.rate_limit_session_tokens > 0
        )

    @property
    def window_s(self) -> int:
        return max(1, self.settings.rate_limit_window_s)

    @staticmethod
    def _turns_key(session_id: str) -> str:
        return f"cuepilot:budget:turns:{session_id}"

    @staticmethod
    def _tokens_key(session_id: str) -> str:
        return f"cuepilot:budget:tokens:{session_id}"

    async def check(self, session_id: str) -> LimitDecision:
        if not self.enabled:
            return ALLOW
        turns = self.settings.rate_limit_session_turns
        tokens = self.settings.rate_limit_session_tokens
        window_s = self.window_s
        try:
            if turns > 0:
                used = await self.cache.window_sum(self._turns_key(session_id), window_s=window_s)
                if used >= turns:
                    self.metrics.incr("turns_rejected_turn_budget")
                    return LimitDecision(
                        False,
                        code="session_turn_budget",
                        message=f"this session used {int(used)} of {turns} turns in {window_s}s",
                        # An upper bound: the oldest turn ages out somewhere in
                        # the next window, and `window_sum` does not expose when.
                        retry_after_ms=window_s * 1000,
                    )
            if tokens > 0:
                used = await self.cache.window_sum(
                    self._tokens_key(session_id), window_s=window_s
                )
                if used >= tokens:
                    self.metrics.incr("turns_rejected_token_budget")
                    return LimitDecision(
                        False,
                        code="session_token_budget",
                        message=(
                            f"this session used {int(used)} of {tokens} tokens in {window_s}s"
                        ),
                        retry_after_ms=window_s * 1000,
                    )
        except Exception as exc:  # noqa: BLE001 - fail open, but count it
            self.metrics.incr("rate_limit_cache_errors")
            logger.warning("turn budget cache failed, allowing turn: %s", exc)
        return ALLOW

    async def record_turn(self, session_id: str, tokens: int) -> None:
        """Record a completed turn. Called from the terminal path, never mid-stream."""
        if not self.enabled:
            return
        window_s = self.window_s
        try:
            if self.settings.rate_limit_session_turns > 0:
                await self.cache.window_add(
                    self._turns_key(session_id), amount=1, window_s=window_s
                )
            if tokens > 0 and self.settings.rate_limit_session_tokens > 0:
                await self.cache.window_add(
                    self._tokens_key(session_id), amount=float(tokens), window_s=window_s
                )
        except Exception as exc:  # noqa: BLE001 - budgets must not break the turn
            self.metrics.incr("rate_limit_cache_errors")
            logger.warning("turn budget record failed: %s", exc)

    def stats(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "backend": self.cache.backend,
            "window_s": self.window_s,
            "session_turns": self.settings.rate_limit_session_turns,
            "session_tokens": self.settings.rate_limit_session_tokens,
            "token_budget": "post_hoc_from_completed_usage",
            "max_overshoot_tokens": self.settings.llm_max_tokens,
        }
