"""Cross-instance event bus (Redis pub/sub) with an in-process fallback.

Used for two things that must not depend on the instance that served the
original socket:
  * `kb.invalidate` - a KB ingest on instance A must bust retrieval caches on
    instances B..N;
  * `turn.abort`    - an abort delivered to a different replica still cancels
    the in-flight generation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis

from app.config import Settings

logger = logging.getLogger(__name__)

CHANNEL = "cuepilot.events"


class EventBus:
    backend: str = "memory"

    async def publish(self, event: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        return None


class MemoryEventBus(EventBus):
    backend = "memory"

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)


class RedisEventBus(EventBus):
    backend = "redis"

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def publish(self, event: dict[str, Any]) -> None:
        await self._client.publish(CHANNEL, json.dumps(event, separators=(",", ":")))

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        pubsub = self._client.pubsub()
        await pubsub.subscribe(CHANNEL)
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    yield json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(CHANNEL)
            with contextlib.suppress(Exception):
                await pubsub.aclose()

    async def close(self) -> None:
        await self._client.aclose()


async def build_bus(settings: Settings) -> tuple[EventBus, str]:
    try:
        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=settings.infra_probe_timeout_s,
            socket_timeout=settings.infra_probe_timeout_s,
        )
        await asyncio.wait_for(client.ping(), timeout=settings.infra_probe_timeout_s)
        return RedisEventBus(client), ""
    except Exception as exc:  # noqa: BLE001
        if not settings.allow_degraded_infra:
            raise
        logger.info("event bus degraded to in-process: %s", exc)
        return MemoryEventBus(), f"redis unavailable ({type(exc).__name__})"
