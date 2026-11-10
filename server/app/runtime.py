"""Process-wide dependency graph, built once in the FastAPI lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.infra.bus import EventBus, build_bus
from app.infra.cache import Cache, build_cache
from app.infra.db import VectorStore, build_store
from app.llm.agent import Agent
from app.llm.base import ChatProvider
from app.llm.registry import build_provider
from app.llm.tools import ToolRegistry
from app.retrieval.corpus import all_documents
from app.retrieval.embedder import build_embedder
from app.retrieval.orchestrator import HybridRetriever
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder, configure_tracing
from app.ws.session import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    metrics: Metrics
    recorder: TraceRecorder
    cache: Cache
    bus: EventBus
    store: VectorStore
    retriever: HybridRetriever
    provider: ChatProvider
    tools: ToolRegistry
    agent: Agent
    sessions: SessionManager
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    degraded: list[str] = field(default_factory=list)
    seed_info: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    _bus_task: asyncio.Task | None = None

    # ── construction ───────────────────────────────────────────────────────
    @classmethod
    async def create(cls, settings: Settings) -> Runtime:
        metrics = Metrics()
        recorder = configure_tracing(settings)
        cache, cache_reason = await build_cache(settings)
        bus, bus_reason = await build_bus(settings)

        # The embedder is built first: a local model declares its own width, and
        # the store's schema is migrated to that width before the first write.
        embedder = build_embedder(
            settings.embedding_mode,
            dim=settings.embedding_dim,
            api_key=settings.openai_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            timeout_s=settings.embedding_timeout_s,
            local_model=settings.local_embedding_model,
        )
        store, store_reason = await build_store(settings, dim=embedder.dim)
        degraded = [reason for reason in (cache_reason, bus_reason, store_reason) if reason]

        retriever = HybridRetriever(settings, store, embedder, cache, metrics, recorder)
        provider = build_provider(settings, metrics)
        tools = ToolRegistry(retriever, settings, metrics, recorder)
        agent = Agent(provider, retriever, tools, settings, metrics, recorder)
        sessions = SessionManager(cache, settings)

        runtime = cls(
            settings=settings,
            metrics=metrics,
            recorder=recorder,
            cache=cache,
            bus=bus,
            store=store,
            retriever=retriever,
            provider=provider,
            tools=tools,
            agent=agent,
            sessions=sessions,
            degraded=degraded,
        )
        metrics.gauge("ws_active", 0)
        return runtime

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        await self.retriever.warm()
        if self.settings.seed_on_start:
            await self._seed_if_empty()
        await self.sessions.start()
        self._bus_task = asyncio.create_task(self._bus_loop(), name="event-bus")
        logger.info(
            "runtime ready instance=%s provider=%s model=%s store=%s cache=%s chunks=%s "
            "embedder=%s dim=%s degraded=%s",
            self.instance_id,
            self.provider.name,
            self.provider.model,
            self.store.backend,
            self.cache.backend,
            await self.store.count(),
            self.retriever.embedder.name,
            self.retriever.embedder.dim,
            self.degraded or "none",
        )

    async def stop(self) -> None:
        if self._bus_task:
            self._bus_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._bus_task
        await self.sessions.stop()
        with contextlib.suppress(Exception):
            await self.provider.close()
        with contextlib.suppress(Exception):
            await self.store.close()
        with contextlib.suppress(Exception):
            await self.bus.close()
        with contextlib.suppress(Exception):
            await self.cache.close()

    async def _seed_if_empty(self) -> None:
        count = await self.store.count()
        if count > 0:
            self.seed_info = {"seeded": False, "chunks": count}
            return
        try:
            self.seed_info = {"seeded": True, **(await self.retriever.ingest(all_documents()))}
        except Exception as exc:  # noqa: BLE001 - seeding must not block startup
            logger.warning("seed failed: %s", exc)
            self.seed_info = {"seeded": False, "error": str(exc)}

    # ── cross-instance events ──────────────────────────────────────────────
    async def _bus_loop(self) -> None:
        try:
            async for event in self.bus.subscribe():
                kind = event.get("kind")
                if kind == "kb.invalidate":
                    if event.get("origin") != self.instance_id:
                        logger.info("kb invalidate from %s", event.get("origin"))
                        await self.retriever.refresh()
                elif kind == "turn.abort":
                    if event.get("origin") == self.instance_id:
                        continue
                    session = await self.sessions.get(event.get("session_id", ""))
                    if session is None or session.active_turn is None:
                        continue
                    handle = session.active_turn
                    if event.get("turn_id") and event["turn_id"] != handle.turn_id:
                        continue
                    if handle.task is not None:
                        handle.task.cancel()
                        self.metrics.incr("turns_aborted_cross_instance")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("event bus loop failed")

    # ── health ─────────────────────────────────────────────────────────────
    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if not self.degraded else "degraded",
            "instance_id": self.instance_id,
            "uptime_s": round(time.time() - self.started_at, 1),
            "degraded": self.degraded,
            "backends": {
                "cache": self.cache.backend,
                "bus": self.bus.backend,
                "vector_store": self.store.backend,
                "embedder": self.retriever.embedder.name,
                "embedding_dim": self.retriever.embedder.dim,
            },
            "provider": {"mode": self.provider.name, "model": self.provider.model},
            "kb": {
                "chunks": await self.store.count(),
                "kb_version": self.retriever.kb_version,
                "seed": self.seed_info,
            },
            "transcription_enabled": bool(self.settings.openai_api_key)
            and self.settings.audio_enabled,
            "sessions_active": self.sessions.active_sessions,
        }
