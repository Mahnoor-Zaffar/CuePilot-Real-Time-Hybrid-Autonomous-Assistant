"""Process-wide dependency graph, built once in the FastAPI lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.infra.bus import EventBus, build_bus
from app.infra.cache import Cache, build_cache
from app.infra.db import PgVectorStore, VectorStore, build_store
from app.llm.agent import Agent
from app.llm.base import ChatProvider
from app.llm.registry import build_provider
from app.llm.tools import ToolRegistry
from app.retrieval.corpus import all_documents
from app.retrieval.embedder import build_embedder
from app.retrieval.orchestrator import HybridRetriever
from app.telemetry.aggregate import merge_snapshots
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder, configure_tracing
from app.ws.journal import TurnJournal
from app.ws.limits import ClientLimiter
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
    limiter: ClientLimiter
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    degraded: list[str] = field(default_factory=list)
    seed_info: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    _bus_task: asyncio.Task | None = None
    _metrics_task: asyncio.Task | None = None

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

        # The journal shares the vector store's pool. With no Postgres there is
        # nothing durable to write to, so the manager stays mirror-only rather
        # than pretending otherwise.
        journal: TurnJournal | None = None
        if settings.session_journal_enabled and isinstance(store, PgVectorStore):
            journal = TurnJournal(store.pool, settings, metrics)
        sessions = SessionManager(cache, settings, journal)
        limiter = ClientLimiter(cache, settings, metrics)

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
            limiter=limiter,
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
        await self._publish_metrics()
        self._metrics_task = asyncio.create_task(self._metrics_loop(), name="metrics-publisher")
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
        if self._metrics_task:
            self._metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._metrics_task
            self._metrics_task = None
        await self.sessions.stop()
        # Remove this replica's published snapshot so a graceful restart does
        # not look like a second instance for the rest of the TTL.
        with contextlib.suppress(Exception):
            await self.cache.delete(self.metrics_key)
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

    # ── metrics across replicas ────────────────────────────────────────────
    @property
    def metrics_key(self) -> str:
        return f"cuepilot:metrics:{self.instance_id}"

    async def _metrics_loop(self) -> None:
        interval = max(1.0, self.settings.metrics_publish_interval_s)
        while True:
            await asyncio.sleep(interval)
            await self._publish_metrics()

    async def _publish_metrics(self) -> None:
        try:
            snapshot = self.metrics.snapshot(
                include_samples=True, sample_limit=self.settings.metrics_sample_limit
            )
            snapshot["instance_id"] = self.instance_id
            snapshot["published_at"] = time.time()
            await self.cache.set(
                self.metrics_key,
                json.dumps(snapshot, separators=(",", ":")),
                self.settings.metrics_publish_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never break the app
            self.metrics.incr("metrics_publish_failures")
            logger.debug("metrics publish failed: %s", exc)

    async def aggregated_metrics(self) -> dict[str, Any]:
        """Publish this replica, then merge every live snapshot in the cache."""
        local = self.metrics.snapshot(
            include_samples=True, sample_limit=self.settings.metrics_sample_limit
        )
        local["instance_id"] = self.instance_id
        local["published_at"] = time.time()
        snapshots: dict[str, dict[str, Any]] = {self.instance_id: local}
        error: str | None = None
        try:
            await self.cache.set(
                self.metrics_key,
                json.dumps(local, separators=(",", ":")),
                self.settings.metrics_publish_ttl_s,
            )
            for key in await self.cache.scan("cuepilot:metrics:"):
                if key == self.metrics_key:
                    continue
                raw = await self.cache.get(key)
                if not raw:
                    continue
                try:
                    snapshot = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(snapshot, dict):
                    continue
                instance_id = str(snapshot.get("instance_id") or key.rsplit(":", 1)[-1])
                snapshots[instance_id] = snapshot
        except Exception as exc:  # noqa: BLE001 - local metrics are still useful
            error = f"{type(exc).__name__}: {exc}"
            self.metrics.incr("metrics_aggregation_failures")

        merged = merge_snapshots(list(snapshots.values()))
        approximate = merged.pop("approximate_histograms", [])
        merged["aggregation"] = {
            "backend": self.cache.backend,
            "instances": len(snapshots),
            "instance_ids": sorted(snapshots),
            "merged": len(snapshots) > 1,
            "approximate_histograms": approximate,
            "sample_limit": self.settings.metrics_sample_limit,
            "note": (
                "histogram percentiles merge the most recent samples per instance; "
                "counters are summed and reset when a replica restarts"
            ),
        }
        if error is not None:
            merged["aggregation"]["error"] = error
        return merged

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
            "journal": (
                self.sessions.journal.stats()
                if self.sessions.journal is not None
                else {"backend": "none"}
            ),
            "limits": self.limiter.stats(),
            "transcription_enabled": bool(self.settings.openai_api_key)
            and self.settings.audio_enabled,
            "sessions_active": self.sessions.active_sessions,
        }
