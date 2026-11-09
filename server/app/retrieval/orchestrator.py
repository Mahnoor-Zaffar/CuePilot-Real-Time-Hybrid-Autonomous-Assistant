"""Dual-retrieval orchestrator.

Parallel BM25 + dense candidate generation, RRF fusion, versioned caching and
graceful single-retriever degradation. This is the only module that knows how the
lexical and semantic layers are combined.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from app.config import Settings
from app.infra.cache import Cache, cache_get_json, cache_set_json
from app.infra.db import Chunk, VectorStore
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import chunk_document
from app.retrieval.embedder import Embedder
from app.retrieval.rrf import FusedHit, reciprocal_rank_fusion
from app.retrieval.types import RetrievalResult
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder

VALID_MODES = ("hybrid", "bm25", "dense")


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        cache: Cache,
        metrics: Metrics,
        recorder: TraceRecorder,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.cache = cache
        self.metrics = metrics
        self.recorder = recorder
        self._bm25: BM25Index | None = None
        self._kb_version: int = 1
        self._refresh_lock = asyncio.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────────
    @property
    def kb_version(self) -> int:
        return self._kb_version

    async def refresh(self) -> None:
        """Rebuild the lexical index from the store (source of truth)."""
        async with self._refresh_lock:
            chunks = await self.store.all_chunks()
            self._bm25 = await asyncio.to_thread(
                BM25Index, chunks, self.settings.bm25_k1, self.settings.bm25_b
            )
            self._kb_version = await self.store.version()

    async def warm(self) -> None:
        await self.refresh()

    async def stats(self) -> dict[str, Any]:
        return {
            "chunks": await self.store.count(),
            "kb_version": self._kb_version,
            "bm25": self._bm25.stats() if self._bm25 else None,
            "store_backend": self.store.backend,
            "embedder": self.embedder.name,
        }

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        return await self.store.get_chunk(chunk_id)

    # ── ingest ─────────────────────────────────────────────────────────────
    async def ingest(
        self,
        documents: list[dict[str, Any]],
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with self.recorder.span("kb.ingest", documents=len(documents), replace=replace) as span:
            if replace:
                await self.store.clear()
            chunks: list[Chunk] = []
            for doc in documents:
                chunks.extend(
                    chunk_document(
                        doc_id=str(doc["doc_id"]),
                        title=str(doc.get("title", "")),
                        text=str(doc["text"]),
                        max_chars=self.settings.chunk_max_chars,
                        overlap=self.settings.chunk_overlap_chars,
                        meta={"tags": doc.get("tags", []), "source": doc.get("source", "seed")},
                    )
                )
            if chunks:
                with self.recorder.span("kb.embed", chunks=len(chunks)) as embed_span:
                    vectors = await self.embedder.embed_documents([c.content for c in chunks])
                    embed_span.set(embedder=self.embedder.name)
                for chunk, vector in zip(chunks, vectors, strict=True):
                    chunk.embedding = vector
                with self.recorder.span("kb.upsert", chunks=len(chunks)) as upsert_span:
                    written = await self.store.upsert(chunks)
                    upsert_span.set(written=written)
            version = await self.store.bump_version()
            await self.refresh()
            self._kb_version = version
            span.set(kb_version=version, chunks=len(chunks))
        result = {
            "documents": len(documents),
            "chunks": len(chunks),
            "kb_version": version,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "store_backend": self.store.backend,
            "embedder": self.embedder.name,
        }
        return result

    # ── search ─────────────────────────────────────────────────────────────
    def _cache_key(self, query: str, mode: str, top_k: int, candidate_k: int) -> str:
        digest = hashlib.sha1(f"{query}|{mode}|{top_k}|{candidate_k}".encode()).hexdigest()
        return f"cuepilot:ret:{self._kb_version}:{digest}"

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "hybrid",
        candidate_k: int | None = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        query = (query or "").strip()
        top_k = top_k or self.settings.retrieval_top_k
        candidate_k = candidate_k or self.settings.retrieval_candidate_k
        mode = mode if mode in VALID_MODES else "hybrid"
        if not query:
            return RetrievalResult(query="", mode=mode, hits=[], kb_version=self._kb_version)

        cache_key = self._cache_key(query, mode, top_k, candidate_k)
        if use_cache:
            cached = await cache_get_json(self.cache, cache_key)
            if cached is not None:
                self.metrics.incr("retrieval_cache_hits")
                return self._from_cache(cached)
            self.metrics.incr("retrieval_cache_misses")

        timings: dict[str, float] = {}
        candidates: dict[str, int] = {}
        degraded: list[str] = []
        ranked: dict[str, list[tuple[Chunk, float]]] = {}

        with self.recorder.span("retrieval", query=query[:200], mode=mode) as root:
            tasks: dict[str, asyncio.Task] = {}
            if mode in ("hybrid", "bm25"):
                tasks["bm25"] = asyncio.create_task(self._lexical(query, candidate_k))
            if mode in ("hybrid", "dense"):
                tasks["dense"] = asyncio.create_task(self._semantic(query, candidate_k))

            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, outcome in zip(tasks, gathered, strict=True):
                if isinstance(outcome, BaseException):
                    degraded.append(name)
                    root.set(**{f"{name}_error": type(outcome).__name__})
                    self.metrics.incr(f"retrieval_{name}_failures")
                    continue
                hits, elapsed = outcome
                ranked[name] = hits
                timings[name] = elapsed
                candidates[name] = len(hits)
                self.metrics.observe(f"{name}_ms", elapsed)

            if not ranked:
                root.set(status="degraded")
                return RetrievalResult(
                    query=query,
                    mode=mode,
                    hits=[],
                    kb_version=self._kb_version,
                    degraded=degraded,
                    timings=timings,
                    embedder=self.embedder.name,
                )

            if len(ranked) == 1:
                # single retriever: still run through RRF so ordering/provenance match
                fused_started = time.perf_counter()
                fused = reciprocal_rank_fusion(ranked, k=self.settings.rrf_k, top_k=top_k)
                timings["fusion"] = (time.perf_counter() - fused_started) * 1000
            else:
                fused_started = time.perf_counter()
                fused = reciprocal_rank_fusion(ranked, k=self.settings.rrf_k, top_k=top_k)
                timings["fusion"] = (time.perf_counter() - fused_started) * 1000

            total_ms = (time.perf_counter() - root.started_at) * 1000
            root.set(hits=len(fused), degraded=",".join(degraded) or None)

        self.metrics.observe("retrieval_ms", total_ms)
        self.metrics.observe("fusion_ms", timings.get("fusion", 0.0))

        result = RetrievalResult(
            query=query,
            mode=mode,
            hits=fused,
            kb_version=self._kb_version,
            cache_hit=False,
            degraded=degraded,
            timings=timings,
            candidates=candidates,
            embedder=self.embedder.name,
        )
        if use_cache and fused:
            await cache_set_json(
                self.cache, cache_key, self._to_cache(result), self.settings.retrieval_cache_ttl_s
            )
        return result

    async def _lexical(self, query: str, k: int) -> tuple[list[tuple[Chunk, float]], float]:
        started = time.perf_counter()
        with self.recorder.span("retrieval.bm25", candidates=k) as span:
            if self._bm25 is None:
                await self.refresh()
            assert self._bm25 is not None
            hits = await asyncio.wait_for(
                asyncio.to_thread(self._bm25.search, query, k),
                timeout=self.settings.retrieval_timeout_s,
            )
            span.set(results=len(hits))
        return [(h.chunk, h.score) for h in hits], (time.perf_counter() - started) * 1000

    async def _semantic(self, query: str, k: int) -> tuple[list[tuple[Chunk, float]], float]:
        started = time.perf_counter()
        with self.recorder.span("retrieval.dense", candidates=k) as span:
            vector = await asyncio.wait_for(
                self.embedder.embed_query(query), timeout=self.settings.retrieval_timeout_s
            )
            hits = await asyncio.wait_for(
                self.store.search(vector, k), timeout=self.settings.retrieval_timeout_s
            )
            span.set(results=len(hits), dim=len(vector))
        return hits, (time.perf_counter() - started) * 1000

    # ── cache (de)serialisation ────────────────────────────────────────────
    @staticmethod
    def _to_cache(result: RetrievalResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "mode": result.mode,
            "kb_version": result.kb_version,
            "degraded": result.degraded,
            "timings": result.timings,
            "candidates": result.candidates,
            "embedder": result.embedder,
            "hits": [
                {
                    "chunk": {
                        "id": h.chunk.id,
                        "doc_id": h.chunk.doc_id,
                        "title": h.chunk.title,
                        "content": h.chunk.content,
                        "meta": h.chunk.meta,
                    },
                    "score": h.score,
                    "ranks": h.ranks,
                    "contributions": h.contributions,
                    "raw_scores": h.raw_scores,
                }
                for h in result.hits
            ],
        }

    def _from_cache(self, payload: dict[str, Any]) -> RetrievalResult:
        hits = [
            FusedHit(
                chunk=Chunk(embedding=[], **item["chunk"]),
                score=item["score"],
                ranks=item.get("ranks", {}),
                contributions=item.get("contributions", {}),
                raw_scores=item.get("raw_scores", {}),
            )
            for item in payload.get("hits", [])
        ]
        return RetrievalResult(
            query=payload.get("query", ""),
            mode=payload.get("mode", "hybrid"),
            hits=hits,
            kb_version=payload.get("kb_version", self._kb_version),
            cache_hit=True,
            degraded=payload.get("degraded", []),
            timings=payload.get("timings", {}),
            candidates=payload.get("candidates", {}),
            embedder=payload.get("embedder", self.embedder.name),
        )
