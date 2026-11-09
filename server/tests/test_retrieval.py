from __future__ import annotations

import pytest
from app.config import Settings
from app.infra.cache import MemoryCache
from app.infra.db import MemoryVectorStore
from app.retrieval.corpus import all_documents
from app.retrieval.embedder import HashEmbedder
from app.retrieval.orchestrator import HybridRetriever
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder
from tests.conftest import make_settings


def build(settings: Settings | None = None, *, embedder=None) -> tuple[HybridRetriever, Metrics]:
    settings = settings or make_settings()
    metrics = Metrics()
    retriever = HybridRetriever(
        settings,
        MemoryVectorStore(settings.embedding_dim),
        embedder or HashEmbedder(settings.embedding_dim),
        MemoryCache(),
        metrics,
        TraceRecorder(),
    )
    return retriever, metrics


async def seeded() -> tuple[HybridRetriever, Metrics]:
    retriever, metrics = build()
    await retriever.ingest(all_documents(include_distractors=True))
    return retriever, metrics


async def test_exact_identifier_retrieval() -> None:
    retriever, _ = await seeded()
    result = await retriever.search("CUE-E1042", top_k=5)
    assert result.hits
    assert result.hits[0].chunk.doc_id == "kb-error-codes"


async def test_hybrid_returns_fused_provenance() -> None:
    retriever, _ = await seeded()
    result = await retriever.search("how does reciprocal rank fusion work", top_k=5)
    assert result.hits
    top = result.hits[0]
    assert "bm25" in top.ranks
    assert "dense" in top.ranks
    payload = result.to_public()
    assert payload["hits"][0]["explain"]["contributions"]
    assert payload["mode"] == "hybrid"


async def test_cache_hit_then_version_bump_invalidates() -> None:
    retriever, metrics = await seeded()
    first = await retriever.search("websocket heartbeat interval", top_k=3)
    assert first.cache_hit is False
    second = await retriever.search("websocket heartbeat interval", top_k=3)
    assert second.cache_hit is True
    assert metrics.counters["retrieval_cache_hits"] == 1

    await retriever.ingest(
        [{"doc_id": "extra", "title": "Extra", "text": "A brand new document about heartbeats."}]
    )
    third = await retriever.search("websocket heartbeat interval", top_k=3)
    assert third.cache_hit is False
    assert third.kb_version > first.kb_version


async def test_single_mode_retrievers_work() -> None:
    retriever, _ = await seeded()
    for mode in ("bm25", "dense"):
        result = await retriever.search("pgvector hnsw index", mode=mode, top_k=3)
        assert result.hits, mode
        assert set(result.hits[0].ranks) == {mode}


async def test_dense_failure_degrades_to_lexical() -> None:
    class BrokenEmbedder(HashEmbedder):
        async def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("embedding provider down")

    retriever, metrics = build(embedder=BrokenEmbedder(1536))
    await retriever.ingest(all_documents(include_distractors=False))
    result = await retriever.search("CUE-E2041 rate limit", top_k=3)
    assert result.hits
    assert result.degraded == ["dense"]
    assert metrics.counters["retrieval_dense_failures"] == 1


async def test_empty_query_short_circuits() -> None:
    retriever, _ = await seeded()
    result = await retriever.search("   ")
    assert result.hits == []
    assert result.to_public()["hits"] == []


async def test_ingest_is_idempotent() -> None:
    retriever, _ = await seeded()
    before = await retriever.store.count()
    await retriever.ingest(all_documents(include_distractors=True))
    after = await retriever.store.count()
    assert before == after


@pytest.mark.parametrize("top_k", [1, 8, 20])
async def test_top_k_is_respected(top_k: int) -> None:
    retriever, _ = await seeded()
    result = await retriever.search("retrieval", top_k=top_k)
    assert len(result.hits) <= top_k
