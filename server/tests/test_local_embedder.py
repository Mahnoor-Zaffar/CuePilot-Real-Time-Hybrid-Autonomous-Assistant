from __future__ import annotations

import numpy as np
import pytest
from app.retrieval.embedder import LocalEmbedder, build_embedder
from tests.conftest import make_settings

# The model is an optional dependency and a ~30 MB download, so the default
# suite skips this file. The `local-embedder` CI job installs the extra and
# runs it for real: that job is what keeps this path from being a test double.
pytest.importorskip("model2vec", reason="install the local extra: uv sync --extra local")

pytestmark = pytest.mark.local


def cosine(a: list[float], b: list[float]) -> float:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


async def test_local_embedder_is_actually_semantic() -> None:
    embedder = LocalEmbedder()
    documents = await embedder.embed_documents(
        [
            "clients all reconnect at once after an outage and hammer the gateway",
            "invoice proration rules for mid-cycle upgrades",
        ]
    )
    query = await embedder.embed_query("thundering herd backoff jitter")

    assert embedder.name.startswith("model2vec:")
    assert embedder.dim == len(query) == len(documents[0])
    # A lexical hash proxy cannot do this: the query shares no words with the
    # on-topic document, so only a semantic encoder ranks it above the off-topic one.
    assert cosine(query, documents[0]) > cosine(query, documents[1])


async def test_local_embedder_handles_empty_batches() -> None:
    embedder = LocalEmbedder()
    assert await embedder.embed_documents([]) == []


async def test_build_embedder_selects_the_local_model() -> None:
    settings = make_settings(embedding_provider="local")
    assert settings.embedding_mode == "local"

    embedder = build_embedder(
        settings.embedding_mode,
        dim=settings.embedding_dim,
        api_key=None,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        timeout_s=settings.embedding_timeout_s,
        local_model=settings.local_embedding_model,
    )
    assert isinstance(embedder, LocalEmbedder)
    vector = await embedder.embed_query("how does rrf fuse bm25 and dense results")
    assert len(vector) == embedder.dim
