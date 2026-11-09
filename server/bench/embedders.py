"""Embedders selectable from the benchmark harness.

The application has two embedding paths: `HttpEmbedder` when an OpenAI-compatible
key is configured, and `HashEmbedder` otherwise. The hash embedder is a lexical
proxy (word tokens, bigrams, character trigrams), so measuring "dense" retrieval
with it measures lexical robustness, not semantics.

That makes it a poor instrument for the one claim that matters here: that a
semantic layer adds recall on paraphrase queries. So the benchmark can also run
against a real embedding model.

`Model2VecEmbedder` uses a static sentence-embedding model (distilled token
embeddings, no transformer forward pass, no PyTorch, ~30 MB). It is a genuine
semantic encoder - "stop clients reconnecting at once" lands near "thundering
herd backoff jitter" and far from "invoice proration" - while staying cheap enough
to run the whole benchmark on a laptop.

It is a stand-in for the hosted embedding path, not a replacement: a hosted model
is stronger, and `--embedder openai` uses the real `HttpEmbedder` when a key is
present. All three produce the same interface, so the retrieval code under test
is identical in every run.
"""

from __future__ import annotations

import asyncio

import numpy as np
from app.retrieval.embedder import Embedder, HashEmbedder, HttpEmbedder


class Model2VecEmbedder:
    """Static sentence embeddings via model2vec. Optional dependency."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise SystemExit(
                "model2vec is not installed. Install the benchmark extra:\n"
                "  cd server && uv pip install -e '.[bench]'"
            ) from exc
        self._model = StaticModel.from_pretrained(model_name)
        self.name = f"model2vec:{model_name.rsplit('/', 1)[-1]}"
        self.dim = int(self._model.dim)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32).tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


DEFAULT_LOCAL_MODEL = "minishlab/potion-base-8M"


def build_bench_embedder(
    choice: str,
    *,
    dim: int,
    api_key: str | None,
    base_url: str,
    model: str,
    timeout_s: float,
    local_model: str = DEFAULT_LOCAL_MODEL,
) -> Embedder:
    if choice == "hash":
        return HashEmbedder(dim=dim)
    if choice == "local":
        return Model2VecEmbedder(local_model)
    if choice == "openai":
        if not api_key:
            raise SystemExit("--embedder openai requires CUEPILOT_OPENAI_API_KEY")
        return HttpEmbedder(
            base_url=base_url,
            api_key=api_key,
            model=model,
            dim=dim,
            timeout_s=timeout_s,
        )
    raise SystemExit(f"unknown embedder {choice!r} (choose hash, local or openai)")
