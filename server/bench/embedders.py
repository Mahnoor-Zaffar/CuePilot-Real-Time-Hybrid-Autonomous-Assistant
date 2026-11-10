"""Embedders selectable from the benchmark harness.

The application has three embedding paths: `HttpEmbedder` when an
OpenAI-compatible key is configured, `LocalEmbedder` for offline semantic
embeddings, and `HashEmbedder` otherwise. The hash embedder is a lexical proxy
(word tokens, bigrams, character trigrams), so measuring "dense" retrieval with
it measures lexical robustness, not semantics.

That makes it a poor instrument for the one claim that matters here: that a
semantic layer adds recall on paraphrase queries. So the benchmark can also run
against a real embedding model.

`LocalEmbedder` uses a static sentence-embedding model (distilled token
embeddings, no transformer forward pass, no PyTorch, ~30 MB). It is a genuine
semantic encoder - "stop clients reconnecting at once" lands near "thundering
herd backoff jitter" and far from "invoice proration" - while staying cheap
enough to run the whole benchmark on a laptop. It is the same class the gateway
uses when `CUEPILOT_EMBEDDING_PROVIDER=local`, so the benchmark measures the
production code path rather than a parallel copy.

It is a stand-in for the hosted embedding path, not a replacement: a hosted
model is stronger, and `--embedder openai` uses the real `HttpEmbedder` when a
key is present. All three produce the same interface, so the retrieval code
under test is identical in every run.
"""

from __future__ import annotations

from app.retrieval.embedder import (
    DEFAULT_LOCAL_MODEL,
    Embedder,
    HashEmbedder,
    HttpEmbedder,
    LocalEmbedder,
)

__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "Model2VecEmbedder",
    "build_bench_embedder",
]

# Historical name, kept so older scripts and notes keep working.
Model2VecEmbedder = LocalEmbedder


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
        try:
            return LocalEmbedder(local_model)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
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
