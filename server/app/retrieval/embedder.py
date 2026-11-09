"""Embedding providers.

* `HttpEmbedder` - any OpenAI-compatible `/embeddings` endpoint (OpenAI itself,
  Azure, a local TEI server, ...). Used when a key is configured.
* `HashEmbedder` - deterministic signed feature hashing over word tokens,
  bigrams and character trigrams. No network, no model download; it keeps the
  dense layer, RRF and the benchmarks fully exercisable offline. Documented as
  a stand-in, never presented as a semantic model.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Protocol

import httpx
import numpy as np

from app.retrieval.tokenize import char_ngrams, tokenize


class Embedder(Protocol):
    name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


def _hash_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashEmbedder:
    name = "hash-feature-v1"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                weight = 1.0 + math.log(count)  # sublinear tf
                idx, sign = _hash_index(f"w:{token}", self.dim)
                vector[idx] += sign * weight
            for left, right in zip(tokens, tokens[1:], strict=False):
                idx, sign = _hash_index(f"b:{left}_{right}", self.dim)
                vector[idx] += sign * 0.5
        for gram in char_ngrams(text):
            idx, sign = _hash_index(f"c:{gram}", self.dim)
            vector[idx] += sign * 0.35
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Keep the event loop free-ish for large ingests.
        return await asyncio.to_thread(lambda: [self._embed_one(t) for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_one, text)


class HttpEmbedder:
    name = "http-embeddings"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        timeout_s: float = 20.0,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.dim = dim
        self._timeout = timeout_s
        self._batch_size = batch_size
        self._max_retries = max_retries

    async def _post(self, client: httpx.AsyncClient, payload: dict) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"embedding provider {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()["data"]
                return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
            except Exception as exc:  # noqa: BLE001 - retried then surfaced
                last_error = exc
                await asyncio.sleep(0.25 * (2**attempt))
        raise RuntimeError(f"embedding request failed after retries: {last_error}")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                vectors.extend(await self._post(client, {"model": self.model, "input": batch}))
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(f"embedding dim mismatch: got {len(vector)}, expected {self.dim}")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def build_embedder(
    mode: str,
    *,
    dim: int,
    api_key: str | None,
    base_url: str,
    model: str,
    timeout_s: float,
) -> Embedder:
    if mode == "openai" and api_key:
        return HttpEmbedder(
            base_url=base_url, api_key=api_key, model=model, dim=dim, timeout_s=timeout_s
        )
    return HashEmbedder(dim=dim)
