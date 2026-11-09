"""Okapi BM25 over the chunk corpus.

Exact-match layer of the hybrid stack: identifiers, error codes and entity names
are scored literally, with no embedding lossy-compression in the way.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from app.infra.db import Chunk
from app.retrieval.tokenize import tokenize


@dataclass(slots=True)
class BM25Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """Inverted index with per-chunk length normalisation (k1, b)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = chunks
        self._term_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._postings: dict[str, list[int]] = {}
        self._build()

    def _build(self) -> None:
        for idx, chunk in enumerate(self.chunks):
            tokens = tokenize(f"{chunk.title}\n{chunk.content}")
            freqs = Counter(tokens)
            self._term_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            for term in freqs:
                self._doc_freq[term] += 1
                self._postings.setdefault(term, []).append(idx)
        self._avgdl = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0
        n = len(self.chunks)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self._doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 50) -> list[BM25Hit]:
        if not self.chunks:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = {}
        avgdl = self._avgdl or 1.0
        for term in set(terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx in self._postings.get(term, ()):
                tf = self._term_freqs[idx][term]
                dl = self._doc_lens[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [BM25Hit(chunk=self.chunks[idx], score=round(score, 6)) for idx, score in ranked]

    def stats(self) -> dict[str, float | int]:
        return {
            "documents": len(self.chunks),
            "vocabulary": len(self._doc_freq),
            "avg_doc_tokens": round(self._avgdl, 2),
        }
