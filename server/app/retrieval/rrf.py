"""Reciprocal Rank Fusion.

    score(d) = Σ_l weight_l / (k + rank_l(d)),  k = 60 (Cormack et al. 2009)

Why RRF instead of a cross-encoder: it needs no training, no second forward
pass, and is robust to score-scale mismatch between BM25 and cosine. The cost is
one O(n) pass over the candidate pool, i.e. microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.infra.db import Chunk


@dataclass(slots=True)
class FusedHit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_public(self) -> dict:
        payload = self.chunk.to_public(score=self.score)
        payload["explain"] = {
            "ranks": self.ranks,
            "contributions": {k: round(v, 6) for k, v in self.contributions.items()},
            "raw_scores": {k: round(v, 6) for k, v in self.raw_scores.items()},
        }
        return payload


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[Chunk, float]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Merge ranked lists into one, preserving per-retriever provenance."""
    weights = weights or {}
    fused: dict[str, FusedHit] = {}
    for retriever, results in ranked_lists.items():
        weight = weights.get(retriever, 1.0)
        for rank, (chunk, raw_score) in enumerate(results, start=1):
            entry = fused.get(chunk.id)
            if entry is None:
                entry = FusedHit(chunk=chunk, score=0.0)
                fused[chunk.id] = entry
            contribution = weight / (k + rank)
            entry.score += contribution
            entry.ranks[retriever] = rank
            entry.contributions[retriever] = contribution
            entry.raw_scores[retriever] = raw_score
    ordered = sorted(fused.values(), key=lambda h: (-h.score, h.chunk.id))
    if top_k is not None:
        ordered = ordered[:top_k]
    for hit in ordered:
        hit.score = round(hit.score, 8)
    return ordered
