from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.rrf import FusedHit


@dataclass
class RetrievalResult:
    query: str
    mode: str
    hits: list[FusedHit]
    kb_version: int
    cache_hit: bool = False
    degraded: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    candidates: dict[str, int] = field(default_factory=dict)
    embedder: str = ""

    def to_public(self, include_content: bool = True) -> dict[str, Any]:
        hits = []
        for hit in self.hits:
            payload = hit.to_public()
            if not include_content:
                payload.pop("content", None)
            hits.append(payload)
        return {
            "query": self.query,
            "mode": self.mode,
            "kb_version": self.kb_version,
            "cache_hit": self.cache_hit,
            "degraded": self.degraded,
            "embedder": self.embedder,
            "timings_ms": {k: round(v, 2) for k, v in self.timings.items()},
            "candidates": self.candidates,
            "hits": hits,
        }

    def to_context(self, max_chars: int = 6000) -> str:
        """Compact context block handed to the model."""
        lines: list[str] = []
        budget = max_chars
        for index, hit in enumerate(self.hits, start=1):
            header = f"[{index}] {hit.chunk.title} ({hit.chunk.doc_id})"
            body = hit.chunk.content.strip()
            block = f"{header}\n{body}"
            if len(block) > budget:
                block = block[: max(0, budget)]
            if not block:
                break
            lines.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        return "\n\n".join(lines)
