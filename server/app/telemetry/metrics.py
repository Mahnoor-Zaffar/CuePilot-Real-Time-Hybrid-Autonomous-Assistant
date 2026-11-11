"""Low-cardinality runtime metrics: counters + latency histograms.

Deliberately dependency-free so the gateway reports numbers even when no
Prometheus sidecar is running. `/api/metrics` serialises this registry.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


def percentile(ordered: list[float], p: float) -> float | None:
    """Nearest-rank percentile over an already-sorted list."""
    if not ordered:
        return None
    idx = min(len(ordered) - 1, max(0, int(math.ceil(p / 100 * len(ordered))) - 1))
    return round(ordered[idx], 2)


@dataclass
class Histogram:
    """Bounded ring buffer of observations with percentile helpers."""

    max_samples: int = 4096
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=4096))

    def __post_init__(self) -> None:
        self._samples = deque(self._samples, maxlen=self.max_samples)

    def observe(self, value_ms: float) -> None:
        if value_ms >= 0 and math.isfinite(value_ms):
            self._samples.append(float(value_ms))

    def samples(self, limit: int = 512) -> list[float]:
        """Most recent observations, capped. Used to merge across replicas."""
        if limit <= 0:
            return []
        return list(self._samples)[-limit:]

    def percentile(self, p: float) -> float | None:
        return percentile(sorted(self._samples), p)

    def snapshot(
        self, *, include_samples: bool = False, sample_limit: int = 512
    ) -> dict[str, float | int | list[float] | None]:
        if not self._samples:
            snapshot: dict[str, float | int | list[float] | None] = {
                "count": 0,
                "p50": None,
                "p95": None,
                "p99": None,
                "mean": None,
                "max": None,
            }
        else:
            ordered = sorted(self._samples)
            snapshot = {
                "count": len(ordered),
                "p50": percentile(ordered, 50),
                "p95": percentile(ordered, 95),
                "p99": percentile(ordered, 99),
                "mean": round(sum(ordered) / len(ordered), 2),
                "max": round(ordered[-1], 2),
            }
        if include_samples:
            snapshot["samples"] = self.samples(sample_limit)
        return snapshot


class Metrics:
    """Single-process metric registry.

    Each replica publishes a sampled snapshot; `/api/metrics` merges the live
    snapshots (see `telemetry/aggregate.py`) instead of reporting one worker.
    """

    def __init__(self) -> None:
        self.started_at = time.time()
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, Histogram] = {
            "ttft_ms": Histogram(),
            "turn_total_ms": Histogram(),
            "retrieval_ms": Histogram(),
            "bm25_ms": Histogram(),
            "dense_ms": Histogram(),
            "fusion_ms": Histogram(),
            "tool_call_ms": Histogram(),
            "provider_attempt_ms": Histogram(),
            "ws_send_ms": Histogram(),
        }

    # counters / gauges -----------------------------------------------------
    def incr(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + amount

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value_ms: float) -> None:
        hist = self.histograms.get(name)
        if hist is not None:
            hist.observe(value_ms)

    # derived ratios -------------------------------------------------------
    @property
    def cache_hit_rate(self) -> float | None:
        hits = self.counters.get("retrieval_cache_hits", 0.0)
        misses = self.counters.get("retrieval_cache_misses", 0.0)
        total = hits + misses
        return round(hits / total, 4) if total else None

    def snapshot(self, *, include_samples: bool = False, sample_limit: int = 512) -> dict:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "counters": {k: round(v, 3) for k, v in sorted(self.counters.items())},
            "gauges": {k: round(v, 3) for k, v in sorted(self.gauges.items())},
            "histograms": {
                k: h.snapshot(include_samples=include_samples, sample_limit=sample_limit)
                for k, h in sorted(self.histograms.items())
            },
            "derived": {"retrieval_cache_hit_rate": self.cache_hit_rate},
        }
