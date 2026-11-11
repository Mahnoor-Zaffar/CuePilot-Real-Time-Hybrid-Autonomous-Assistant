"""Low-cardinality runtime metrics: counters, gauges and bucket histograms.

Deliberately dependency-free so the gateway reports numbers even when no
Prometheus sidecar is running. Histograms are **cumulative buckets**, not a
sample ring: buckets merge additively across replicas, so a fleet percentile is
computed from every observation instead of the last N. The trade is resolution:
a percentile is interpolated inside its bucket. `/api/metrics` returns JSON and
`/metrics` renders the Prometheus text format.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from dataclasses import dataclass, field

# Latency buckets in milliseconds. Dense where the demo lives (TTFT and turn
# latency), coarse in the tail. The last bucket is +Inf.
BUCKETS_MS: tuple[float, ...] = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    500.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
    3000.0,
    5000.0,
    7500.0,
    10000.0,
    15000.0,
    30000.0,
    60000.0,
)


def _label(bound: float) -> str:
    return str(int(bound)) if bound.is_integer() else str(bound)


def histogram_quantile(
    buckets: dict[str, float],
    count: float,
    p: float,
    max_value: float | None = None,
) -> float | None:
    """Interpolate a percentile from cumulative buckets.

    Same shape as Prometheus' `histogram_quantile`: find the bucket where the
    cumulative count crosses the rank, then interpolate between the previous
    bound and this one. `+Inf` falls back to the observed maximum.
    """
    if count <= 0:
        return None
    rank = p / 100.0 * count
    previous_bound = 0.0
    previous_cumulative = 0.0
    for label, cumulative in buckets.items():
        cumulative = float(cumulative)
        if label == "+Inf":
            if cumulative >= rank:
                return round(max_value if max_value is not None else previous_bound, 2)
            break
        bound = float(label)
        if cumulative >= rank:
            in_bucket = cumulative - previous_cumulative
            if in_bucket <= 0:
                return round(bound, 2)
            fraction = (rank - previous_cumulative) / in_bucket
            return round(previous_bound + fraction * (bound - previous_bound), 2)
        previous_bound, previous_cumulative = bound, cumulative
    return round(max_value if max_value is not None else previous_bound, 2)


@dataclass
class Histogram:
    """Cumulative bucket histogram. Every observation is counted exactly once."""

    counts: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS_MS) + 1))
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def observe(self, value_ms: float) -> None:
        if value_ms < 0 or not math.isfinite(value_ms):
            return
        value = float(value_ms)
        index = bisect_left(BUCKETS_MS, value)
        self.counts[index] += 1
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def cumulative(self) -> dict[str, float]:
        buckets: dict[str, float] = {}
        running = 0
        for bound, value in zip(BUCKETS_MS, self.counts[:-1], strict=False):
            running += value
            buckets[_label(bound)] = running
        running += self.counts[-1]
        buckets["+Inf"] = running
        return buckets

    def snapshot(self) -> dict[str, object]:
        buckets = self.cumulative()
        if not self.count:
            return {
                "count": 0,
                "sum": 0.0,
                "p50": None,
                "p95": None,
                "p99": None,
                "mean": None,
                "min": None,
                "max": None,
                "buckets": buckets,
            }
        return {
            "count": self.count,
            "sum": round(self.total, 3),
            "p50": histogram_quantile(buckets, self.count, 50, self.maximum),
            "p95": histogram_quantile(buckets, self.count, 95, self.maximum),
            "p99": histogram_quantile(buckets, self.count, 99, self.maximum),
            "mean": round(self.total / self.count, 2),
            "min": round(self.minimum, 2) if self.minimum is not None else None,
            "max": round(self.maximum, 2) if self.maximum is not None else None,
            "buckets": buckets,
        }


class Metrics:
    """Single-process metric registry.

    Each replica publishes a bucket snapshot; `/api/metrics` merges the live
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

    def snapshot(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "counters": {k: round(v, 3) for k, v in sorted(self.counters.items())},
            "gauges": {k: round(v, 3) for k, v in sorted(self.gauges.items())},
            "histograms": {k: h.snapshot() for k, h in sorted(self.histograms.items())},
            "derived": {"retrieval_cache_hit_rate": self.cache_hit_rate},
        }


def _metric_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return f"cuepilot_{safe}"


def _labels(instance_id: str, **extra: str) -> str:
    parts = [f'instance="{instance_id}"']
    parts.extend(f'{key}="{value}"' for key, value in extra.items())
    return "{" + ",".join(parts) + "}"


def render_prometheus(snapshot: dict, instance_id: str) -> str:
    """Render a merged snapshot in the Prometheus text exposition format."""
    lines: list[str] = []

    lines.append("# TYPE cuepilot_uptime_seconds gauge")
    lines.append(
        f"cuepilot_uptime_seconds{_labels(instance_id)} {snapshot.get('uptime_s', 0)}"
    )

    for name, value in (snapshot.get("counters") or {}).items():
        metric = _metric_name(name)
        lines.append(f"# TYPE {metric} counter")
        lines.append(f"{metric}{_labels(instance_id)} {value}")

    for name, value in (snapshot.get("gauges") or {}).items():
        metric = _metric_name(name)
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric}{_labels(instance_id)} {value}")

    for name, histogram in (snapshot.get("histograms") or {}).items():
        if not isinstance(histogram, dict):
            continue
        metric = _metric_name(name)
        lines.append(f"# TYPE {metric} histogram")
        for label, cumulative in (histogram.get("buckets") or {}).items():
            lines.append(
                f"{metric}_bucket{_labels(instance_id, le=label)} {cumulative}"
            )
        lines.append(f"{metric}_sum{_labels(instance_id)} {histogram.get('sum', 0)}")
        lines.append(f"{metric}_count{_labels(instance_id)} {histogram.get('count', 0)}")

    hit_rate = (snapshot.get("derived") or {}).get("retrieval_cache_hit_rate")
    if hit_rate is not None:
        lines.append("# TYPE cuepilot_retrieval_cache_hit_rate gauge")
        lines.append(
            f"cuepilot_retrieval_cache_hit_rate{_labels(instance_id)} {hit_rate}"
        )

    return "\n".join(lines) + "\n"
