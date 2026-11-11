"""Merge per-replica metric snapshots into one response.

The gateway is per-worker by default: counters live in the process and
`/api/metrics` used to report whichever worker answered the request. That is
fine for one process and misleading for two. Every replica now publishes a
snapshot to the shared cache; this module merges the snapshots without adding a
Prometheus dependency.

Histograms are cumulative bucket counts, so the merge is **additive**:

* every observation is represented, with no sample cap;
* a percentile is interpolated inside its bucket, so its resolution is the
  bucket layout, not the number of samples a worker happened to publish.

Counters are summed. A replica restart resets its own counters, so a merged
total can move backwards; `instances` says how many processes were visible.
Gauges are summed because every gauge published today is additive
(`ws_active`); a non-additive gauge would need an explicit rule here.
"""

from __future__ import annotations

from typing import Any

from app.telemetry.metrics import histogram_quantile

_EMPTY_HISTOGRAM: dict[str, Any] = {
    "count": 0,
    "sum": 0.0,
    "p50": None,
    "p95": None,
    "p99": None,
    "mean": None,
    "min": None,
    "max": None,
    "buckets": {},
}


def merge_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine snapshots from several workers. Pure, so it is easy to test."""
    usable = [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]

    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    names: set[str] = set()
    for snapshot in usable:
        for key, value in (snapshot.get("counters") or {}).items():
            counters[key] = counters.get(key, 0.0) + float(value)
        for key, value in (snapshot.get("gauges") or {}).items():
            gauges[key] = gauges.get(key, 0.0) + float(value)
        names.update((snapshot.get("histograms") or {}).keys())

    histograms: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        histograms[name] = _merge_histogram(
            [(snapshot.get("histograms") or {}).get(name) or {} for snapshot in usable]
        )

    hits = counters.get("retrieval_cache_hits", 0.0)
    misses = counters.get("retrieval_cache_misses", 0.0)
    total = hits + misses
    return {
        "uptime_s": round(
            max((float(snapshot.get("uptime_s") or 0.0) for snapshot in usable), default=0.0), 1
        ),
        "counters": {key: round(value, 3) for key, value in sorted(counters.items())},
        "gauges": {key: round(value, 3) for key, value in sorted(gauges.items())},
        "histograms": histograms,
        "derived": {
            "retrieval_cache_hit_rate": round(hits / total, 4) if total else None,
        },
    }


def _merge_histogram(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(item.get("count") or 0) for item in items)
    total = sum(float(item.get("sum") or 0.0) for item in items)
    maximums = [float(item["max"]) for item in items if item.get("max") is not None]

    buckets: dict[str, float] = {}
    for item in items:
        for label, value in (item.get("buckets") or {}).items():
            buckets[label] = buckets.get(label, 0.0) + float(value)

    if count <= 0:
        return {**_EMPTY_HISTOGRAM, "count": 0, "sum": 0.0, "buckets": buckets}

    maximum = max(maximums) if maximums else None
    return {
        "count": count,
        "sum": round(total, 3),
        "p50": histogram_quantile(buckets, count, 50, maximum),
        "p95": histogram_quantile(buckets, count, 95, maximum),
        "p99": histogram_quantile(buckets, count, 99, maximum),
        "mean": round(total / count, 2),
        "min": min(
            (float(item["min"]) for item in items if item.get("min") is not None),
            default=None,
        ),
        "max": maximum,
        "buckets": buckets,
    }
