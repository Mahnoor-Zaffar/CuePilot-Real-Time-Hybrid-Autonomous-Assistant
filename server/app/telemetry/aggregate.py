"""Merge per-replica metric snapshots into one response.

The gateway is per-worker by default: counters live in the process and
`/api/metrics` used to report whichever worker answered the request. That is
fine for one process and misleading for two. Every replica now publishes a
snapshot to the shared cache; this module merges the snapshots without adding a
Prometheus dependency.

The merge is deliberately honest about what it cannot know:

* **Counters** are summed. A replica restart resets its own counters, so the
  merged total can move backwards; `instances` says how many processes were
  visible when the snapshot was taken.
* **Gauges** are summed. Every gauge we publish today is additive (`ws_active`);
  a non-additive gauge would need an explicit rule here.
* **Histograms** merge the most recent `sample_limit` raw observations per
  worker, then recompute percentiles. `count` is the true sum, but percentiles
  are approximate when a worker observed more values than it published. The
  approximation is named in `approximate_histograms`, never hidden.
"""

from __future__ import annotations

from typing import Any

from app.telemetry.metrics import percentile

_EMPTY_HISTOGRAM: dict[str, Any] = {
    "count": 0,
    "p50": None,
    "p95": None,
    "p99": None,
    "mean": None,
    "max": None,
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
    approximate: list[str] = []
    for name in sorted(names):
        merged, is_approximate = _merge_histogram(
            [(snapshot.get("histograms") or {}).get(name) or {} for snapshot in usable]
        )
        histograms[name] = merged
        if is_approximate:
            approximate.append(name)

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
        "approximate_histograms": approximate,
    }


def _merge_histogram(items: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    count = sum(int(item.get("count") or 0) for item in items)
    samples: list[float] = []
    approximate = False
    for item in items:
        raw = item.get("samples")
        if isinstance(raw, list):
            samples.extend(float(value) for value in raw if isinstance(value, (int, float)))
        if int(item.get("count") or 0) > len(raw or []):
            approximate = True

    if not samples:
        return {**_EMPTY_HISTOGRAM, "count": count}, count > 0

    ordered = sorted(samples)
    return {
        "count": count,
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "mean": round(sum(ordered) / len(ordered), 2),
        "max": round(ordered[-1], 2),
    }, approximate
