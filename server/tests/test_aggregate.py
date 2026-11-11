from __future__ import annotations

from app.telemetry.aggregate import merge_snapshots
from app.telemetry.metrics import Histogram


def histogram(*samples: float) -> dict:
    hist = Histogram()
    for sample in samples:
        hist.observe(sample)
    return hist.snapshot()


def snapshot(
    *,
    uptime: float = 10.0,
    turns: int = 1,
    active: int = 1,
    ttft: dict | None = None,
) -> dict:
    return {
        "uptime_s": uptime,
        "counters": {"turns_total": turns, "retrieval_cache_hits": 1, "retrieval_cache_misses": 1},
        "gauges": {"ws_active": active},
        "histograms": {"ttft_ms": ttft if ttft is not None else histogram()},
        "derived": {"retrieval_cache_hit_rate": 0.5},
    }


def test_counters_and_gauges_are_summed_across_instances() -> None:
    merged = merge_snapshots(
        [
            snapshot(turns=2, active=3),
            snapshot(turns=5, active=4, uptime=90.0),
        ]
    )

    assert merged["counters"]["turns_total"] == 7
    assert merged["gauges"]["ws_active"] == 7
    # The oldest instance is the useful uptime for a fleet.
    assert merged["uptime_s"] == 90.0


def test_histogram_buckets_merge_additively_with_no_sample_cap() -> None:
    merged = merge_snapshots(
        [
            snapshot(ttft=histogram(10.0, 20.0)),
            snapshot(ttft=histogram(30.0, 40.0)),
        ]
    )

    ttft = merged["histograms"]["ttft_ms"]
    assert ttft["count"] == 4
    assert ttft["sum"] == 100.0
    # Every observation is counted; the percentile is interpolated inside a bucket.
    assert ttft["buckets"]["+Inf"] == 4
    assert 10.0 <= ttft["p50"] <= 25.0
    assert 25.0 <= ttft["p95"] <= 50.0
    assert "approximate_histograms" not in merged
    assert "sample_limit" not in merged


def test_a_worker_with_many_more_observations_is_not_flagged() -> None:
    small = histogram(10.0, 20.0)
    large = histogram(*([500.0] * 500))
    merged = merge_snapshots([snapshot(ttft=small), snapshot(ttft=large)])

    ttft = merged["histograms"]["ttft_ms"]
    # The merge is additive: no worker's observations are truncated to a sample cap.
    assert ttft["count"] == 502
    assert ttft["max"] == 500.0


def test_merged_min_and_max_come_from_the_workers() -> None:
    merged = merge_snapshots(
        [
            snapshot(ttft=histogram(5.0, 200.0)),
            snapshot(ttft=histogram(75.0)),
        ]
    )

    ttft = merged["histograms"]["ttft_ms"]
    assert ttft["min"] == 5.0
    assert ttft["max"] == 200.0


def test_cache_hit_rate_is_recomputed_from_summed_counters() -> None:
    merged = merge_snapshots([snapshot(), snapshot()])

    assert merged["derived"]["retrieval_cache_hit_rate"] == 0.5


def test_an_empty_fleet_is_an_empty_snapshot() -> None:
    merged = merge_snapshots([])

    assert merged["counters"] == {}
    assert merged["histograms"] == {}
    assert merged["derived"]["retrieval_cache_hit_rate"] is None
