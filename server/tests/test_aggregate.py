from __future__ import annotations

from app.telemetry.aggregate import merge_snapshots


def snapshot(
    *,
    uptime: float = 10.0,
    turns: int = 1,
    active: int = 1,
    ttft: list[float] | None = None,
    ttft_count: int | None = None,
) -> dict:
    samples = ttft or []
    return {
        "uptime_s": uptime,
        "counters": {"turns_total": turns, "retrieval_cache_hits": 1, "retrieval_cache_misses": 1},
        "gauges": {"ws_active": active},
        "histograms": {
            "ttft_ms": {
                "count": ttft_count if ttft_count is not None else len(samples),
                "samples": samples,
            }
        },
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


def test_histogram_percentiles_are_recomputed_from_merged_samples() -> None:
    merged = merge_snapshots(
        [
            snapshot(ttft=[10.0, 20.0]),
            snapshot(ttft=[30.0, 40.0]),
        ]
    )

    ttft = merged["histograms"]["ttft_ms"]
    assert ttft["count"] == 4
    assert ttft["p50"] == 20.0
    assert ttft["p95"] == 40.0
    assert merged["approximate_histograms"] == []


def test_a_worker_that_published_fewer_samples_is_flagged_approximate() -> None:
    merged = merge_snapshots(
        [
            snapshot(ttft=[10.0, 20.0], ttft_count=2),
            snapshot(ttft=[30.0], ttft_count=500),
        ]
    )

    ttft = merged["histograms"]["ttft_ms"]
    assert ttft["count"] == 502
    assert merged["approximate_histograms"] == ["ttft_ms"]


def test_cache_hit_rate_is_recomputed_from_summed_counters() -> None:
    merged = merge_snapshots([snapshot(), snapshot()])

    assert merged["derived"]["retrieval_cache_hit_rate"] == 0.5


def test_an_empty_fleet_is_an_empty_snapshot() -> None:
    merged = merge_snapshots([])

    assert merged["counters"] == {}
    assert merged["histograms"] == {}
    assert merged["derived"]["retrieval_cache_hit_rate"] is None
