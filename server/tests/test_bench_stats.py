from __future__ import annotations

import numpy as np
import pytest
from bench.stats import Interval, bootstrap_ci, describe_effect, paired_bootstrap_delta

HITS = [1.0] * 87 + [0.0] * 13


def test_interval_brackets_the_point_estimate() -> None:
    interval = bootstrap_ci(HITS, resamples=500, seed=1)

    assert interval.point == pytest.approx(0.87)
    assert interval.low <= interval.point <= interval.high
    assert interval.n == 100
    assert interval.resamples == 500
    assert not interval.paired


def test_a_seed_makes_the_interval_reproducible() -> None:
    first = bootstrap_ci(HITS, resamples=500, seed=3)
    second = bootstrap_ci(HITS, resamples=500, seed=3)
    other = bootstrap_ci(HITS, resamples=500, seed=4)

    assert (first.low, first.high) == (second.low, second.high)
    assert (first.low, first.high) != (other.low, other.high)


def test_degenerate_samples_have_zero_width() -> None:
    all_hits = bootstrap_ci([1.0] * 20, resamples=200)
    single = bootstrap_ci([0.5], resamples=200)

    assert (all_hits.low, all_hits.high) == (1.0, 1.0)
    assert (single.low, single.high) == (0.5, 0.5)


def test_pairing_is_tighter_than_resampling_each_arm_independently() -> None:
    # Two retrievers that agree on 99 of 100 queries. The unpaired interval has
    # to absorb the variance of both arms; the paired one only sees the flips.
    base = [1.0 if index % 5 else 0.0 for index in range(100)]
    candidate = list(base)
    candidate[0] = 1.0 - candidate[0]

    paired = paired_bootstrap_delta(base, candidate, resamples=2000, seed=2)

    rng = np.random.default_rng(2)
    b = np.asarray(base)
    c = np.asarray(candidate)
    bi = rng.integers(0, b.size, size=(2000, b.size))
    ci = rng.integers(0, c.size, size=(2000, c.size))
    unpaired = c[ci].mean(axis=1) - b[bi].mean(axis=1)
    unpaired_width = float(np.quantile(unpaired, 0.975) - np.quantile(unpaired, 0.025))

    assert paired.point == pytest.approx(float(np.mean(c - b)))
    assert paired.paired
    assert paired.width < unpaired_width


def test_paired_delta_rejects_mismatched_samples() -> None:
    with pytest.raises(ValueError, match="must match"):
        paired_bootstrap_delta([1.0, 0.0], [1.0])


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, float("nan")])
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ci([1.0], resamples=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci([1.0], alpha=1.0)


def test_excludes_zero_and_effect_description() -> None:
    real = Interval(point=0.07, low=0.01, high=0.13, n=100, resamples=10, paired=True)
    noise = Interval(point=0.01, low=-0.04, high=0.06, n=100, resamples=10, paired=True)

    assert real.excludes_zero
    assert not noise.excludes_zero
    assert "excludes zero" in describe_effect(real)
    assert "includes zero" in describe_effect(noise)
    assert describe_effect(noise).startswith("+1.0 pp")


def test_effect_verdict_matches_the_printed_bounds() -> None:
    # A lower bound of +0.0001 prints as +0.0, so calling it "excludes zero"
    # would contradict the sentence a reader actually sees.
    borderline = Interval(point=0.04, low=0.0001, high=0.09, n=100, resamples=10)
    text = describe_effect(borderline)

    assert "+0.0 to +9.0" in text
    assert "includes zero" in text


def test_interval_serialises_to_plain_json_types() -> None:
    payload = bootstrap_ci(HITS, resamples=100, seed=1).to_dict()
    assert set(payload) == {"point", "low", "high", "n", "resamples", "paired"}
    assert all(isinstance(payload[key], (int, float, bool)) for key in payload)
