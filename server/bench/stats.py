"""Uncertainty for retrieval metrics.

With 100 labelled queries, one percentage point is one query. A bare point
estimate cannot separate a real effect from label noise, and comparing two
overlapping intervals by eye is not a test. So every metric carries a
percentile bootstrap interval, and every comparison is a *paired* bootstrap over
the same queries: pairing removes query-difficulty variance, which is the
dominant term here, and gives a much tighter interval on the difference than
two independent intervals would.

What this does not do: it quantifies sampling noise in this label set. It says
nothing about whether the corpus resembles a real domain. Those are different
questions and both need answering.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float
    n: int
    resamples: int
    paired: bool = False

    @property
    def excludes_zero(self) -> bool:
        """True when the interval is strictly positive or strictly negative."""
        return self.low > 0 or self.high < 0

    @property
    def width(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "point": round(self.point, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n": self.n,
            "resamples": self.resamples,
            "paired": self.paired,
        }


def _validate(resamples: int, alpha: float) -> None:
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _as_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("expected a non-empty 1-D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> Interval:
    """Percentile bootstrap interval for a statistic of one sample."""
    _validate(resamples, alpha)
    array = _as_array(values)
    point = float(statistic(array))
    n = int(array.size)
    if n == 1:
        return Interval(point, point, point, n, resamples)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    samples = np.array([statistic(array[row]) for row in idx], dtype=np.float64)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(low), float(high), n, resamples)


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for mean(candidate - baseline) on matched samples.

    Both sequences must be per-query, in the same query order. Resampling the
    per-query differences keeps the pairing, so a query that is hard for every
    retriever contributes the same to both sides.
    """
    _validate(resamples, alpha)
    base = _as_array(baseline)
    cand = _as_array(candidate)
    if base.shape != cand.shape:
        raise ValueError(f"paired samples must match: {base.size} != {cand.size}")

    diff = cand - base
    n = int(diff.size)
    point = float(diff.mean())
    if n == 1:
        return Interval(point, point, point, n, resamples, paired=True)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    samples = diff[idx].mean(axis=1)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(low), float(high), n, resamples, paired=True)


def describe_effect(interval: Interval, *, scale: float = 100.0, unit: str = "pp") -> str:
    """One sentence a reader can act on, instead of two numbers to eyeball.

    The verdict is taken from the *rounded* bounds so it always agrees with the
    numbers printed next to it: an interval that displays `+0.0 to +9.0` is
    reported as including zero, even if the unrounded lower bound was barely
    positive.
    """
    point = round(interval.point * scale, 1)
    low = round(interval.low * scale, 1)
    high = round(interval.high * scale, 1)
    excludes_zero = low > 0 or high < 0
    verdict = (
        "excludes zero"
        if excludes_zero
        else "includes zero, so this sample cannot distinguish it from noise"
    )
    return (
        f"{point:+.1f} {unit} (95% CI {low:+.1f} to {high:+.1f}; "
        f"n={interval.n}, {verdict})"
    )
