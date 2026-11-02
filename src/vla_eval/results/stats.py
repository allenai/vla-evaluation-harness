"""Uncertainty for episode metrics: 95% intervals, paired tests, and the reproduction verdict.

One module so aggregates (:func:`metric_interval`), ``vla-eval compare`` and the docs agree on the method:

- **Binary** metrics (every value a bool or 0/1): Wilson score interval; paired runs use the exact McNemar test;
  independent runs use Newcombe's hybrid score interval (method 10) for the difference.
- **Non-binary** metrics (CALVIN chain length, scores): percentile bootstrap of the mean, and of the paired mean
  difference stratified by task.

Bootstraps use a fixed seed, so the same episodes always give the same interval. numpy only; no scipy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Z95 = 1.959963984540054  # two-sided 95% standard-normal quantile
AGGREGATE_RESAMPLES = 2000  # per aggregate interval (every task + overall, at every export)
COMPARE_RESAMPLES = 10_000  # one-off comparisons can afford more
BOOTSTRAP_SEED = 0
TOLERANCE = 0.05  # reproduction verdict: 5 pp for rates (docs/reproductions/README.md)

_CHUNK = 1 << 22  # bootstrap index elements per block (~32 MB of int64)


def metric_values(episodes: Iterable[Mapping[str, Any]], key: str) -> list[int | float | bool]:
    """Numeric values of ``metrics[key]``, skipping episodes that lack it.  The population every aggregate uses:
    harness errors record ``success=False`` and so count as failures (``num_errors`` reports them)."""
    return [
        m[key]
        for e in episodes
        if isinstance((m := e.get("metrics", {})), dict) and key in m and isinstance(m[key], (int, float, bool))
    ]


def is_binary(values: Iterable[Any]) -> bool:
    """True when every value is a bool or numerically 0/1 (so the mean is a proportion)."""
    return all(isinstance(v, bool) or v in (0, 1) for v in values)


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``successes / n``; ``(0, 1)`` when ``n == 0``.  Unlike the normal (Wald)
    interval it stays inside [0, 1] and is not degenerate at 0/n or n/n."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # Exact at the boundaries; avoids 1e-17 drift past 0/1.
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == n else min(1.0, center + half)
    return lo, hi


def _bootstrap_sums(values: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Sums of ``resamples`` with-replacement resamples of ``values`` (same size), in bounded-memory blocks."""
    n = len(values)
    out = np.empty(resamples, dtype=np.float64)
    block = max(1, _CHUNK // max(n, 1))
    for start in range(0, resamples, block):
        stop = min(resamples, start + block)
        out[start:stop] = values[rng.integers(0, n, size=(stop - start, n))].sum(axis=1)
    return out


def _percentiles(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_mean_interval(
    values: Sequence[float], *, resamples: int = AGGREGATE_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of the mean.  Seeded, and the sample is sorted first, so the interval
    depends only on the multiset of values (not on episode order)."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        raise ValueError("bootstrap of an empty sample")
    return _percentiles(_bootstrap_sums(arr, resamples, np.random.default_rng(seed)) / arr.size)


def mean_interval(values: Sequence[Any], *, resamples: int = AGGREGATE_RESAMPLES) -> tuple[float, float]:
    """95% interval of the mean: Wilson for binary values, seeded percentile bootstrap otherwise."""
    if is_binary(values):
        return wilson_interval(sum(bool(v) for v in values), len(values))
    return bootstrap_mean_interval([float(v) for v in values], resamples=resamples)


def metric_interval(values: Sequence[Any]) -> tuple[list[float], int]:
    """``([lo, hi], n)`` for an aggregate's ``ci95_<key>`` / ``n_<key>``, rounded like the ``mean_<key>`` it
    brackets."""
    lo, hi = mean_interval(values)
    return [round(lo, 4), round(hi, 4)], len(values)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs: ``a_only`` pairs where only A succeeded,
    ``b_only`` where only B did.  Under H0 each discordant pair is a fair coin, so this is the two-sided binomial
    test of ``min(a_only, b_only)`` in ``a_only + b_only`` at p=0.5 (1.0 with no discordant pairs)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2**n)  # exact int arithmetic, one correctly rounded division


def newcombe_difference_interval(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """95% interval for ``x1/n1 - x2/n2`` from independent samples (Newcombe 1998, method 10: hybrid score
    from the two Wilson intervals)."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_interval(x1, n1)
    l2, u2 = wilson_interval(x2, n2)
    d = p1 - p2
    return d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def paired_bootstrap_difference(
    diffs: Sequence[float],
    strata: Sequence[Any] | None = None,
    *,
    resamples: int = COMPARE_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean paired difference.  With ``strata`` (the task of each pair)
    resampling stays within each stratum, so every resample keeps the run's task mix.  Order-invariant like
    :func:`bootstrap_mean_interval`."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap of an empty sample")
    rng = np.random.default_rng(seed)
    if strata is None:
        return _percentiles(_bootstrap_sums(np.sort(arr), resamples, rng) / arr.size)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(str(s), []).append(i)
    totals = np.zeros(resamples, dtype=np.float64)
    for key in sorted(groups):
        totals += _bootstrap_sums(np.sort(arr[groups[key]]), resamples, rng)
    return _percentiles(totals / arr.size)


def independent_bootstrap_difference(
    a: Sequence[float], b: Sequence[float], *, resamples: int = COMPARE_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """95% percentile bootstrap interval of ``mean(a) - mean(b)`` from independent samples."""
    xa, xb = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    if xa.size == 0 or xb.size == 0:
        raise ValueError("bootstrap of an empty sample")
    rng = np.random.default_rng(seed)
    return _percentiles(_bootstrap_sums(xa, resamples, rng) / xa.size - _bootstrap_sums(xb, resamples, rng) / xb.size)


VERDICTS = {
    "within_ci": ("✅", "within 95% CI"),
    "within_tolerance": ("🟡", "outside CI but within tolerance"),
    "outside_tolerance": ("🔧", "outside tolerance"),
}


def reproduction_verdict(
    mean: float, interval: Sequence[float], reference: float, tolerance: float = TOLERANCE
) -> str:
    """The reproduction table's verdict (docs/reproductions/README.md): ``within_ci`` when the reference lies in
    our run's 95% interval, else ``within_tolerance`` when ``|mean - reference| <= tolerance`` (5 pp for rates),
    else ``outside_tolerance``.  Keys of :data:`VERDICTS`."""
    lo, hi = interval
    if lo <= reference <= hi:
        return "within_ci"
    # Absorb float noise so a 5.0 pp gap counts as within 5 pp.
    if abs(mean - reference) <= tolerance + 1e-9:
        return "within_tolerance"
    return "outside_tolerance"
