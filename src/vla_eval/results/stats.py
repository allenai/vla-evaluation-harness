"""Uncertainty for episode metrics: 95% intervals, paired tests, and the reproduction verdict.

One module so aggregates (:func:`metric_interval`), ``vla-eval compare`` and the docs agree on the method:

- **Binary** metrics (every value in the run a bool or 0/1, see :func:`binary_metrics`): Wilson score interval.
  Differences use Newcombe's score intervals: method 10 of Newcombe (1998b) for paired runs (with the exact McNemar
  test), method 10 of Newcombe (1998a) for independent runs.
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
    """Numeric values of ``metrics[key]``, skipping episodes that lack it.  The population every aggregate uses.
    A harness error records only ``success=False``: it counts as a failure in ``success`` and is absent from every
    other metric (``num_errors`` reports it)."""
    return [
        m[key]
        for e in episodes
        if isinstance((m := e.get("metrics", {})), dict) and key in m and isinstance(m[key], (int, float, bool))
    ]


def is_binary(values: Iterable[Any]) -> bool:
    """True when every value is a bool or numerically 0/1 (so the mean is a proportion).  Decide over a whole run
    (:func:`binary_metrics`), never per task: a count metric can be all 0/1 on one task."""
    return all(isinstance(v, bool) or v in (0, 1) for v in values)


def binary_metrics(episodes: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> frozenset[str]:
    """The ``keys`` whose values over ``episodes`` (a whole run, or both compared runs) are all bools or 0/1."""
    episodes = list(episodes)
    return frozenset(k for k in keys if is_binary(metric_values(episodes, k)))


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


def metric_interval(values: Sequence[Any], binary: bool) -> list[float] | None:
    """An aggregate's ``ci95_<key>``, rounded like the ``mean_<key>`` it brackets: Wilson when the metric is binary
    over the run, else a seeded percentile bootstrap; ``None`` for a non-binary sample of fewer than two values."""
    if binary:
        lo, hi = wilson_interval(sum(bool(v) for v in values), len(values))
    elif len(values) < 2:
        return None
    else:
        lo, hi = bootstrap_mean_interval([float(v) for v in values])
    return [round(lo, 4), round(hi, 4)]


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
    """95% interval for ``x1/n1 - x2/n2`` from independent samples (Newcombe 1998a, method 10: hybrid score
    from the two Wilson intervals)."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_interval(x1, n1)
    l2, u2 = wilson_interval(x2, n2)
    d = p1 - p2
    return d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def newcombe_paired_interval(both: int, a_only: int, b_only: int, neither: int) -> tuple[float, float]:
    """95% interval for ``p_A - p_B`` from paired binary outcomes (Newcombe 1998b, "Improved confidence intervals
    for the difference between binomial proportions based on paired data", Stat Med 17:2635, method 10): the
    Wilson limits of both marginal rates, combined with the table's phi coefficient.  phi's numerator
    ``both·neither − a_only·b_only`` is shrunk toward 0 by ``n/2`` when positive, and phi is 0 when a margin is
    empty.  Unlike a bootstrap it has positive width with no discordant pairs."""
    n = both + a_only + b_only + neither
    if n <= 0:
        raise ValueError("paired interval of an empty sample")
    p1, p2 = (both + a_only) / n, (both + b_only) / n
    l1, u1 = wilson_interval(both + a_only, n)
    l2, u2 = wilson_interval(both + b_only, n)
    margins = (both + a_only) * (b_only + neither) * (both + b_only) * (a_only + neither)
    phi = 0.0
    if margins:
        cov = both * neither - a_only * b_only
        phi = (max(cov - n / 2, 0.0) if cov > 0 else cov) / math.sqrt(margins)

    def half(x: float, y: float) -> float:
        return math.sqrt(max(0.0, x * x - 2 * phi * x * y + y * y))

    d = p1 - p2
    return max(-1.0, d - half(p1 - l1, u2 - p2)), min(1.0, d + half(u1 - p1, p2 - l2))


def paired_bootstrap_difference(
    diffs: Sequence[float],
    strata: Sequence[Any] | None = None,
    *,
    resamples: int = COMPARE_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean paired difference.  With ``strata`` (the task of each pair)
    resampling stays within each stratum, so every resample keeps the run's task mix; every stratum needs two or
    more pairs.  Order-invariant like :func:`bootstrap_mean_interval`."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap of an empty sample")
    rng = np.random.default_rng(seed)
    if strata is None:
        return _percentiles(_bootstrap_sums(np.sort(arr), resamples, rng) / arr.size)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(str(s), []).append(i)
    if min(len(g) for g in groups.values()) < 2:
        raise ValueError("stratified bootstrap needs two or more pairs per stratum")
    totals = np.zeros(resamples, dtype=np.float64)
    for key in sorted(groups):
        x = np.sort(arr[groups[key]])
        # Resampling n_h values underestimates the stratum mean's variance by (n_h - 1)/n_h, which compounds
        # across many small strata.  Inflating deviations from the stratum mean by sqrt(n_h / (n_h - 1)) restores
        # it.  Simulated (400 pairs, 1000 runs, ternary and task-heterogeneous chain-length diffs), coverage at
        # 2/3/5/10/50 pairs per task: 0.84-0.86/0.90-0.91/0.92-0.93/0.94-0.96/0.95 without, 0.95-0.97 with.
        m = x.mean()
        totals += _bootstrap_sums(m + (x - m) * math.sqrt(x.size / (x.size - 1)), resamples, rng)
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
