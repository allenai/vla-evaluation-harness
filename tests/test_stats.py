"""Interval and test helpers (vla_eval.results.stats) and the additive aggregate keys built on them."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from vla_eval.results import stats
from vla_eval.results.collector import EpisodeResult, ResultCollector


# ---------------------------------------------------------------------------
# Wilson / Newcombe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        (0, 10, (0.0, 0.2775328)),  # upper = z²/(n+z²)
        (10, 10, (0.7224672, 1.0)),
        (485, 500, (0.9510963, 0.9817368)),
        # docs/reproductions/lerobot.md, 500 episodes: 85.4% [82.0, 88.2], 83.2% [79.7, 86.2], 30.0% [26.1, 34.2]
        (427, 500, (0.8203507, 0.8822512)),
        (416, 500, (0.7967257, 0.8622118)),
        (150, 500, (0.2614819, 0.3415678)),
    ],
)
def test_wilson_known_values(successes: int, n: int, expected: tuple[float, float]) -> None:
    assert stats.wilson_interval(successes, n) == pytest.approx(expected, abs=1e-7)


def test_wilson_closed_form_at_zero() -> None:
    z2 = stats.Z95**2
    assert stats.wilson_interval(0, 7)[1] == pytest.approx(z2 / (7 + z2))


@pytest.mark.parametrize(("successes", "n"), [(0, 1), (3, 17), (50, 100), (499, 500)])
def test_wilson_is_symmetric_and_brackets_the_rate(successes: int, n: int) -> None:
    lo, hi = stats.wilson_interval(successes, n)
    lo2, hi2 = stats.wilson_interval(n - successes, n)
    assert lo == pytest.approx(1 - hi2) and hi == pytest.approx(1 - lo2)
    assert 0.0 <= lo <= successes / n <= hi <= 1.0


def test_wilson_empty_sample_is_uninformative() -> None:
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_newcombe_matches_the_smolvla_footnote() -> None:
    """docs/reproductions/lerobot.md (and footnote 6 of docs/reproductions/README.md): 85.8% (429/500) vs 82.7%
    (124/150) → +3.1 pp, 95% CI [−3.1, 10.5]."""
    lo, hi = stats.newcombe_difference_interval(429, 500, 124, 150)
    assert (round(lo * 100, 1), round(hi * 100, 1)) == (-3.1, 10.5)


def test_newcombe_is_antisymmetric() -> None:
    lo, hi = stats.newcombe_difference_interval(30, 40, 20, 50)
    lo2, hi2 = stats.newcombe_difference_interval(20, 50, 30, 40)
    assert (lo, hi) == pytest.approx((-hi2, -lo2))


@pytest.mark.parametrize(
    ("table", "expected", "places"),
    [
        # Newcombe (1998b) Stat Med 17:2635, Table II, method 10 (both, A-only, B-only, neither)
        ((36, 12, 2, 0), (0.0569, 0.3404), 4),
        ((20, 12, 2, 16), (0.0562, 0.3292), 4),  # phi > 0: exercises the n/2 shrink
        # Fagerland, Lydersen & Laake (2014) Stat Med 33:2850, Newcombe's method 10 on their example
        ((1, 1, 7, 12), (-0.507, -0.026), 3),
        # Tang, Li & Tang (2010) example, as reproduced by R ratesci::moverpairci (tests/testthat/test3.R)
        ((8, 3, 1, 2), (-0.1574, 0.4136), 4),
    ],
)
def test_newcombe_paired_matches_published_examples(
    table: tuple[int, int, int, int], expected: tuple[float, float], places: int
) -> None:
    lo, hi = stats.newcombe_paired_interval(*table)
    assert (round(lo, places), round(hi, places)) == expected


# R ratesci 1.1.0 moverpairci(type="wilson", corc=TRUE), as tabulated in statsmodels' test_proportion.py: positive
# and negative phi, no discordant pairs, no concordant pairs, an empty margin (phi = 0), a large table, one cell.
@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ((10, 5, 2, 20), (-0.0643926051, 0.2211160179)),
        ((3, 12, 7, 3), (-0.1336023844, 0.4874632352)),
        ((10, 0, 0, 20), (-0.0722182713, 0.0722182713)),
        ((0, 5, 3, 0), (-0.3885152108, 0.7263114284)),
        ((0, 0, 4, 6), (-0.6873262303, -0.0383858209)),
        ((5, 3, 0, 2), (-0.0529530013, 0.5665478379)),
        ((1, 1, 1, 1), (-0.4949196078, 0.4949196078)),
        ((100, 20, 30, 850), (-0.0243592378, 0.0041365528)),
        ((2, 0, 0, 0), (-0.6576197725, 0.6576197725)),
        ((0, 3, 0, 0), (0.2059232825, 1.0)),
    ],
)
def test_newcombe_paired_matches_ratesci(table: tuple[int, int, int, int], expected: tuple[float, float]) -> None:
    assert stats.newcombe_paired_interval(*table) == pytest.approx(expected, abs=1e-9)


def test_newcombe_paired_has_width_without_discordant_pairs() -> None:
    """The percentile bootstrap collapses to [0, 0] here; the score interval does not."""
    lo, hi = stats.newcombe_paired_interval(400, 0, 0, 100)
    assert lo < 0 < hi and hi - lo > 0.005
    # 4 A-only, 0 B-only: McNemar p = 0.125, and the interval keeps 0 (the bootstrap excluded it).
    lo, hi = stats.newcombe_paired_interval(400, 4, 0, 96)
    assert stats.mcnemar_exact(4, 0) == 0.125 and lo < 0 < hi


def test_newcombe_paired_is_antisymmetric() -> None:
    lo, hi = stats.newcombe_paired_interval(30, 9, 4, 7)
    lo2, hi2 = stats.newcombe_paired_interval(30, 4, 9, 7)
    assert (lo, hi) == pytest.approx((-hi2, -lo2))


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_only", "b_only", "p"),
    [
        (0, 0, 1.0),
        (0, 5, 0.0625),  # 2 · 1/32
        (5, 0, 0.0625),
        (1, 4, 0.375),  # 2 · (1 + 5)/32
        (2, 8, 0.109375),  # 2 · (1 + 10 + 45)/1024
        (3, 3, 1.0),  # capped
        (0, 1, 1.0),
    ],
)
def test_mcnemar_exact_hand_checked(a_only: int, b_only: int, p: float) -> None:
    assert stats.mcnemar_exact(a_only, b_only) == pytest.approx(p, abs=1e-15)


def test_mcnemar_matches_a_rational_binomial_tail() -> None:
    a_only, b_only = 37, 61
    n = a_only + b_only
    tail = sum(Fraction(math.comb(n, k), 2**n) for k in range(a_only + 1))
    assert stats.mcnemar_exact(a_only, b_only) == pytest.approx(float(2 * tail), rel=1e-12)


def test_mcnemar_large_counts_stay_finite() -> None:
    p = stats.mcnemar_exact(400, 600)
    assert 0.0 < p < 1e-9


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_is_seeded_and_order_invariant() -> None:
    values = [0, 1, 2, 3, 4, 5, 5, 5, 2, 1]
    first = stats.bootstrap_mean_interval(values)
    assert stats.bootstrap_mean_interval(list(reversed(values))) == first
    lo, hi = first
    assert lo < sum(values) / len(values) < hi


def test_bootstrap_constant_sample_has_zero_width() -> None:
    assert stats.bootstrap_mean_interval([3.0] * 12) == (3.0, 3.0)


def test_bootstrap_blocks_do_not_change_the_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    values = [float(i % 7) for i in range(40)]
    whole = stats.bootstrap_mean_interval(values)
    monkeypatch.setattr(stats, "_CHUNK", 40 * 3)  # 3 resamples per block
    assert stats.bootstrap_mean_interval(values) == whole


def test_bootstrap_rejects_empty() -> None:
    with pytest.raises(ValueError):
        stats.bootstrap_mean_interval([])


def test_paired_bootstrap_stratified_and_pooled() -> None:
    diffs = [1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0]
    strata = ["a", "a", "a", "a", "b", "b", "b", "b"]
    lo, hi = stats.paired_bootstrap_difference(diffs, strata)
    assert lo < sum(diffs) / len(diffs) < hi
    assert stats.paired_bootstrap_difference(diffs, strata) == (lo, hi)
    pooled = stats.paired_bootstrap_difference(diffs)
    assert pooled[0] < 0.25 < pooled[1]


def test_paired_bootstrap_rejects_singleton_strata() -> None:
    """A one-pair stratum cannot vary: compare pools such runs instead."""
    with pytest.raises(ValueError, match="two or more"):
        stats.paired_bootstrap_difference([1.0, 0.0, 0.0], ["a", "a", "b"])


def test_small_strata_are_not_too_narrow() -> None:
    """Two pairs per task, tasks alike: plain within-stratum resampling halves the variance (interval ~0.71x the
    pooled width, coverage 0.84); the sqrt(n/(n-1)) rescaling brings it back to the pooled width."""
    diffs = np.random.default_rng(0).choice([-1.0, 0.0, 1.0], size=2000, p=[0.15, 0.7, 0.15]).tolist()
    strata = [i // 2 for i in range(len(diffs))]
    lo_s, hi_s = stats.paired_bootstrap_difference(diffs, strata)
    lo_p, hi_p = stats.paired_bootstrap_difference(diffs)
    assert (hi_s - lo_s) / (hi_p - lo_p) == pytest.approx(1.0, abs=0.08)


def test_independent_bootstrap_brackets_the_difference() -> None:
    lo, hi = stats.independent_bootstrap_difference([3.0, 4.0, 5.0, 4.0], [1.0, 2.0, 2.0, 3.0])
    assert lo < 2.0 < hi


# ---------------------------------------------------------------------------
# Dispatch and verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "binary"),
    [([True, False], True), ([0, 1, 1.0], True), ([0, 2], False), ([0.5], False), ([1, 1], True)],
)
def test_is_binary(values: list[Any], binary: bool) -> None:
    assert stats.is_binary(values) is binary


def _rounded(interval: tuple[float, float]) -> list[float]:
    return [round(x, 4) for x in interval]


def test_metric_interval_dispatches_on_the_run_level_type() -> None:
    assert stats.metric_interval([True] * 485 + [False] * 15, True) == _rounded(stats.wilson_interval(485, 500))
    assert stats.metric_interval([1, 2, 3, 4], False) == _rounded(stats.bootstrap_mean_interval([1, 2, 3, 4]))
    # 0/1 values of a metric that is not binary over the run are not a rate; one value has no interval.
    assert stats.metric_interval([1, 0, 1], False) == _rounded(stats.bootstrap_mean_interval([1, 0, 1]))
    assert stats.metric_interval([1], False) is None and stats.metric_interval([2], False) is None


def test_binary_metrics_over_a_run() -> None:
    episodes = [
        {"metrics": {"success": True, "completed_subtasks": 1, "score": 1.0}},
        {"metrics": {"success": False, "completed_subtasks": 3, "score": 0}},
        {"metrics": {"success": False}, "failure_reason": "timeout"},
    ]
    assert stats.binary_metrics(episodes, ["success", "completed_subtasks", "score"]) == {"success", "score"}


def test_metric_values_skips_missing_and_non_numeric() -> None:
    episodes = [{"metrics": {"s": True}}, {"metrics": {"s": "yes"}}, {"metrics": {}}, {}, {"metrics": {"s": 0.5}}]
    assert stats.metric_values(episodes, "s") == [True, 0.5]


@pytest.mark.parametrize(
    ("mean", "reference", "verdict"),
    [
        (0.977, 0.969, "within_ci"),
        (0.919, 0.969, "within_tolerance"),  # exactly 5 pp
        (0.949, 0.995, "within_tolerance"),
        (0.900, 0.969, "outside_tolerance"),
    ],
)
def test_reproduction_verdict(mean: float, reference: float, verdict: str) -> None:
    interval = (mean - 0.015, mean + 0.012)
    assert stats.reproduction_verdict(mean, interval, reference) == verdict
    assert verdict in stats.VERDICTS


# ---------------------------------------------------------------------------
# Aggregate keys (collector; export shares _aggregate_metrics)
# ---------------------------------------------------------------------------


def _collector(metric_keys: dict[str, str], episodes: dict[str, list[dict[str, Any]]]) -> ResultCollector:
    c = ResultCollector(benchmark_name="B", metric_keys=metric_keys)
    for task, eps in episodes.items():
        for i, ep in enumerate(eps):
            c.record(task, cast(EpisodeResult, {"episode_id": i, **ep}))
    return c


def test_binary_mean_gets_wilson_interval_and_n() -> None:
    eps: list[dict[str, Any]] = [{"metrics": {"success": i < 8}} for i in range(10)]
    eps.append({"metrics": {"success": False}, "failure_reason": "timeout"})  # counted, as in the mean
    result = _collector({"success": "mean"}, {"t": eps}).get_benchmark_result()
    assert result.get("mean_success") == round(8 / 11, 4)
    assert result.get("n_success") == 11
    assert result.get("ci95_success") == [round(x, 4) for x in stats.wilson_interval(8, 11)]
    task = result["tasks"][0]
    assert task.get("ci95_success") == result.get("ci95_success") and task.get("n_success") == 11


def test_metric_type_is_decided_per_run_not_per_task() -> None:
    """CALVIN chain length: a task whose single episode completed 1 subtask is not a rate with Wilson [0.21, 1.0],
    and a task at 2 does not get a zero-width [2.0, 2.0]; neither has an interval at n = 1."""
    keys = {"completed_subtasks": "mean"}
    episodes = {"seq0": [{"metrics": {"completed_subtasks": 1}}], "seq1": [{"metrics": {"completed_subtasks": 2}}]}
    episodes["seq2"] = [{"metrics": {"completed_subtasks": c}} for c in (0, 1, 1)]
    result: dict[str, Any] = dict(_collector(keys, episodes).get_benchmark_result())
    seq0, seq1, seq2 = (dict(t) for t in result["tasks"])
    assert "ci95_completed_subtasks" not in seq0 and seq0["n_completed_subtasks"] == 1
    assert "ci95_completed_subtasks" not in seq1
    assert seq2["ci95_completed_subtasks"] == _rounded(stats.bootstrap_mean_interval([0, 1, 1]))  # not Wilson
    assert result["ci95_completed_subtasks"] == stats.metric_interval([1, 2, 0, 1, 1], False)


def test_non_binary_mean_gets_bootstrap_interval() -> None:
    chain = [5, 4, 3, 5, 2, 5, 1, 0, 4, 5]
    eps: list[dict[str, Any]] = [{"metrics": {"success": c == 5, "completed_subtasks": c}} for c in chain]
    eps.append({"metrics": {"success": False}, "failure_reason": "exception"})  # errors record no chain length
    result: dict[str, Any] = dict(
        _collector({"success": "mean", "completed_subtasks": "mean"}, {"t": eps}).get_benchmark_result()
    )
    lo, hi = stats.bootstrap_mean_interval([float(c) for c in chain])
    assert result["ci95_completed_subtasks"] == [round(lo, 4), round(hi, 4)]
    assert result["n_completed_subtasks"] == 10 and result["n_success"] == 11
    assert result["mean_completed_subtasks"] == 3.4


def test_intervals_are_additive_only() -> None:
    eps = [{"metrics": {"success": True, "reward": 2.0}}, {"metrics": {"success": False, "reward": 1.0}}]
    result: dict[str, Any] = dict(_collector({"success": "mean", "reward": "sum"}, {"t": eps}).get_benchmark_result())
    assert result["mean_success"] == 0.5 and result["sum_reward"] == 3.0
    assert "ci95_reward" not in result and "n_reward" not in result  # intervals bracket means only


def test_no_values_no_interval() -> None:
    result: dict[str, Any] = dict(_collector({"success": "mean"}, {"t": [{"metrics": {}}]}).get_benchmark_result())
    assert not {"mean_success", "ci95_success", "n_success"} & set(result)


def test_summary_prints_intervals(capsys: pytest.CaptureFixture[str]) -> None:
    eps = [{"metrics": {"success": i < 427}} for i in range(500)]
    _collector({"success": "mean"}, {"task": eps}).print_summary()
    out = capsys.readouterr().out
    assert "85.4% [82.0, 88.2] (427/500)" in out
    assert out.count("[82.0, 88.2]") == 2  # task row and overall
