"""``vla-eval compare``: task matching, pairing, intervals, warnings, the reproduction verdict, and export round trip."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pytest

from vla_eval.cli import main as cli
from vla_eval.recording import RecordingStore
from vla_eval.results import stats
from vla_eval.results.compare import compare_runs, parse_reference


def _aggregate(
    tasks: dict[str, list[Any]], *, task_idx: bool = True, metric: str = "success", benchmark: str = "B"
) -> dict[str, Any]:
    """An aggregate whose task ``i`` has one episode per value, ``episode_id`` = position."""
    return {
        "benchmark": benchmark,
        "tasks": [
            {
                "task": name,
                "episodes": [
                    {"episode_id": e, "metrics": {metric: v}, **({"task_idx": i} if task_idx else {})}
                    for e, v in enumerate(values)
                ],
            }
            for i, (name, values) in enumerate(tasks.items())
        ],
    }


def _write(path: Path, body: dict[str, Any]) -> str:
    path.write_text(json.dumps(body))
    return str(path)


def _ns(a: str, b: str | None = None, **kw: Any) -> argparse.Namespace:
    return argparse.Namespace(
        a=a,
        b=b,
        metric=kw.get("metric", "success"),
        reference=kw.get("reference"),
        tolerance=kw.get("tolerance"),
        allow_benchmark_mismatch=kw.get("allow_benchmark_mismatch", False),
        json=kw.get("json", True),
    )


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def _text(tmp_path: Path, capsys: pytest.CaptureFixture[str], a: Any, b: Any = None, **kw: Any) -> str:
    """Normal (non --json) output of ``compare a [b]``."""
    pa = _write(tmp_path / "a.json", a)
    pb = _write(tmp_path / "b.json", b) if b is not None else None
    cli.cmd_compare(_ns(pa, pb, json=False, **kw))
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def test_paired_binary_uses_mcnemar_and_newcombe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # t1: A wins 4 discordant pairs, B wins 1; t2: identical.
    a = _aggregate({"t1": [1, 1, 1, 1, 1, 0, 0, 1], "t2": [1, 0, 1, 0]})
    b = _aggregate({"t1": [0, 0, 0, 0, 1, 0, 1, 1], "t2": [1, 0, 1, 0]})
    cli.cmd_compare(_ns(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b)))
    report = _json(capsys)
    cmp = report["comparison"]
    assert report["binary"] is True and report["warnings"] == [] and cmp["warnings"] == []
    assert cmp["method"] == "paired" and cmp["paired"] == 12 and cmp["tasks_compared"] == 2
    assert cmp["unmatched_a"] == cmp["unmatched_b"] == 0
    assert cmp["pair_key"] == ["task", "episode_id"]
    assert cmp["mcnemar"] == {"a_only": 4, "b_only": 1, "p_value": 0.375}
    assert cmp["difference"] == round(3 / 12, 4)
    # both 4, A-only 4, B-only 1, neither 3
    assert cmp["ci95"] == [round(x, 4) for x in stats.newcombe_paired_interval(4, 4, 1, 3)]
    assert cmp["interval"] == "Newcombe paired score"
    t1, t2 = cmp["tasks"]
    assert (t1["task"], t1["difference"], t1["p_value"]) == ("t1", 0.375, 0.375)
    assert t1["ci95"] == [round(x, 4) for x in stats.newcombe_paired_interval(2, 4, 1, 1)]  # same method per task
    assert (t2["difference"], t2["p_value"]) == (0.0, 1.0)
    assert t2["ci95"][0] < 0 < t2["ci95"][1]  # identical outcomes still get a real interval


def test_task_filtered_rerun_pairs_on_task_identity() -> None:
    """A ran t0, t1, t2; B reran only t2 (``tasks: [t2]``), so B's t2 has task_idx 0.  Same task, same episodes:
    they pair, and A's other tasks are left out instead of being compared against B's t2."""
    a = _aggregate({"t0": [1] * 10, "t1": [1] * 10, "t2": [0] * 5 + [1] * 5})
    b = _aggregate({"t2": [0] * 5 + [1] * 5})
    cmp = compare_runs(a, b)
    assert cmp["method"] == "paired" and cmp["paired"] == 10 and cmp["tasks_compared"] == 1
    assert (cmp["tasks_only_a"], cmp["tasks_only_b"]) == (2, 0)
    assert cmp["difference"] == 0.0 and cmp["mean_a"] == cmp["mean_b"] == 0.5
    assert cmp["warnings"] == ["tasks run by only one side are left out: A 2 (20 episodes), B 0 (0 episodes)"]


def test_same_named_tasks_are_told_apart_by_task_idx_rank() -> None:
    def run(outcomes: dict[int, list[int]]) -> dict[str, Any]:
        eps = [
            {"episode_id": e, "task_idx": i, "metrics": {"success": v}}
            for i, vs in outcomes.items()
            for e, v in enumerate(vs)
        ]
        return {"benchmark": "B", "tasks": [{"task": "same", "episodes": eps}]}

    cmp = compare_runs(run({0: [1, 1], 1: [0, 0]}), run({3: [1, 1], 7: [0, 0]}))  # other positions, same order
    assert cmp["paired"] == 4 and cmp["difference"] == 0.0 and "duplicate_keys" not in cmp
    assert [t["task"] for t in cmp["tasks"]] == ["same #1", "same #2"]


def _copies_run(outcomes: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    eps = [{"episode_id": e, "task_idx": i, "metrics": m} for i, ms in outcomes.items() for e, m in enumerate(ms)]
    return {"benchmark": "B", "tasks": [{"task": "soup", "episodes": eps}]}


def test_copy_rank_counts_episodes_without_the_metric() -> None:
    """A's first copy only errored (no chain_len): its second copy must still pair with B's second."""
    err = {"success": False}
    a = _copies_run({10: [err, err], 40: [{"chain_len": 2}, {"chain_len": 2}]})
    b = _copies_run({10: [{"chain_len": 5}, {"chain_len": 5}], 40: [{"chain_len": 2}, {"chain_len": 2}]})
    cmp = compare_runs(a, b, "chain_len")
    assert cmp["paired"] == 2 and cmp["difference"] == 0.0
    assert [t["task"] for t in cmp["tasks"]] == ["soup #2"]


def test_copy_rank_needs_task_idx_in_both_runs_and_equal_copies() -> None:
    ok, bad = {"success": True}, {"success": False}
    a = _copies_run({0: [ok, ok], 1: [bad, bad]})
    b = _copies_run({0: [ok, ok], 1: [bad, bad]})
    for ep in b["tasks"][0]["episodes"]:
        del ep["task_idx"]
    cmp = compare_runs(a, b)  # without B's task_idx neither run's copies are ranked: ambiguous keys, never paired
    assert cmp["paired"] == 0 and cmp["duplicate_keys"] == [4, 4]
    with pytest.raises(ValueError, match="share no task"):  # A ran 'soup' twice, B once: no correspondence
        compare_runs(a, _copies_run({5: [ok, ok]}))


def test_shared_prefix_title_is_escaped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    names = ["put [/] the long object on the left shelf", "put [/] the long object on the right shelf"]
    a = {
        "benchmark": "B",
        "tasks": [{"task": n, "episodes": [{"episode_id": 0, "metrics": {"success": 1}}]} for n in names],
    }
    out = _text(tmp_path, capsys, a, a)
    assert "put [/] the long object" in out


def test_malformed_aggregate_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = _write(tmp_path / "bad.json", {"tasks": [1, {"task": "t", "episodes": [{"episode_id": 0}]}]})
    with pytest.raises(SystemExit) as exc:
        cli.cmd_compare(_ns(bad, reference="50%"))
    assert exc.value.code == 1 and "malformed task entry" in capsys.readouterr().err


def test_suite_is_part_of_task_identity_when_both_runs_record_it() -> None:
    a = _aggregate({"t": [1, 0]})
    b = _aggregate({"t": [1, 0]})
    for ep in a["tasks"][0]["episodes"]:
        ep["suite"] = "libero_goal"
    for ep in b["tasks"][0]["episodes"]:
        ep["suite"] = "libero_10"
    with pytest.raises(ValueError, match="share no task"):
        compare_runs(a, b)
    del b["tasks"][0]["episodes"][0]["suite"]  # one run without suites: match on the name only
    assert compare_runs(a, b)["paired"] == 2


def test_unmatched_episodes_are_counted_and_warned(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({"t": [1, 0] * 5})
    b = _aggregate({"t": [1, 0] * 5 + [1]})  # 10 of 11 pair: >= 90%
    cmp = compare_runs(a, b)
    assert cmp["method"] == "paired" and (cmp["paired"], cmp["unmatched_a"], cmp["unmatched_b"]) == (10, 0, 1)
    out = _text(tmp_path, capsys, a, b)
    assert "⚠ unpaired episodes are left out: A 0, B 1" in out


def test_partial_overlap_is_compared_independently(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """250 vs 250 episodes sharing one episode id: one pair is not a sample of the runs (was +100 pp, CI [100, 100])."""
    a = _aggregate({"t": [1] * 250})
    b = _aggregate({"t": [0] * 250})
    for ep in b["tasks"][0]["episodes"][1:]:
        ep["episode_id"] += 1000
    cmp = compare_runs(a, b)
    assert cmp["method"] == "independent" and cmp["paired"] == 1 and "mcnemar" not in cmp
    assert (cmp["n_a"], cmp["n_b"]) == (250, 250)
    assert cmp["ci95"] == [round(x, 4) for x in stats.newcombe_difference_interval(250, 250, 0, 250)]
    out = _text(tmp_path, capsys, a, b)
    assert "⚠ only 1 pairs for 250 A / 250 B episodes on the shared tasks (< 90%)" in out
    assert "Independent samples over 1 shared tasks" in out


def test_pairing_threshold_is_ninety_percent() -> None:
    a = _aggregate({"t": [1, 0] * 5})
    b = _aggregate({"t": [1, 0] * 5})
    for ep in b["tasks"][0]["episodes"][8:]:  # 8 of 10 pair
        ep["episode_id"] += 100
    assert compare_runs(a, b)["method"] == "independent"
    b["tasks"][0]["episodes"][8]["episode_id"] = 8  # 9 of 10
    assert compare_runs(a, b)["method"] == "paired"


def test_duplicate_keys_are_not_paired_and_are_shown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({"t": [1, 0]}, task_idx=False)
    a["tasks"][0]["episodes"][1]["episode_id"] = 0
    b = _aggregate({"t": [1, 0]}, task_idx=False)
    cmp = compare_runs(a, b)
    assert cmp["duplicate_keys"] == [2, 0] and cmp["method"] == "independent"
    out = _text(tmp_path, capsys, a, b)
    assert "⚠ episodes sharing a (task, episode_id) key are not paired: A 2, B 0" in out


def test_nothing_pairs_falls_back_to_newcombe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({"t": [1] * 429 + [0] * 71}, task_idx=False)
    b = _aggregate({"t": [1] * 124 + [0] * 26}, task_idx=False)
    for ep in b["tasks"][0]["episodes"]:
        ep["episode_id"] += 1000
    cli.cmd_compare(_ns(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b)))
    cmp = _json(capsys)["comparison"]
    assert cmp["method"] == "independent" and cmp["paired"] == 0
    assert "mcnemar" not in cmp
    assert [round(x * 100, 1) for x in cmp["ci95"]] == [-3.1, 10.5]  # the SmolVLA footnote
    assert cmp["tasks"][0]["n_a"] == 500 and cmp["tasks"][0]["n_b"] == 150


def test_non_binary_metric_uses_paired_bootstrap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({f"seq{i}": [c] for i, c in enumerate([5, 4, 3, 5, 2])}, metric="chain")
    b = _aggregate({f"seq{i}": [c] for i, c in enumerate([4, 4, 1, 5, 2])}, metric="chain")
    cmp = compare_runs(a, b, "chain")
    assert cmp["difference"] == 0.6 and "mcnemar" not in cmp
    # One episode per task: pooled, not stratified (singleton strata never vary).
    assert cmp["interval"] == "paired bootstrap"
    assert cmp["ci95"][0] < 0.6 < cmp["ci95"][1]
    out = _text(tmp_path, capsys, a, b, metric="chain")
    assert "+0.600" in out and not re.search(r"\d\.\d%|\bpp\b", out)


def test_count_metric_with_0_1_tasks_is_not_a_rate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Binary-ness is decided over both runs: B's 2s make chain length a count everywhere."""
    a = _aggregate({"s0": [1], "s1": [0]}, metric="chain")
    b = _aggregate({"s0": [2], "s1": [1]}, metric="chain")
    out = _text(tmp_path, capsys, a, b, metric="chain", reference="3", tolerance=1.0)
    assert "0.500" in out and not re.search(r"\d\.\d%", out)
    assert "Reference 3.000" in out


def test_identical_non_binary_runs_print_no_zero_width_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _aggregate({"t": [3, 4, 5, 4]}, metric="chain")
    cmp = compare_runs(a, a, "chain")
    assert cmp["ci95"] == [0.0, 0.0]
    out = _text(tmp_path, capsys, a, a, metric="chain")
    assert "95% CI n/a (no variation)" in out and "[0.000, 0.000]" not in out


def test_stratified_bootstrap_when_every_task_has_two_pairs() -> None:
    a = _aggregate({"t1": [3, 4, 5], "t2": [1, 2]}, metric="chain")
    b = _aggregate({"t1": [2, 4, 4], "t2": [1, 1]}, metric="chain")
    assert compare_runs(a, b, "chain")["interval"] == "paired bootstrap, stratified by task"


# ---------------------------------------------------------------------------
# Refusals and warnings
# ---------------------------------------------------------------------------


def test_different_benchmarks_are_refused_unless_allowed(tmp_path: Path) -> None:
    a = _aggregate({"s": [1, 1, 0]}, benchmark="libero_spatial")
    b = _aggregate({"s": [0, 0, 0]}, benchmark="libero_goal")
    with pytest.raises(ValueError, match="--allow-benchmark-mismatch"):
        compare_runs(a, b)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_compare(_ns(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b)))
    assert exc.value.code == 1
    assert compare_runs(a, b, allow_benchmark_mismatch=True)["paired"] == 3


def test_condition_mismatches_are_warned(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({"t": [1, 0]})
    b = _aggregate({"t": [1, 0]})
    a.update(seed=7, mode="sync", config={"max_steps": 220, "params": {"seed": 7, "num_steps_wait": 10}})
    b.update(seed=0, mode="live", config={"max_steps": 280, "params": {"seed": 0, "num_steps_wait": 10}}, partial=True)
    out = _text(tmp_path, capsys, a, b)
    assert "⚠ seeds differ (A 7, B 0)" in out
    assert "⚠ max_steps differs (A 220, B 280)" in out
    assert "⚠ modes differ (A sync, B live)" in out
    assert "⚠ B is partial" in out
    a["config"]["params"]["num_steps_wait"] = 5
    assert "benchmark params differ: num_steps_wait (A 5, B 10)" in compare_runs(a, b)["warnings"]
    a["config"]["params"]["cameras"] = ["agentview", "wrist"]  # brackets are text, not Rich markup
    assert "cameras (A ['agentview', 'wrist'], B None)" in _text(tmp_path, capsys, a, b)
    assert compare_runs(a, b, allow_benchmark_mismatch=True)["warnings"][0].startswith("seeds differ")
    b["benchmark"] = "other"
    assert compare_runs(a, b, allow_benchmark_mismatch=True)["warnings"][0] == "benchmarks differ (A B, B other)"


def test_errors_are_counted_for_non_success_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Error episodes record only success=False: they drop out of other metrics, and compare says so."""
    eps = [
        {"episode_id": 0, "metrics": {"success": True, "completed_subtasks": 5}},
        {"episode_id": 1, "metrics": {"success": False, "completed_subtasks": 2}},
        {"episode_id": 2, "metrics": {"success": False}, "failure_reason": "server_error"},
    ]
    body = {"benchmark": "calvin", "tasks": [{"task": "seq", "episodes": eps}]}
    cli.cmd_compare(_ns(_write(tmp_path / "a.json", body), metric="completed_subtasks", reference="4", tolerance=1))
    report = _json(capsys)
    assert report["a"]["num_errors"] == 1 and report["a"]["errors_without_metric"] == 1
    assert report["a"]["n"] == 2 and report["a"]["num_episodes"] == 3
    assert report["warnings"] == [
        "A: 1 harness error(s) record no 'completed_subtasks' and are left out of it "
        "(an error counts only as a success failure)"
    ]
    out = _text(tmp_path, capsys, body, metric="completed_subtasks", reference="4", tolerance=1)
    assert "⚠ A: 1 harness error(s)" in out


# ---------------------------------------------------------------------------
# Reference verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "verdict"), [("97.0%", "within_ci"), ("0.93", "within_tolerance"), ("80%", "outside_tolerance")]
)
def test_reference_verdict_on_a_single_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], reference: str, verdict: str
) -> None:
    a = _write(tmp_path / "a.json", _aggregate({"t": [1] * 485 + [0] * 15}))
    cli.cmd_compare(_ns(a, reference=reference))
    report = _json(capsys)
    assert "comparison" not in report and "b" not in report
    assert report["a"]["ci95"] == [round(x, 4) for x in stats.wilson_interval(485, 500)]
    assert report["reference"]["verdict"] == verdict and report["reference"]["tolerance"] == 0.05
    assert report["reference"]["symbol"] == stats.VERDICTS[verdict][0]


def test_non_rate_reference_needs_an_explicit_tolerance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _write(tmp_path / "a.json", _aggregate({"t": [3, 4, 5, 4]}, metric="chain"))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_compare(_ns(a, metric="chain", reference="4.43"))
    assert exc.value.code == 1
    cli.cmd_compare(_ns(a, metric="chain", reference="5.2", tolerance=1.5))
    assert _json(capsys)["reference"]["verdict"] == "within_tolerance"


def test_parse_reference() -> None:
    assert parse_reference("96.9%", binary=True) == pytest.approx(0.969)
    assert parse_reference("0.969", binary=True) == 0.969
    assert parse_reference("4.43", binary=False) == 4.43
    with pytest.raises(ValueError, match="append %"):
        parse_reference("96.9", binary=True)


@pytest.mark.parametrize(
    ("body", "args", "code"),
    [
        (None, {}, 2),  # neither B nor --reference
        ({"tasks": [{"task": "t", "episodes": []}]}, {"reference": "0.5"}, 1),  # no per-episode rows
        (_aggregate({"t": [1]}), {"reference": "50"}, 1),  # percent without %
        (_aggregate({"t": [1]}), {"reference": "0.5", "metric": "nope"}, 1),
        (_aggregate({"t": [3]}, metric="m"), {"reference": "3", "metric": "m", "tolerance": 1}, 1),  # no interval
    ],
)
def test_errors_exit_nonzero(tmp_path: Path, body: Any, args: dict[str, Any], code: int) -> None:
    path = _write(tmp_path / "a.json", body) if body is not None else str(tmp_path / "a.json")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_compare(_ns(path, **args))
    assert exc.value.code == code


def test_missing_file_exits_1(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.cmd_compare(_ns(str(tmp_path / "missing.json"), reference="0.5"))
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Rendering and wiring
# ---------------------------------------------------------------------------


def test_rich_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({"t1": [1, 1, 1], "t2": [1, 1, 0]})
    b = _aggregate({"t1": [0, 0, 0], "t2": [0, 0, 0]})
    out = _text(tmp_path, capsys, a, b, reference="10%")
    assert "6 pairs over 2 tasks" in out
    assert "Exact McNemar: A-only 5, B-only 0, p = 0.0625" in out
    assert "+83.3 pp" in out and "(Newcombe paired score)" in out
    assert "not adjusted for multiple comparisons" in out
    assert "🔧" in out


@pytest.mark.parametrize("columns", [80, 100, 130])
def test_per_task_table_fits_the_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, columns: int
) -> None:
    monkeypatch.setenv("COLUMNS", str(columns))
    names = [f"{w} put the white mug on the left plate and the yellow mug on the right" for w in ("one", "two", "six")]
    a = _aggregate({n: [1] * 45 + [0] * 5 for n in names})
    b = _aggregate({n: [1] * 30 + [0] * 20 for n in names})
    out = _text(tmp_path, capsys, a, b)
    table = [line for line in out.splitlines() if line.startswith(("┃", "│"))]
    assert table and all(len(line) <= columns for line in table)
    header = next(line for line in table if "task" in line)
    assert [c.strip() for c in header.strip("┃").split("┃")][1:] == ["n", "A", "B", "A − B", "95% CI", "p"]
    assert any("│ 50 │" in line and "90.0%" in line and "60.0%" in line and "+30.0" in line for line in table)
    assert any("│ one put the" in line for line in table)


def test_shared_task_name_prefix_moves_to_the_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """LIBERO-style names ("pick up the black bowl ...") stay distinguishable in an 80-column table."""
    monkeypatch.setenv("COLUMNS", "80")
    names = [
        f"pick up the black bowl {where} and place it on the plate" for where in ("on the stove", "in the drawer")
    ]
    a = _aggregate({n: [1, 1, 0] for n in names})
    b = _aggregate({n: [1, 0, 0] for n in names})
    out = _text(tmp_path, capsys, a, b)
    assert "(tasks: pick up the black bowl …)" in out
    assert "│ …on the stove" in out and "│ …in the drawer" in out


def test_parser_wires_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(cli, "cmd_compare", lambda args: called.append(args))
    monkeypatch.setattr("sys.argv", ["vla-eval", "compare", "a.json", "--reference", "96.9%", "--json"])
    cli.main()
    args = called[0]
    assert (args.a, args.b, args.reference, args.metric, args.tolerance, args.json) == (
        "a.json",
        None,
        "96.9%",
        "success",
        None,
        True,
    )
    assert args.allow_benchmark_mismatch is False


# ---------------------------------------------------------------------------
# Export round trip
# ---------------------------------------------------------------------------


def _recording(path: Path, outcomes: list[bool], *, error_at: int | None = None, tasks: int = 2) -> None:
    store = RecordingStore(path)
    store.upsert_eval_metadata("run-demo", "demo", {"benchmark": "demo", "metric_keys": {"success": "mean"}})
    for i, ok in enumerate(outcomes):
        task_idx = i % tasks
        store.upsert_episode_result(
            sid=f"s{i}",
            eid=f"e{i}",
            eval_id="run-demo",
            task_name=f"task{task_idx}",
            episode_id=i // tasks,
            status="error" if i == error_at else ("success" if ok else "failure"),
            metrics={"success": ok},
            steps=1,
            elapsed_sec=0.1,
            context={"task_idx": task_idx, "name": f"task{task_idx}", "suite": "demo_suite"},
            jsonl_path="",
            failure_reason="timeout" if i == error_at else None,
            failure_detail=None,
        )
    store.close()


def test_export_aggregates_carry_intervals_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from vla_eval.results.export import export_db

    outcomes_a = [True] * 16 + [False] * 4
    outcomes_b = [True] * 10 + [False] * 10
    _recording(tmp_path / "a.sqlite", outcomes_a, error_at=19)
    _recording(tmp_path / "b.sqlite", outcomes_b)
    (agg_a,) = export_db(tmp_path / "a.sqlite", tmp_path / "a")
    export_db(tmp_path / "b.sqlite", tmp_path / "b")

    assert agg_a["mean_success"] == 0.8 and agg_a["num_errors"] == 1  # the error stays a failure
    assert agg_a["n_success"] == 20
    assert agg_a["ci95_success"] == [round(x, 4) for x in stats.wilson_interval(16, 20)]
    for task in agg_a["tasks"]:
        assert task["n_success"] == 10
        assert task["ci95_success"] == [
            round(x, 4) for x in stats.wilson_interval(round(task["mean_success"] * 10), 10)
        ]
        assert all("task_idx" in ep for ep in task["episodes"])

    cli.cmd_compare(_ns(str(tmp_path / "a" / "demo_aggregate.json"), str(tmp_path / "b" / "demo_aggregate.json")))
    cmp = _json(capsys)["comparison"]
    assert cmp["pair_key"] == ["task", "suite", "episode_id"] and cmp["paired"] == 20
    assert cmp["mcnemar"] == {"a_only": 6, "b_only": 0, "p_value": 0.03125}


def test_export_of_a_task_subset_rerun_pairs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End to end: B recorded only task1, at task_idx 0 (the position after ``tasks:`` filtering)."""
    from vla_eval.results.export import export_db

    _recording(tmp_path / "a.sqlite", [True, False] * 10)  # task0 all True, task1 all False
    store = RecordingStore(tmp_path / "b.sqlite")
    store.upsert_eval_metadata("run-demo", "demo", {"benchmark": "demo", "metric_keys": {"success": "mean"}})
    for e in range(10):
        store.upsert_episode_result(
            sid=f"s{e}",
            eid=f"e{e}",
            eval_id="run-demo",
            task_name="task1",
            episode_id=e,
            status="success" if e < 3 else "failure",
            metrics={"success": e < 3},
            steps=1,
            elapsed_sec=0.1,
            context={"task_idx": 0, "name": "task1", "suite": "demo_suite"},
            jsonl_path="",
            failure_reason=None,
            failure_detail=None,
        )
    store.close()
    export_db(tmp_path / "a.sqlite", tmp_path / "a")
    export_db(tmp_path / "b.sqlite", tmp_path / "b")
    cli.cmd_compare(_ns(str(tmp_path / "a" / "demo_aggregate.json"), str(tmp_path / "b" / "demo_aggregate.json")))
    cmp = _json(capsys)["comparison"]
    assert cmp["method"] == "paired" and cmp["paired"] == 10 and cmp["tasks_only_a"] == 1
    assert cmp["mcnemar"] == {"a_only": 0, "b_only": 3, "p_value": 0.25}
    assert cmp["difference"] == -0.3
