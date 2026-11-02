"""``vla-eval compare``: pairing, McNemar, independent fallback, the reproduction verdict, and export round trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from vla_eval.cli import main as cli
from vla_eval.recording import RecordingStore
from vla_eval.results import stats
from vla_eval.results.compare import compare_runs, parse_reference


def _aggregate(tasks: dict[str, list[Any]], *, task_idx: bool = True, metric: str = "success") -> dict[str, Any]:
    """An aggregate whose task ``i`` has one episode per value, ``episode_id`` = position."""
    return {
        "benchmark": "B",
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
        a=a, b=b, metric=kw.get("metric", "success"), reference=kw.get("reference"), tolerance=0.05, json=True
    )


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def test_paired_binary_uses_mcnemar_and_reports_per_task(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # t1: A wins 4 discordant pairs, B wins 1; t2: identical.
    a = _aggregate({"t1": [1, 1, 1, 1, 1, 0, 0, 1], "t2": [1, 0, 1, 0]})
    b = _aggregate({"t1": [0, 0, 0, 0, 1, 0, 1, 1], "t2": [1, 0, 1, 0]})
    cli.cmd_compare(_ns(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b)))
    report = _json(capsys)
    cmp = report["comparison"]
    assert report["binary"] is True
    assert cmp["method"] == "paired" and cmp["paired"] == 12
    assert cmp["unmatched_a"] == cmp["unmatched_b"] == 0
    assert cmp["pair_key"] == ["task", "task_idx", "episode_id"]
    assert cmp["mcnemar"] == {"a_only": 4, "b_only": 1, "p_value": 0.375}
    assert cmp["difference"] == round(3 / 12, 4)
    assert cmp["ci95"][0] <= cmp["difference"] <= cmp["ci95"][1]
    assert cmp["interval"] == "paired bootstrap, stratified by task"
    t1, t2 = cmp["tasks"]
    assert (t1["task"], t1["difference"], t1["p_value"]) == ("t1", 0.375, 0.375)
    assert (t2["difference"], t2["ci95"], t2["p_value"]) == (0.0, [0.0, 0.0], 1.0)


def test_unmatched_episodes_are_counted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _aggregate({"t1": [1, 0, 1], "t2": [1]})
    b = _aggregate({"t1": [1, 1]})
    cli.cmd_compare(_ns(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b)))
    cmp = _json(capsys)["comparison"]
    assert (cmp["paired"], cmp["unmatched_a"], cmp["unmatched_b"]) == (2, 2, 0)


def test_task_idx_ignored_unless_both_runs_record_it() -> None:
    a = _aggregate({"t": [1, 0]})
    b = _aggregate({"t": [1, 1]}, task_idx=False)
    cmp = compare_runs(a, b)
    assert cmp["pair_key"] == ["task", "episode_id"] and cmp["paired"] == 2
    b_shifted = _aggregate({"other": [], "t": [1, 1]})  # same task name, different task_idx → not the same state
    assert compare_runs(a, b_shifted)["paired"] == 0


def test_duplicate_keys_are_not_paired() -> None:
    a = _aggregate({"t": [1, 0]}, task_idx=False)
    a["tasks"][0]["episodes"][1]["episode_id"] = 0
    cmp = compare_runs(a, _aggregate({"t": [1, 0]}, task_idx=False))
    assert cmp["duplicate_keys"] == [2, 0] and cmp["method"] == "independent"


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


def test_non_binary_metric_uses_paired_bootstrap() -> None:
    a = _aggregate({f"seq{i}": [c] for i, c in enumerate([5, 4, 3, 5, 2])}, metric="chain")
    b = _aggregate({f"seq{i}": [c] for i, c in enumerate([4, 4, 1, 5, 2])}, metric="chain")
    cmp = compare_runs(a, b, "chain")
    assert cmp["difference"] == 0.6 and "mcnemar" not in cmp
    # One episode per task: pooled, not stratified (singleton strata never vary).
    assert cmp["interval"] == "paired bootstrap"
    assert cmp["ci95"][0] < 0.6 < cmp["ci95"][1]


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
    assert report["reference"]["verdict"] == verdict
    assert report["reference"]["symbol"] == stats.VERDICTS[verdict][0]


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


def test_rich_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _write(tmp_path / "a.json", _aggregate({"t": [1, 1, 1, 1, 1, 0]}))
    b = _write(tmp_path / "b.json", _aggregate({"t": [0, 0, 0, 0, 0, 0]}))
    cli.cmd_compare(argparse.Namespace(a=a, b=b, metric="success", reference="10%", tolerance=0.05, json=False))
    out = capsys.readouterr().out
    assert "6 pairs" in out
    assert "Exact McNemar: A-only 5, B-only 0, p = 0.0625" in out
    assert "+83.3 pp" in out
    assert "🔧" in out


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
        0.05,
        True,
    )


def _recording(path: Path, outcomes: list[bool], *, error_at: int | None = None) -> None:
    store = RecordingStore(path)
    store.upsert_eval_metadata("run-demo", "demo", {"benchmark": "demo", "metric_keys": {"success": "mean"}})
    for i, ok in enumerate(outcomes):
        task_idx = i % 2
        store.upsert_episode_result(
            sid=f"s{i}",
            eid=f"e{i}",
            eval_id="run-demo",
            task_name=f"task{task_idx}",
            episode_id=i // 2,
            status="error" if i == error_at else ("success" if ok else "failure"),
            metrics={"success": ok},
            steps=1,
            elapsed_sec=0.1,
            context={"task_idx": task_idx},
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
    assert cmp["pair_key"] == ["task", "task_idx", "episode_id"] and cmp["paired"] == 20
    assert cmp["mcnemar"] == {"a_only": 6, "b_only": 0, "p_value": 0.03125}
