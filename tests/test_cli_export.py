"""Recording export selects one DB and resumes tracking from saved metadata."""

from __future__ import annotations

import argparse
import json
import sqlite3

import pytest

from vla_eval.cli import main as cli
from vla_eval.recording import RecordingStore


def _database(path):
    store = RecordingStore(path)
    store.set_run_metadata("original-id", {"tracking": {"report_to": "wandb"}, "output_dir": "old-results"})
    store.upsert_eval_metadata("original-id-demo", "demo", {"benchmark": "demo", "metric_keys": {"success": "mean"}})
    store.upsert_episode_result(
        sid="s",
        eid="e",
        eval_id="original-id-demo",
        task_name="task",
        episode_id=0,
        status="success",
        metrics={"success": True},
        steps=1,
        elapsed_sec=0.1,
        context={},
        jsonl_path="episode.jsonl",
        failure_reason=None,
        failure_detail=None,
    )
    store.upsert_step_rows("s", "e", {0: {"reward": 1}})
    store.close()


@pytest.mark.parametrize("override", [False, True])
def test_export_exports_and_reports_saved_run(tmp_path, monkeypatch, override):
    db = tmp_path / "renamed.sqlite"
    _database(db)
    events = []

    class Tracker:
        def on_eval_begin(self, eval_id, config):
            events.append(("begin", eval_id, config))

        def on_benchmark_begin(self, *args):
            pass

        def on_benchmark_end(self, *args):
            pass

        def on_eval_end(self, aggregates):
            events.append(("end", aggregates))

        def close(self):
            events.append(("close",))

    def trackers(report_to):
        assert report_to == "wandb"
        return [Tracker()]

    monkeypatch.setattr("vla_eval.tracking.get_reporting_trackers", trackers)
    output = tmp_path / "exported" if override else tmp_path
    cli.cmd_export(argparse.Namespace(db=str(db), output_dir=str(output) if override else None))
    assert json.loads((output / "demo_aggregate.json").read_text())["mean_success"] == 1
    assert json.loads((output / "episode.jsonl").read_text()) == {"step": 0, "reward": 1}
    assert events[0][0:2] == ("begin", "original-id")
    assert events[0][2] == {"tracking": {"report_to": "wandb"}}
    assert events[-1] == ("close",)


def test_missing_db_fails_without_creating_it(tmp_path):
    db = tmp_path / "missing.sqlite"
    with pytest.raises(SystemExit) as exc:
        cli.cmd_export(argparse.Namespace(db=str(db), output_dir=None))
    assert exc.value.code == 1
    assert not db.exists()


def test_legacy_database_exports_without_tracking(tmp_path, monkeypatch):
    db = tmp_path / "legacy.sqlite"
    _database(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE run_metadata")

    def unexpected(*args):
        raise AssertionError("Legacy database must not invent tracker settings")

    monkeypatch.setattr("vla_eval.tracking.get_reporting_trackers", unexpected)
    cli.cmd_export(argparse.Namespace(db=str(db), output_dir=None))
    assert (tmp_path / "demo_aggregate.json").exists()


def test_export_parser_accepts_positional_db(tmp_path, monkeypatch):
    db = tmp_path / "input.sqlite"
    output = tmp_path / "exported"
    called = []
    monkeypatch.setattr(cli, "cmd_export", lambda args: called.append(args))
    monkeypatch.setattr("sys.argv", ["vla-eval", "export", str(db), "-o", str(output)])
    cli.main()
    assert called[0].db == str(db)
    assert called[0].output_dir == str(output)


@pytest.mark.parametrize("flag", ["--db", "--config", "--eval-id"])
def test_removed_export_flags_are_rejected(flag, monkeypatch):
    monkeypatch.setattr("sys.argv", ["vla-eval", "export", flag, "abc"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_merge_command_has_no_alias(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vla-eval", "merge", "recording.sqlite"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
