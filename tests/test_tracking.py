"""Tests for the ``report_to`` tracker integration.

Covers the dispatch / availability / robustness pieces directly, and runs an
end-to-end wandb integration test in offline mode (``WANDB_MODE=offline``) when
the ``wandb`` package is installed. The offline test asserts the on-disk run
artifact exists and that the harness-injected ``eval_id`` flowed through —
the same convergence handle the orchestrator and ``vla-eval merge`` rely on.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from vla_eval.tracking import (
    INTEGRATION_TO_TRACKER,
    Tracker,
    _episode_log_dict,
    _scalar_summary,
    call_each,
    get_reporting_trackers,
    is_trackio_available,
    is_wandb_available,
)

from tests.conftest import BrokenTracker, RecordingTracker


# ---------- get_reporting_trackers dispatch ----------


def test_get_reporting_trackers_none_or_empty_returns_empty() -> None:
    assert get_reporting_trackers(None) == []
    assert get_reporting_trackers("none") == []
    assert get_reporting_trackers([]) == []


def test_get_reporting_trackers_unknown_raises_with_supported_list() -> None:
    with pytest.raises(ValueError, match="not a supported tracker"):
        get_reporting_trackers("not_a_real_backend")
    with pytest.raises(ValueError, match="not a supported tracker"):
        get_reporting_trackers(["wandb", "not_a_real_backend"])


def test_get_reporting_trackers_all_picks_only_installed_backends() -> None:
    # Force both availability checks to False — "all" should give an empty list,
    # not error, even when no backend is installed.
    with mock.patch("vla_eval.tracking._IS_AVAILABLE", {"wandb": lambda: False, "trackio": lambda: False}):
        assert get_reporting_trackers("all") == []


# ---------- availability helpers ----------


def test_is_wandb_available_false_when_spec_missing() -> None:
    with mock.patch("importlib.util.find_spec", return_value=None):
        assert is_wandb_available() is False


def test_is_trackio_available_false_when_spec_missing() -> None:
    with mock.patch("importlib.util.find_spec", return_value=None):
        assert is_trackio_available() is False


# ---------- base Tracker is a complete no-op ----------


def test_base_tracker_hooks_dont_raise() -> None:
    t = Tracker()
    t.on_eval_begin("eval-id", {"some": "config"})
    t.on_benchmark_begin("bench", {})
    t.on_episode_end("bench", "task", {"metrics": {"success": True}}, "success")
    t.on_benchmark_end("bench", {"mean_success": 0.5})
    t.on_eval_end([])
    t.close()


# ---------- call_each robustness ----------


def test_call_each_isolates_per_tracker_errors() -> None:
    good = RecordingTracker()
    bad = BrokenTracker()
    # bad is first so good must still fire even when bad raised
    call_each([bad, good], "on_eval_begin", "eid", {"x": 1})
    assert good.calls == [("on_eval_begin", ("eid", {"x": 1}))]


def test_call_each_handles_unknown_hook_per_tracker() -> None:
    good = RecordingTracker()
    call_each([good], "nonexistent_hook")  # getattr raises AttributeError; call_each swallows
    assert good.calls == []


# ---------- log-dict / summary helpers ----------


def test_episode_log_dict_flattens_with_prefix() -> None:
    got = _episode_log_dict(
        "robosuite",
        "lift",
        {"metrics": {"success": True, "label": "ok"}, "steps": 42, "elapsed_sec": 0.5},
        "success",
    )
    assert got == {
        "robosuite/lift/status": "success",
        "robosuite/lift/success": 1.0,
        "robosuite/lift/steps": 42.0,
        "robosuite/lift/elapsed_sec": 0.5,
    }


def test_scalar_summary_drops_bools_and_strings() -> None:
    got = _scalar_summary({"mean_success": 0.75, "num_episodes": 4, "partial": True, "benchmark": "foo", "tasks": []})
    assert got == {"mean_success": 0.75, "num_episodes": 4}


# ---------- INTEGRATION_TO_TRACKER table ----------


def test_dispatch_table_keys_match_availability_checks() -> None:
    # Anyone adding a backend must wire both the class AND the availability check;
    # this guards the README claim that ``report_to: all`` picks installed backends.
    from vla_eval.tracking import _IS_AVAILABLE

    assert set(INTEGRATION_TO_TRACKER) == set(_IS_AVAILABLE)


# ---------- WandbTracker end-to-end in offline mode ----------


@pytest.mark.skipif(not is_wandb_available(), reason="wandb not installed")
def test_wandb_tracker_offline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the full lifecycle against wandb's offline mode.

    Asserts that:
    - The full sequence of hooks runs without raising.
    - ``on_eval_begin`` creates an offline run directory keyed by ``eval_id``
      (proves the harness-injected id flowed through to wandb.init).
    - The summary keys set by ``on_benchmark_end`` are visible on the live
      ``wandb.run.summary`` dict before ``finish``. (In offline mode wandb does
      not write a JSON-shaped summary file; the binary ``.wandb`` record is the
      source of truth, so we inspect the live object instead.)
    """
    from vla_eval.tracking import WandbTracker

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_PROJECT", "vla-eval-test")
    monkeypatch.setenv("WANDB_SILENT", "true")  # don't auth or sync

    eval_id = "test-eval-1234"

    t = WandbTracker()
    t.on_eval_begin(eval_id, {"output_dir": str(tmp_path), "benchmarks": []})
    t.on_episode_end(
        "robosuite",
        "lift",
        {"metrics": {"success": True}, "steps": 17, "elapsed_sec": 0.8},
        "success",
    )
    t.on_episode_end(
        "robosuite",
        "lift",
        {"metrics": {"success": False}, "steps": 30, "elapsed_sec": 1.5},
        "fail",
    )
    t.on_benchmark_end("robosuite", {"benchmark": "robosuite", "mean_success": 0.5, "num_episodes": 2})

    # Inspect the live summary before finish() drops `wandb.run`.
    assert t._wandb.run is not None
    summary_snapshot = dict(t._wandb.run.summary)

    t.on_eval_end([])
    t.close()

    # Run directory carries the eval_id we injected (proves the convergence handle works).
    run_dirs = list(tmp_path.glob("wandb/offline-run-*"))
    assert len(run_dirs) == 1, f"expected one offline run dir, found {run_dirs}"
    assert eval_id in run_dirs[0].name, f"eval_id not in run dir name: {run_dirs[0]}"

    assert summary_snapshot.get("robosuite/mean_success") == 0.5
    assert summary_snapshot.get("robosuite/num_episodes") == 2
