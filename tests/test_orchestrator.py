"""Tests for Orchestrator: integration and error handling paths.

With the SQLite-recording model, the orchestrator no longer writes JSON
files itself — its outputs are (1) the in-memory return value and (2) a
``recording-<eval-id>.sqlite`` file when ``no_save=False``. These tests
assert against the in-memory return value; ``vla-eval merge`` is tested
separately in ``tests/test_recording_sqlite.py``.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest
import websockets.exceptions

from vla_eval.orchestrator import Orchestrator

from tests.conftest import BrokenTracker, RecordingTracker, StubBenchmark


@pytest.mark.anyio
async def test_orchestrator_runs_to_completion(echo_server, tmp_path):
    """Echo server + stub benchmark → orchestrator returns a complete result."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "stub_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 3, "num_tasks": 2},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ):
        orchestrator = Orchestrator(config, no_save=True)
        results = await orchestrator.run()

    assert len(results) == 1
    result = results[0]
    assert result.get("mean_success") == pytest.approx(1.0)
    assert len(result["tasks"]) == 2
    assert "partial" not in result
    assert list(tmp_path.glob("*.sqlite")) == []


@pytest.mark.anyio
async def test_orchestrator_returns_partial_on_server_death(tmp_path):
    """Connection failure mid-run → return value carries partial=True."""

    call_count = 0

    class FlakyConnection:
        def __init__(self, url, **kwargs):
            self.url = url
            self.server_info = {}

        async def connect(self, **kwargs):
            pass

        async def close(self):
            pass

        async def start_episode(self, cfg):
            pass

        async def end_episode(self, result):
            pass

        async def act(self, obs):
            nonlocal call_count
            call_count += 1
            if call_count >= 10:
                raise websockets.exceptions.ConnectionClosed(None, None)
            return {"actions": np.ones(7, dtype=np.float32)}

        async def reconnect(self):
            raise ConnectionError("Server unreachable after retries")

    config = {
        "server": {"url": "ws://fake:9999"},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "partial_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 3, "num_tasks": 5},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ), patch(
        "vla_eval.orchestrator.Connection",
        FlakyConnection,
    ):
        orchestrator = Orchestrator(config, no_save=True)
        results = await orchestrator.run()

    assert len(results) == 1
    assert results[0].get("partial") is True
    total_episodes = sum(len(t["episodes"]) for t in results[0]["tasks"])
    assert total_episodes == 4


@pytest.mark.anyio
async def test_orchestrator_sharding_splits_work(echo_server, tmp_path):
    """Sharded runs produce disjoint subsets that cover all episodes."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "shard_test",
                "episodes_per_task": 3,
                "max_steps": 50,
                "params": {"done_at_step": 2, "num_tasks": 2},
            }
        ],
    }

    all_episode_keys: list[tuple[str, int]] = []
    for shard_id in range(3):
        with patch(
            "vla_eval.orchestrator.resolve_import_string",
            return_value=StubBenchmark,
        ):
            orch = Orchestrator(config, shard_id=shard_id, num_shards=3, no_save=True)
            results = await orch.run()

        for task_result in results[0]["tasks"]:
            for ep in task_result["episodes"]:
                all_episode_keys.append((task_result["task"], ep["episode_id"]))

    assert len(all_episode_keys) == 6
    assert len(set(all_episode_keys)) == 6


@pytest.mark.anyio
async def test_orchestrator_records_by_default_without_recording_block(echo_server, tmp_path):
    """Absent ``recording:`` still records episode results + step rows, with video off."""

    class StepRecordingStub(StubBenchmark):
        def step(self, action):
            res = super().step(action)
            self._recorder.record_step(reward=float(self._step_count))
            return res

    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "sqlite_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 2, "num_tasks": 1},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StepRecordingStub,
    ):
        orch = Orchestrator(config, eval_id="ev-test", no_save=False)
        await orch.run()

    db_path = tmp_path / "recording-ev-test.sqlite"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    eval_count = conn.execute("SELECT COUNT(*) FROM eval_metadata").fetchone()[0]
    episode_count = conn.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0]
    step_count = conn.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0]
    jsonl_path = conn.execute("SELECT jsonl_path FROM episode_results").fetchone()[0]
    conn.close()

    assert eval_count == 1
    assert episode_count == 1
    assert step_count == 2
    assert jsonl_path == "episodes/sqlite_test/task0000_ep0000_success.jsonl"
    assert list(tmp_path.rglob("*.mp4")) == []


@pytest.mark.anyio
async def test_orchestrator_records_steps_from_yaml_recording_block(echo_server, tmp_path):
    """A partial ``recording:`` block overrides paths while inheriting row defaults."""

    class StepRecordingStub(StubBenchmark):
        """StubBenchmark that pushes one step row per env step."""

        def step(self, action):
            res = super().step(action)
            self._recorder.record_step(reward=float(self._step_count))
            return res

    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "rec_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 3, "num_tasks": 1},
                "recording": {
                    "output_dir": str(tmp_path / "episodes"),
                    "filename_stem": "{name}_ep{episode_idx:04d}_{status}",
                },
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StepRecordingStub,
    ):
        orch = Orchestrator(config, eval_id="ev-rec", no_save=False)
        await orch.run()

    db = tmp_path / "recording-ev-rec.sqlite"
    assert db.exists()
    conn = sqlite3.connect(str(db))
    # Steps recorded via the orchestrator-built recorder.
    step_count = conn.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0]
    episode_count = conn.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0]
    # jsonl_path uses the template configured in YAML.
    jsonl_path = conn.execute("SELECT jsonl_path FROM episode_results").fetchone()[0]
    conn.close()
    assert step_count == 3  # done_at_step=3 → 3 steps recorded
    assert episode_count == 1
    assert jsonl_path == "episodes/task_0_ep0000_success.jsonl"


@pytest.mark.anyio
async def test_orchestrator_recording_block_overrides_default_step_recording(echo_server, tmp_path):
    """``recording.record_step: false`` disables step rows but keeps episode results."""

    class StepRecordingStub(StubBenchmark):
        def step(self, action):
            res = super().step(action)
            self._recorder.record_step(reward=float(self._step_count))
            return res

    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "default_rec",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
            },
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "no_steps",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
                "recording": {"record_step": False},
            },
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StepRecordingStub,
    ):
        orch = Orchestrator(config, eval_id="ev-mixed", no_save=False)
        await orch.run()

    db = tmp_path / "recording-ev-mixed.sqlite"
    assert db.exists()
    conn = sqlite3.connect(str(db))
    eval_rows = list(conn.execute("SELECT eval_id, safe_name FROM eval_metadata"))
    episode_rows = list(conn.execute("SELECT eval_id, task_name FROM episode_results"))
    step_count = conn.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0]
    conn.close()
    assert eval_rows == [("ev-mixed-default_rec", "default_rec"), ("ev-mixed-no_steps", "no_steps")]
    assert episode_rows == [("ev-mixed-default_rec", "task_0"), ("ev-mixed-no_steps", "task_0")]
    assert step_count == 1


@pytest.mark.anyio
async def test_orchestrator_default_recording_paths_do_not_collide_across_benchmarks(echo_server, tmp_path):
    """Default recording paths must include the benchmark identity.

    Several official configs contain many benchmark entries whose tasks reuse
    the same ``name``. A default filename based only on task name would make
    merge write multiple episodes to the same JSONL path.
    """

    class StepRecordingStub(StubBenchmark):
        def step(self, action):
            res = super().step(action)
            self._recorder.record_step(reward=float(self._step_count))
            return res

    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "bench_a",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
            },
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "bench_b",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
            },
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StepRecordingStub,
    ):
        orch = Orchestrator(config, eval_id="ev-paths", no_save=False)
        await orch.run()

    db = tmp_path / "recording-ev-paths.sqlite"
    conn = sqlite3.connect(str(db))
    rows = list(conn.execute("SELECT eval_id, jsonl_path, context FROM episode_results ORDER BY eval_id"))
    conn.close()

    assert [row[0] for row in rows] == ["ev-paths-bench_a", "ev-paths-bench_b"]
    paths = [row[1] for row in rows]
    assert len(paths) == len(set(paths))
    assert paths == [
        "episodes/bench_a/task0000_ep0000_success.jsonl",
        "episodes/bench_b/task0000_ep0000_success.jsonl",
    ]

    contexts = [json.loads(row[2]) for row in rows]
    assert all(context["name"] == "task_0" for context in contexts)
    assert all(
        "task_safe_name" not in context
        and "benchmark_safe_name" not in context
        and "task_idx" not in context
        and "episode_id" not in context
        for context in contexts
    )


@pytest.mark.anyio
async def test_orchestrator_default_recording_paths_are_stable_across_shards(echo_server, tmp_path):
    """Task indices are assigned before shard filtering, so shard paths are globally stable."""

    class StepRecordingStub(StubBenchmark):
        def step(self, action):
            res = super().step(action)
            self._recorder.record_step(reward=float(self._step_count))
            return res

    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "sharded_rec",
                "episodes_per_task": 3,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 2},
            }
        ],
    }

    for shard_id in range(2):
        with patch(
            "vla_eval.orchestrator.resolve_import_string",
            return_value=StepRecordingStub,
        ):
            orch = Orchestrator(config, shard_id=shard_id, num_shards=2, eval_id="ev-shards", no_save=False)
            await orch.run()

    db = tmp_path / "recording-ev-shards.sqlite"
    conn = sqlite3.connect(str(db))
    paths = [row[0] for row in conn.execute("SELECT jsonl_path FROM episode_results ORDER BY jsonl_path")]
    step_count = conn.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0]
    conn.close()

    assert paths == [
        "episodes/sharded_rec/task0000_ep0000_success.jsonl",
        "episodes/sharded_rec/task0000_ep0001_success.jsonl",
        "episodes/sharded_rec/task0000_ep0002_success.jsonl",
        "episodes/sharded_rec/task0001_ep0000_success.jsonl",
        "episodes/sharded_rec/task0001_ep0001_success.jsonl",
        "episodes/sharded_rec/task0001_ep0002_success.jsonl",
    ]
    assert step_count == 6


@pytest.mark.anyio
async def test_orchestrator_default_recording_paths_use_raw_episode_id_in_throughput_mode(echo_server, tmp_path):
    """Filenames use raw run episodes even when throughput mode wraps benchmark episode_idx."""

    class ThroughputStub(StubBenchmark):
        def get_metadata(self) -> dict:
            return {"max_steps": 50, "max_episodes_per_task": 1}

        def step(self, action):
            res = super().step(action)
            self._recorder.record_step(reward=float(self._step_count))
            return res

    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "throughput_rec",
                "episodes_per_task": 2,
                "max_steps": 50,
                "throughput_mode": True,
                "params": {"done_at_step": 1, "num_tasks": 1},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=ThroughputStub,
    ):
        orch = Orchestrator(config, eval_id="ev-throughput", no_save=False)
        await orch.run()

    db = tmp_path / "recording-ev-throughput.sqlite"
    conn = sqlite3.connect(str(db))
    rows = list(conn.execute("SELECT episode_id, jsonl_path, context FROM episode_results ORDER BY episode_id"))
    conn.close()

    assert [(row[0], row[1]) for row in rows] == [
        (0, "episodes/throughput_rec/task0000_ep0000_success.jsonl"),
        (1, "episodes/throughput_rec/task0000_ep0001_success.jsonl"),
    ]
    assert [json.loads(row[2])["episode_idx"] for row in rows] == [0, 0]


@pytest.mark.anyio
async def test_orchestrator_no_save_disables_recording_block(echo_server, tmp_path):
    """``--no-save`` wins even when a benchmark overrides recording."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "rec_disabled",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
                "recording": {"record_video": False, "record_step": True},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ):
        orch = Orchestrator(config, eval_id="ev-nosave", no_save=True)
        await orch.run()

    assert not (tmp_path / "recording-ev-nosave.sqlite").exists()


@pytest.mark.anyio
async def test_orchestrator_fires_tracker_lifecycle_on_success(echo_server, tmp_path):
    """A clean run drives eval_begin → benchmark_begin → episode_end(success)*
    → benchmark_end → eval_end → close."""
    tracker = RecordingTracker()
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "tracking": {"report_to": "wandb"},  # value is irrelevant; we patch the factory
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "ok",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 2, "num_tasks": 2},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ), patch("vla_eval.orchestrator.get_reporting_trackers", return_value=[tracker]):
        orch = Orchestrator(config, no_save=True)
        await orch.run()

    hooks = [c[0] for c in tracker.calls]
    assert hooks[0] == "on_eval_begin"
    assert hooks[1] == "on_benchmark_begin"
    # on_episode_end args: (bench_name, task_name, ep_dict, status) — status at index 3
    episode_calls = [c for c in tracker.calls if c[0] == "on_episode_end"]
    assert len(episode_calls) == 2
    assert all(c[1][3] == "success" for c in episode_calls), f"expected all success, got {episode_calls}"
    assert "on_benchmark_end" in hooks
    assert hooks[-2] == "on_eval_end"
    assert hooks[-1] == "close"


@pytest.mark.anyio
async def test_orchestrator_tracker_sees_error_status(tmp_path):
    """Server-death path: tracker must receive on_episode_end with status='error'
    so the wandb timeline reflects failed episodes, matching Recording's coverage."""

    class DyingConnection:
        def __init__(self, url, **kwargs):
            self.url = url
            self.server_info = {}

        async def connect(self, **kwargs):
            pass

        async def close(self):
            pass

        async def start_episode(self, cfg):
            pass

        async def end_episode(self, result):
            pass

        async def act(self, obs):
            raise websockets.exceptions.ConnectionClosed(None, None)

        async def reconnect(self):
            raise ConnectionError("server gone")

    tracker = RecordingTracker()
    config = {
        "server": {"url": "ws://fake:0"},
        "output_dir": str(tmp_path),
        "tracking": {"report_to": "wandb"},
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "err",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ), patch("vla_eval.orchestrator.Connection", DyingConnection), patch(
        "vla_eval.orchestrator.get_reporting_trackers", return_value=[tracker]
    ):
        orch = Orchestrator(config, no_save=True)
        await orch.run()

    # on_episode_end args at index 3 = status; reconnect path must produce status="error"
    error_episodes = [c for c in tracker.calls if c[0] == "on_episode_end" and c[1][3] == "error"]
    assert error_episodes, f"expected at least one status='error', got {tracker.calls}"


@pytest.mark.anyio
async def test_orchestrator_sharded_skips_live_hooks(echo_server, tmp_path):
    """Under sharding, the orchestrator must instantiate trackers (config errors
    fail fast on every shard) but skip live + eval-end hooks — merge owns those."""
    tracker = RecordingTracker()
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "tracking": {"report_to": "wandb"},
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "shard",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 2},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ), patch("vla_eval.orchestrator.get_reporting_trackers", return_value=[tracker]):
        orch = Orchestrator(config, shard_id=0, num_shards=2, no_save=True)
        await orch.run()

    assert tracker.calls == [], f"sharded mode should fire no live hooks; got {tracker.calls}"


@pytest.mark.anyio
async def test_orchestrator_isolates_broken_tracker(echo_server, tmp_path):
    """A backend that raises on every hook must not abort the eval."""
    good = RecordingTracker()
    bad = BrokenTracker()
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "tracking": {"report_to": ["wandb", "trackio"]},
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "robust",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ), patch("vla_eval.orchestrator.get_reporting_trackers", return_value=[bad, good]):
        orch = Orchestrator(config, no_save=True)
        results = await orch.run()

    assert len(results) == 1, "eval should have completed despite the broken backend"
    assert any(c[0] == "on_episode_end" for c in good.calls), "good tracker should still have fired"


@pytest.mark.anyio
async def test_orchestrator_fails_fast_on_bad_filename_template(echo_server, tmp_path):
    """A ``filename_stem`` that references a key the task dict doesn't have must
    fail at startup, not at the end of the first episode."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "bad_template",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 1, "num_tasks": 1},
                "recording": {
                    "output_dir": str(tmp_path / "episodes"),
                    "filename_stem": "{does_not_exist}_{status}",
                    "record_video": False,
                },
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ):
        orch = Orchestrator(config, eval_id="ev-bad", no_save=False)
        with pytest.raises(ValueError, match="filename_stem"):
            await orch.run()
