"""Unit tests for the LADDER benchmark adapters (simulator-independent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from vla_eval import columns
from vla_eval.benchmarks.ladder.benchmark import (
    ROBOT_ARENA_LITE_TASKS,
    ROBOT_ARENA_TASKS,
    ROBOLAB_LITE_TASKS,
    ROBOLAB_TASKS,
    SIMPLER_LITE_TASKS,
    SIMPLER_TASKS,
    LadderRoboLabBenchmark,
    LadderRobotArenaBenchmark,
    LadderSimplerBenchmark,
)
from vla_eval.recording import NullEpisodeRecorder


@dataclass
class _FakeState:
    eef_pose: np.ndarray
    gripper_state: np.ndarray
    joints: np.ndarray | None = None


@dataclass
class _FakeObs:
    images: dict[str, np.ndarray]
    state: _FakeState
    metadata: dict[str, Any]


def test_ladder_task_counts_and_lite_subsets() -> None:
    """All 145 LADDER tasks and 10 LADDER-Lite tasks resolve across the three suites."""
    assert len(SIMPLER_TASKS) == 54
    assert len(ROBOT_ARENA_TASKS) == 47
    assert len(ROBOLAB_TASKS) == 44
    assert len(SIMPLER_TASKS) + len(ROBOT_ARENA_TASKS) + len(ROBOLAB_TASKS) == 145

    simpler = LadderSimplerBenchmark(tasks="lite")
    ra = LadderRobotArenaBenchmark(tasks="lite")
    rl = LadderRoboLabBenchmark(tasks="lite")
    assert [t["name"] for t in simpler.get_tasks()] == list(SIMPLER_LITE_TASKS)
    assert [t["name"] for t in ra.get_tasks()] == list(ROBOT_ARENA_LITE_TASKS)
    assert [t["name"] for t in rl.get_tasks()] == list(ROBOLAB_LITE_TASKS)
    assert len(SIMPLER_LITE_TASKS) + len(ROBOT_ARENA_LITE_TASKS) + len(ROBOLAB_LITE_TASKS) == 10

    with pytest.raises(ValueError, match="unknown task"):
        LadderSimplerBenchmark(tasks=["nonexistent_task"])


def test_ladder_wave_lifecycle_and_subgoal_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wave reset/step/result tracks per-slot subgoal progress and latches early success."""
    bench = LadderSimplerBenchmark(num_envs=2, tasks=["all2_plate"], expose_oracle=True)
    step_idx = 0

    def fake_reset_wave(task_name: str, episode_ids: list[int]) -> list[_FakeObs]:
        assert task_name == "all2_plate" and episode_ids == [0, 1]
        bench._last_eef_poses = [np.eye(4, dtype=np.float32) for _ in episode_ids]
        bench._last_joints = [np.zeros(7, dtype=np.float32) for _ in episode_ids]
        bench._last_actions = [None for _ in episode_ids]
        return [
            _FakeObs(
                images={"main": np.zeros((16, 16, 3), dtype=np.uint8)},
                state=_FakeState(np.eye(4, dtype=np.float32), np.array([1.0], dtype=np.float32)),
                metadata={
                    "instruction": "put the carrot and the spoon on the plate",
                    "done": False,
                    "truncated": False,
                    "num_subgoals_complete": 0,
                    "oracle": {"subtasks": ["put carrot on plate", "put spoon on plate"], "achieved": [False, False]},
                },
            )
            for _ in episode_ids
        ]

    def fake_step_wave(actions: list[Any]) -> list[_FakeObs]:
        nonlocal step_idx
        step_idx += 1
        assert len(actions) == 2
        # Slot 0 completes both goals at step 1; slot 1 completes 1/2 goals and truncates at step 2.
        return [
            _FakeObs(
                images={"main": np.zeros((16, 16, 3), dtype=np.uint8)},
                state=_FakeState(np.eye(4, dtype=np.float32), np.array([0.0], dtype=np.float32)),
                metadata={
                    "instruction": "put the carrot and the spoon on the plate",
                    "done": step_idx == 1,
                    "truncated": step_idx >= 2,
                    "num_subgoals_complete": 2 if step_idx == 1 else 1,
                    "oracle": {
                        "subtasks": ["put carrot on plate", "put spoon on plate"],
                        "achieved": [True, step_idx == 1],
                    },
                },
            ),
            _FakeObs(
                images={"main": np.zeros((16, 16, 3), dtype=np.uint8)},
                state=_FakeState(np.eye(4, dtype=np.float32), np.array([1.0], dtype=np.float32)),
                metadata={
                    "instruction": "put the carrot and the spoon on the plate",
                    "done": False,
                    "truncated": step_idx >= 2,
                    "num_subgoals_complete": 1,
                    "oracle": {
                        "subtasks": ["put carrot on plate", "put spoon on plate"],
                        "achieved": [True, False],
                    },
                },
            ),
        ]

    monkeypatch.setattr(bench, "_reset_wave", fake_reset_wave)
    monkeypatch.setattr(bench, "_step_wave", fake_step_wave)
    monkeypatch.setattr(bench, "_decode_action", lambda env_i, raw: raw)

    tasks = [{"name": "all2_plate", "episode_idx": 0}, {"name": "all2_plate", "episode_idx": 1}]
    obs = bench.reset([0, 1], tasks, [NullEpisodeRecorder(), NullEpisodeRecorder()])
    assert columns.num_rows(obs) == 2
    assert obs["state"].shape == (2, 7)
    assert len(obs["oracle"]) == 2

    # Step 1: slot 0 succeeds and finishes; slot 1 keeps running
    obs1, done1 = bench.step([0, 1], {"actions": np.zeros((2, 7), dtype=np.float32)})
    assert done1.tolist() == [True, False]
    assert bench.result(0) == {"success": True, "subgoal_progress": 1.0, "num_subgoals_complete": 2}

    # Step 2: only slot 1 is active; slot 1 truncates with 1/2 subgoals completed
    obs2, done2 = bench.step([1], {"actions": np.zeros((1, 7), dtype=np.float32)})
    assert columns.num_rows(obs2) == 1
    assert done2.tolist() == [True]
    assert bench.result(1) == {"success": False, "subgoal_progress": 0.5, "num_subgoals_complete": 1}
