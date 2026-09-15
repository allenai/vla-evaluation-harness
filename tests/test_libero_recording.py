"""LIBERO records the executed action and the proprioceptive state per step.

Runs without the LIBERO simulator: ``step()`` only touches ``_env``, ``_recorder`` and
``_quat_to_aa``, so a benchmark built with ``__new__`` and a fake env exercises the real call path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder, RecordingStore
from vla_eval.rotation import quat_to_axisangle

EEF_POS = np.array([0.1, -0.2, 0.9])
EEF_QUAT = np.array([0.0, 0.7071068, 0.0, 0.7071068])  # 90 deg about y, robosuite [x, y, z, w]
GRIPPER = np.array([0.03, -0.03])


class FakeEnv:
    def __init__(self) -> None:
        self.actions: list[list[float]] = []

    def step(self, action):
        self.actions.append(list(action))
        obs = {"robot0_eef_pos": EEF_POS, "robot0_eef_quat": EEF_QUAT, "robot0_gripper_qpos": GRIPPER}
        return obs, 0.0, False, {}


class CapturingRecorder(NullEpisodeRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def record_step(self, **fields: Any) -> None:  # type: ignore[override]
        self.rows.append(fields)


def _benchmark() -> tuple[LIBEROBenchmark, FakeEnv, CapturingRecorder]:
    bench = LIBEROBenchmark.__new__(LIBEROBenchmark)
    env, recorder = FakeEnv(), CapturingRecorder()
    bench._env = env  # type: ignore[assignment]
    bench._recorder = recorder
    bench._quat_to_aa = quat_to_axisangle
    return bench, env, recorder


def test_record_fields_include_action_and_state() -> None:
    assert {"reward", "done", "success", "action", "state"} <= LIBEROBenchmark._ALL_RECORD_FIELDS


def test_step_records_executed_action_and_proprio_state() -> None:
    bench, env, recorder = _benchmark()

    bench.step({"actions": np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.4], dtype=np.float32)})

    assert len(recorder.rows) == 1
    row = recorder.rows[0]
    assert set(row) == {"reward", "done", "success", "action", "state"}
    # The recorded action is the one the env executed: gripper discretized to +1.
    assert row["action"].dtype == np.float32 and row["action"].shape == (7,)
    np.testing.assert_allclose(row["action"], env.actions[0])
    assert row["action"][-1] == 1.0
    assert row["state"].dtype == np.float32 and row["state"].shape == (8,)
    expected = np.concatenate([EEF_POS, quat_to_axisangle(EEF_QUAT), GRIPPER])
    np.testing.assert_allclose(row["state"], expected, rtol=1e-6)


def test_recorded_gripper_is_discretized_like_the_executed_one() -> None:
    bench, env, recorder = _benchmark()

    bench.step({"actions": [0.0] * 6 + [-0.01]})

    assert env.actions[0][-1] == -1.0
    assert recorder.rows[0]["action"][-1] == -1.0


def test_step_fields_action_state_validate_against_allowed_fields(tmp_path) -> None:
    store = RecordingStore(tmp_path / "recording.sqlite")
    try:
        recorder = EpisodeRecorder(
            store=store,
            sid="s",
            eid="e",
            eval_id="ev",
            output_dir=tmp_path,
            filename_stem="ep",
            context={},
            step_fields=["action", "state"],
            allowed_fields=LIBEROBenchmark._ALL_RECORD_FIELDS,
        )
        recorder.record_step(reward=0.0, done=False, success=False, action=np.ones(7), state=np.zeros(8))
        assert set(recorder._steps[0]) == {"action", "state"}

        with pytest.raises(ValueError, match="Unknown step_fields"):
            EpisodeRecorder(
                store=store,
                sid="s",
                eid="e2",
                eval_id="ev",
                output_dir=tmp_path,
                filename_stem="ep",
                context={},
                step_fields=["velocity"],
                allowed_fields=LIBEROBenchmark._ALL_RECORD_FIELDS,
            )
    finally:
        store.close()
