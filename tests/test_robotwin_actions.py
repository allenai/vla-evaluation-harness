"""RoboTwin action types and end-pose state, against a stand-in env (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.benchmarks.robotwin.benchmark import RoboTwinBenchmark
from vla_eval.recording import NullEpisodeRecorder
from vla_eval.specs import EE_POSE_DUAL_WXYZ, IMAGE_RGB, LANGUAGE, STATE_EE_POSE_DUAL_WXYZ, check_specs


class _Env:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, str]] = []
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 400

    def take_action(self, action: np.ndarray, action_type: str) -> None:
        self.calls.append((action, action_type))
        self.take_action_cnt += 1

    def get_obs(self) -> None:
        return None


def _bench(**kwargs) -> tuple[RoboTwinBenchmark, _Env]:
    bench = RoboTwinBenchmark(task_name="grab_roller", **kwargs)
    bench._env = env = _Env()
    bench._recorder = NullEpisodeRecorder()
    return bench, env


def test_ee_action_passes_through_unchanged() -> None:
    bench, env = _bench(action_type="ee")
    act = np.arange(16, dtype=np.float32) / 10
    bench.step({"actions": act})
    sent, kind = env.calls[0]
    assert kind == "ee"
    np.testing.assert_array_equal(sent, act)


@pytest.mark.parametrize("dim", [14, 20])
def test_ee_action_rejects_other_lengths(dim: int) -> None:
    bench, env = _bench(action_type="ee")
    with pytest.raises(ValueError, match="16D"):
        bench.step({"actions": np.zeros(dim)})
    assert not env.calls


def test_qpos_action_keeps_trimming() -> None:
    bench, env = _bench()
    bench.step({"actions": np.arange(16.0)})
    sent, kind = env.calls[0]
    assert kind == "qpos"
    np.testing.assert_array_equal(sent, np.arange(14.0))


def test_ee_needs_the_curobo_planner() -> None:
    assert RoboTwinBenchmark(task_name="grab_roller").fast_init
    assert not RoboTwinBenchmark(task_name="grab_roller", action_type="ee").fast_init
    with pytest.raises(ValueError):
        RoboTwinBenchmark(task_name="grab_roller", action_type="eef")


def test_send_state_adds_end_pose_in_action_layout() -> None:
    bench, _ = _bench(action_type="ee", send_state=True)
    rgb = {"rgb": np.zeros((2, 2, 3), np.uint8)}
    raw = {
        "observation": {"head_camera": rgb, "left_camera": rgb, "right_camera": rgb},
        "joint_action": {"vector": np.zeros(14)},
        "endpose": {
            "left_endpose": [1, 2, 3, 1, 0, 0, 0],
            "left_gripper": 0.5,
            "right_endpose": [4, 5, 6, 0, 1, 0, 0],
            "right_gripper": 1.0,
        },
    }
    obs = bench.make_obs(raw, {"instruction": "x"})
    np.testing.assert_array_equal(obs["states"], [1, 2, 3, 1, 0, 0, 0, 0.5, 4, 5, 6, 0, 1, 0, 0, 1.0])
    assert "states" not in _bench()[0].make_obs(raw, {"instruction": "x"})


def test_ee_specs() -> None:
    bench, _ = _bench(action_type="ee", send_state=True)
    server_action = {"ee_pose": EE_POSE_DUAL_WXYZ}
    server_obs = {"head_camera": IMAGE_RGB, "ee_state": STATE_EE_POSE_DUAL_WXYZ, "language": LANGUAGE}
    assert check_specs(server_action, bench.get_action_spec(), server_obs, bench.get_observation_spec()) == []
    assert check_specs(server_action, _bench()[0].get_action_spec(), {}, {})  # a qpos benchmark warns
    assert bench.get_metadata()["action_dim"] == 16
