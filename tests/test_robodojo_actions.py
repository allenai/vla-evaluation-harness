"""RoboDojo action types: 14-D joint qpos vs 16-D end-effector poses, against a stand-in env (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.benchmarks.robodojo.benchmark import RoboDojoBenchmark
from vla_eval.recording import NullEpisodeRecorder
from vla_eval.specs import EE_POSE_DUAL_WXYZ, IMAGE_RGB, LANGUAGE, STATE_EE_POSE_DUAL_WXYZ, check_specs


class _Env:
    """The parts of RoboDojo's ``EvalEnv`` a round touches, recording what the policy sent."""

    def __init__(self, num_envs: int) -> None:
        self.end_flag = [False] * num_envs
        self.success = [False] * num_envs
        self.unstable_envs: list[int] = []
        self.physx_monitor_enabled = False
        self.sent: list[dict[str, np.ndarray]] = []

    def take_action_batch(self, actions: list[dict[str, np.ndarray]], env_idx: list[int]) -> None:
        """Keep the per-env action dicts ``validate_action_dict`` would have received."""
        assert len(actions) == len(env_idx)
        self.sent.extend(actions)

    def is_episode_end(self) -> None:
        return None

    def get_running_env_idx_list(self) -> list[int]:
        return [i for i, ended in enumerate(self.end_flag) if not ended]

    def get_obs_batch(self, env_idx: list[int]) -> list[dict[str, object]]:
        return [_raw_obs(i) for i in env_idx]


def _raw_obs(env_idx: int) -> dict[str, object]:
    """A raw observation with one camera and RoboDojo's per-arm state (both spellings)."""
    camera = {"color": np.zeros((2, 2, 3), np.uint8)}
    return {
        "env_idx": env_idx,
        "vision": {"cam_head": camera, "cam_left_wrist": camera, "cam_right_wrist": camera},
        "state": {
            "left_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": np.zeros(1),
            "right_arm_joint_state": np.zeros(6),
            "right_ee_joint_state": np.zeros(1),
            "left_ee_pose": np.zeros(7),
            "right_ee_pose": np.zeros(7),
        },
        "instruction": "stack the blocks",
    }


def _bench(**kwargs) -> RoboDojoBenchmark:
    bench = RoboDojoBenchmark(tasks=["stack_blocks"], **kwargs)
    bench._env = _Env(bench.num_envs)
    bench._task_name = "stack_blocks"
    bench._slots = list(range(bench.num_envs))
    bench._seeds = list(range(bench.num_envs))
    bench._recorders = {slot: NullEpisodeRecorder() for slot in bench._slots}
    return bench


def test_joint_actions_are_the_default() -> None:
    bench = _bench()
    sent = bench._unflatten_action(np.arange(14.0))
    assert sorted(sent) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    ]
    np.testing.assert_array_equal(sent["left_arm_joint_state"], np.arange(6.0))
    np.testing.assert_array_equal(sent["left_ee_joint_state"], [6.0])
    np.testing.assert_array_equal(sent["right_arm_joint_state"], np.arange(7.0, 13.0))
    np.testing.assert_array_equal(sent["right_ee_joint_state"], [13.0])


def test_ee_actions_use_the_pose_keys() -> None:
    sent = _bench(action_type="ee")._unflatten_action(np.arange(16.0))
    assert sorted(sent) == ["left_ee_joint_state", "left_ee_pose", "right_ee_joint_state", "right_ee_pose"]
    np.testing.assert_array_equal(sent["left_ee_pose"], np.arange(7.0))
    np.testing.assert_array_equal(sent["left_ee_joint_state"], [7.0])
    np.testing.assert_array_equal(sent["right_ee_pose"], np.arange(8.0, 15.0))
    np.testing.assert_array_equal(sent["right_ee_joint_state"], [15.0])


@pytest.mark.parametrize(("action_type", "dim"), [("joint", 16), ("ee", 14), ("ee", 20), ("joint", 0)])
def test_action_lengths_are_strict(action_type: str, dim: int) -> None:
    with pytest.raises(ValueError, match="action values"):
        _bench(action_type=action_type)._unflatten_action(np.zeros(dim))


def test_unknown_action_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid action_type"):
        RoboDojoBenchmark(tasks=["stack_blocks"], action_type="eef")


@pytest.mark.parametrize(
    ("action_type", "key", "arm_key", "arm_dim", "dim"),
    [("joint", "actions", "left_arm_joint_state", 6, 14), ("ee", "ee_pose", "left_ee_pose", 7, 16)],
)
def test_step_reads_the_column_action_type_names(
    action_type: str, key: str, arm_key: str, arm_dim: int, dim: int
) -> None:
    bench = _bench(action_type=action_type)
    action = np.arange(dim, dtype=np.float32)
    _, done = bench.step(bench._slots, {key: action[None]})
    assert not done.any()
    np.testing.assert_array_equal(bench._env.sent[-1][arm_key], action[:arm_dim])


def test_state_keeps_the_joint_packing_by_default() -> None:
    obs = _bench().make_obs(_raw_obs(0), {"name": "stack_blocks"})
    assert obs["state"].shape == (14,)
    assert "ee_state" not in obs


def test_ee_state_packs_poses_and_grippers() -> None:
    bench = _bench(action_type="ee")
    raw = _raw_obs(0)
    state = raw["state"]
    assert isinstance(state, dict)
    state["left_ee_pose"] = np.arange(7.0)
    state["left_ee_joint_state"] = np.array([0.5])
    state["right_ee_pose"] = np.arange(7.0, 14.0)
    state["right_ee_joint_state"] = np.array([1.0])
    obs = bench.make_obs(raw, {"name": "stack_blocks"})
    assert "state" not in obs
    np.testing.assert_array_equal(obs["ee_state"], [*range(7), 0.5, *range(7, 14), 1.0])


def test_ee_state_needs_world_ee_state() -> None:
    bench = _bench(action_type="ee")
    raw = _raw_obs(0)
    state = raw["state"]
    assert isinstance(state, dict)
    del state["left_ee_pose"]
    with pytest.raises(RuntimeError, match="world_ee_state"):
        bench.make_obs(raw, {"name": "stack_blocks"})


def test_specs_follow_action_type() -> None:
    server_action = {"ee_pose": EE_POSE_DUAL_WXYZ}
    server_obs = {"images": IMAGE_RGB, "ee_state": STATE_EE_POSE_DUAL_WXYZ, "language": LANGUAGE}
    ee = _bench(action_type="ee")
    assert check_specs(server_action, ee.get_action_spec(), server_obs, ee.get_observation_spec()) == []
    assert ee.get_metadata()["action_dim"] == 16
    joint = _bench()
    assert check_specs(server_action, joint.get_action_spec(), {}, {})  # a joint benchmark warns
    assert joint.get_metadata()["action_dim"] == 14
