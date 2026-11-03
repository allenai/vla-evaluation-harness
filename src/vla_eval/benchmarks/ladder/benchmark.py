"""LADDER long-horizon robot manipulation benchmark adapters (arXiv:2610.18053).

LADDER (https://github.com/insait-institute/ladder, https://ladderrobotics.ai) evaluates
policies across 145 long-horizon manipulation tasks (1 to 10 subgoals) in three suites:

* :class:`LadderSimplerBenchmark` (``SimplerEnv-Long``): 54 WidowX tasks on SAPIEN 3 / ManiSkill 3,
  24 episodes per task.
* :class:`LadderRobotArenaBenchmark` (``RobotArena-∞-Long``): 47 WidowX tasks on Genesis,
  8 episodes per task.
* :class:`LadderRoboLabBenchmark` (``RoboLab-Long``): 44 Franka DROID tasks on Isaac Sim 6 / Isaac Lab 3,
  8 episodes per task.

Each adapter subclasses :class:`VectorStepBenchmark` so a single simulator instance steps
``num_envs`` episodes in lockstep per wave while also supporting ``num_envs=1`` smoke runs.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import sys
from typing import Any, ClassVar, TypeVar

import numpy as np

from vla_eval import columns
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.dirs import assets_cache
from vla_eval.recording import EpisodeRecorder
from vla_eval.render import DEFAULT_RENDER_MODE, assert_lavapipe_vulkan, configure_sapien_render
from vla_eval.rotation import axisangle_to_matrix, matrix_to_quat, quat_to_axisangle
from vla_eval.specs import (
    IMAGE_RGB,
    LANGUAGE,
    RAW,
    DimSpec,
)
from vla_eval.types import Batch, EpisodeResult, Task

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_LADDER_ROOT = "/app/ladder"
SIMPLER_LAVAPIPE_ICD_ENV_VAR = "LADDER_SIMPLER_LAVAPIPE_ICD"
_CANONICAL_TO_WIDOWX_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def _intrinsic_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler angles -> 3x3 rotation matrix (matches ManiSkill3 ``euler_angles_to_matrix(..., 'XYZ')``)."""
    x, y, z = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    return np.array(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ],
        dtype=np.float64,
    )


SIMPLER_TASKS: tuple[str, ...] = (
    "all10_plate",
    "all10_towel",
    "all1_plate",
    "all1_towel",
    "all2_plate",
    "all2_towel",
    "all3_plate",
    "all3_towel",
    "all4_plate",
    "all4_towel",
    "all5_plate",
    "all5_towel",
    "all6_plate",
    "all6_towel",
    "all7_plate",
    "all7_towel",
    "all8_plate",
    "all8_towel",
    "all9_plate",
    "all9_towel",
    "except_eggplant_2",
    "except_eggplant_3",
    "except_eggplant_4",
    "except_eggplant_5",
    "move_plate_to_towel_2",
    "move_plate_to_towel_3",
    "move_plate_to_towel_4",
    "move_plate_to_towel_5",
    "move_towel_to_plate_3",
    "ordered_plate_2",
    "ordered_plate_3",
    "relay10e",
    "relay2e",
    "relay3e",
    "relay4e",
    "relay5e",
    "relay6e",
    "relay7e",
    "relay8e",
    "relay9e",
    "shuttle10",
    "shuttle7",
    "shuttle8",
    "shuttle9",
    "split2",
    "split3",
    "split4",
    "split5",
    "split6",
    "split7",
    "spoons2",
    "spoons3",
    "spoons4",
    "spoons5",
)
SIMPLER_LITE_TASKS: tuple[str, ...] = ("relay7e", "split3")

ROBOT_ARENA_TASKS: tuple[str, ...] = (
    "scene36_1goal",
    "scene36_2goal",
    "scene45_1goal",
    "scene45_2goal",
    "scene45_3goal",
    "scene45_4goal",
    "scene45_5goal",
    "scene45_6goal",
    "scene45_all_2",
    "scene45_all_3",
    "scene45_all_4",
    "scene45_all_5",
    "scene45_all_6",
    "scene45_except_2",
    "scene45_except_3",
    "scene45_except_4",
    "scene45_except_5",
    "scene45_except_6",
    "scene50_3goal",
    "scene54_1goal",
    "scene54_2goal",
    "scene54_2goal_ordered",
    "scene54_3goal",
    "scene54_3goal_ordered",
    "scene54_duplicates_2",
    "scene54_duplicates_3",
    "scene54_except",
    "scene54_except_3",
    "scene54_relay_2",
    "scene54_relay_4",
    "scene54_revisit_10",
    "scene54_revisit_3",
    "scene54_revisit_6",
    "scene54_revisit_7",
    "scene54_revisit_8",
    "scene54_revisit_9",
    "scene54_route_2",
    "scene54_route_4",
    "scene54_route_6",
    "scene54_route_7",
    "scene54_route_8",
    "scene54_split_2",
    "scene54_split_3",
    "scene54_transfer_2",
    "scene76_1goal",
    "scene76_2goal",
    "scene76_2goal_ordered",
)
ROBOT_ARENA_LITE_TASKS: tuple[str, ...] = (
    "scene36_1goal",
    "scene45_6goal",
    "scene54_split_3",
    "scene76_2goal",
)

ROBOLAB_TASKS: tuple[str, ...] = (
    "comp_bin_1",
    "comp_bin_10",
    "comp_bin_2",
    "comp_bin_2_distractor",
    "comp_bin_4_distractor",
    "comp_bin_6",
    "comp_identical_2",
    "comp_identical_4",
    "comp_ordered_bin_10",
    "comp_ordered_bin_2",
    "comp_ordered_bin_3",
    "comp_ordered_bin_4",
    "comp_ordered_bin_5",
    "comp_ordered_bin_6",
    "comp_ordered_bin_7",
    "comp_ordered_bin_8",
    "comp_ordered_bin_9",
    "comp_plate_1",
    "comp_relay_2",
    "comp_relay_3",
    "comp_relay_4",
    "comp_relay_5",
    "comp_revisit_1",
    "comp_revisit_2",
    "comp_revisit_3",
    "comp_route_10",
    "comp_route_2",
    "comp_route_2_binfirst",
    "comp_route_3",
    "comp_route_4",
    "comp_route_5",
    "comp_route_6",
    "comp_route_7",
    "comp_route_8",
    "comp_route_9",
    "comp_split_plate_2",
    "comp_split_plate_3",
    "comp_split_plate_4",
    "comp_split_plate_5",
    "comp_split_plate_6",
    "comp_split_plate_7",
    "comp_split_plate_8",
    "comp_transfer_2",
    "comp_transfer_3",
)
ROBOLAB_LITE_TASKS: tuple[str, ...] = (
    "comp_ordered_bin_4",
    "comp_relay_3",
    "comp_route_10",
    "comp_transfer_2",
)

# Published 8-episode object XY offset grid (widowx_composition_k8.yaml).
ROBOT_ARENA_K8_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.03, -0.02),
    (-0.03, +0.02),
    (-0.01, -0.02),
    (-0.01, +0.02),
    (+0.01, -0.02),
    (+0.01, +0.02),
    (+0.03, -0.02),
    (+0.03, +0.02),
)


def _ensure_ladder_on_path(root: Path) -> None:
    """Ensure the ``ladder`` package is importable from *root* when not already installed."""
    try:
        import ladder  # noqa: F401

        return
    except ImportError:
        pass
    if not (root / "ladder").is_dir():
        raise FileNotFoundError(
            f"LADDER source tree not found at {root}. Set LADDER_ROOT or pass root=... in benchmark params."
        )
    for p in (root / "build_support" / "python", root):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


class _LadderVectorBenchmark(VectorStepBenchmark):
    """Shared lockstep wave execution, observation packing, and metric aggregation for LADDER suites."""

    partial_reset: ClassVar[bool] = False
    task_exclusive: ClassVar[bool] = True
    _ALL_RECORD_FIELDS = frozenset(
        {"reward", "done", "truncated", "success", "subgoal_progress", "num_subgoals_complete"}
    )

    SUITE_NAME: ClassVar[str] = "ladder"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = ()
    LITE_TASKS: ClassVar[tuple[str, ...]] = ()
    DEFAULT_MAX_STEPS: ClassVar[int] = 1000

    def __init__(
        self,
        num_envs: int = 8,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "delta_ee",
        control_mode: str | None = None,
        max_episode_steps: int | None = None,
        success_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
        expose_oracle: bool = False,
        expose_privileged: bool = False,
        publish_depth: bool = False,
    ) -> None:
        super().__init__(num_envs)
        if isinstance(tasks, str):
            resolved_tasks = list(self.LITE_TASKS if tasks == "lite" else [tasks])
        elif tasks is None:
            resolved_tasks = list(self.DEFAULT_TASKS)
        elif list(tasks) == ["lite"]:
            resolved_tasks = list(self.LITE_TASKS)
        else:
            resolved_tasks = list(tasks)
        unknown = [t for t in resolved_tasks if t not in self.DEFAULT_TASKS]
        if unknown:
            raise ValueError(f"{type(self).__name__}: unknown task(s) {unknown}")
        if control_mode and "target_base_pose" in control_mode:
            action_mode = "abs_ee"
        if action_mode not in ("delta_ee", "abs_ee", "joint_pos", "joint_vel"):
            raise ValueError(
                f"Invalid action_mode={action_mode!r}. Expected 'delta_ee', 'abs_ee', 'joint_pos', or 'joint_vel'."
            )
        self._task_names = resolved_tasks
        self._root = Path(root or os.environ.get("LADDER_ROOT", DEFAULT_LADDER_ROOT)).expanduser().resolve()
        self._assets_dir = Path(assets_dir).expanduser().resolve() if assets_dir else None
        self._action_mode = action_mode
        self._control_mode = control_mode
        self._max_episode_steps = max_episode_steps
        self._success_mode = success_mode
        self._terminate_on_success = bool(terminate_on_success)
        self._send_state = bool(send_state)
        self._send_wrist_image = bool(send_wrist_image)
        self._expose_oracle = bool(expose_oracle)
        self._expose_privileged = bool(expose_privileged)
        self._publish_depth = bool(publish_depth)

        self._active_slots: list[int] = []
        self._slot_to_env: dict[int, int] = {}
        self._recorders: dict[int, EpisodeRecorder] = {}
        self._results: dict[int, EpisodeResult] = {}
        self._success_latched: dict[int, bool] = {}
        self._last_eef_poses: list[np.ndarray] = []
        self._last_target_poses: list[np.ndarray] = []
        self._last_joints: list[np.ndarray | None] = []
        self._last_actions: list[Any] = []
        self._last_obs_dicts: dict[int, dict[str, Any]] = {}
        self._current_task_name: str = ""

    def get_tasks(self) -> list[Task]:
        return [{"name": name, "suite": self.SUITE_NAME} for name in self._task_names]

    def get_metric_keys(self) -> dict[str, str]:
        return {
            "success": "mean",
            "subgoal_progress": "mean",
            "num_subgoals_complete": "mean",
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            "suite": self.SUITE_NAME,
            "max_steps": self.DEFAULT_MAX_STEPS,
            "num_envs": self.num_envs,
            "action_mode": self._action_mode,
        }

    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch:
        names = {str(t["name"]) for t in tasks}
        if len(names) != 1:
            raise RuntimeError(f"{type(self).__name__}: one wave must run one task, got {sorted(names)}")
        task_name = names.pop()
        self._current_task_name = task_name
        episode_ids = [int(t.get("episode_idx", i)) for i, t in enumerate(tasks)]
        self._active_slots = list(slots)
        self._slot_to_env = {slot: i for i, slot in enumerate(slots)}
        self._recorders = dict(zip(slots, recorders))
        self._results = {}
        self._success_latched = {slot: False for slot in slots}
        self._last_obs_dicts = {}

        raw_obs_list = self._reset_wave(task_name, episode_ids)
        for slot in slots:
            env_i = self._slot_to_env[slot]
            obs_dict, frame, metrics, _ = self._unpack_env_obs(raw_obs_list[env_i], slot)
            self._last_obs_dicts[slot] = obs_dict
            self._results[slot] = metrics
            self._recorders[slot].record_video(frame)
        return columns.stack([self._last_obs_dicts[s] for s in slots])

    def step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]:
        rows = dict(zip(slots, columns.split(actions, len(slots))))
        for slot in slots:
            env_i = self._slot_to_env[slot]
            raw_act = rows[slot].get("actions", rows[slot].get("action"))
            self._last_actions[env_i] = self._decode_action(env_i, np.asarray(raw_act, dtype=np.float32).reshape(-1))

        raw_obs_list = self._step_wave(self._last_actions)
        done_flags = np.zeros(len(slots), dtype=bool)
        for i, slot in enumerate(slots):
            env_i = self._slot_to_env[slot]
            obs_dict, frame, metrics, done = self._unpack_env_obs(raw_obs_list[env_i], slot)
            self._last_obs_dicts[slot] = obs_dict
            self._results[slot] = metrics
            done_flags[i] = done
            rec = self._recorders[slot]
            rec.record_video(frame)
            rec.record_step(
                reward=float(metrics["success"]),
                done=bool(done),
                truncated=bool(metrics.get("_truncated", False)),
                success=bool(metrics["success"]),
                subgoal_progress=float(metrics["subgoal_progress"]),
                num_subgoals_complete=int(metrics["num_subgoals_complete"]),
            )
            metrics.pop("_truncated", None)
        return columns.stack([self._last_obs_dicts[s] for s in slots]), done_flags

    def result(self, slot: int) -> EpisodeResult:
        res = dict(self._results.get(slot, {"success": False, "subgoal_progress": 0.0, "num_subgoals_complete": 0}))
        res.pop("_truncated", None)
        return res

    def _unpack_env_obs(
        self, raw_obs: Any, slot: int
    ) -> tuple[dict[str, Any], np.ndarray | None, EpisodeResult, bool]:
        """Convert a ``RobotObservation`` or RoboLab observation dict into wire format + metrics."""
        if isinstance(raw_obs, dict):
            images = {k: np.asarray(v, dtype=np.uint8) for k, v in (raw_obs.get("images") or {}).items()}
            state_obj = raw_obs.get("state") or {}
            eef_pose = np.asarray(state_obj.get("eef_pose", np.eye(4)), dtype=np.float32)
            target_pose = np.asarray(state_obj.get("eef_pose_from_target", eef_pose), dtype=np.float32)
            gripper = float(np.asarray(state_obj.get("gripper_state", [1.0])).reshape(-1)[0])
            joints = state_obj.get("joints")
            metadata = raw_obs.get("metadata") or {}
        else:
            images = {k: np.asarray(v, dtype=np.uint8) for k, v in (raw_obs.images or {}).items()}
            eef_pose = np.asarray(raw_obs.state.eef_pose, dtype=np.float32)
            target_raw = getattr(raw_obs.state, "eef_pose_from_target", None)
            target_pose = np.asarray(target_raw if target_raw is not None else eef_pose, dtype=np.float32)
            gripper = float(np.asarray(raw_obs.state.gripper_state).reshape(-1)[0])
            joints = getattr(raw_obs.state, "joints", None)
            metadata = raw_obs.metadata or {}

        if self.SUITE_NAME != "ladder_robolab" and "main" in images:
            # WidowX Bridge has a single third-person camera; RobotArena's ObservationBuilder
            # populates a dummy all-zero "wrist" array that would feed a blank second view to VLAs.
            images = {"main": images["main"]}

        env_i = self._slot_to_env[slot]
        self._last_eef_poses[env_i] = eef_pose
        if env_i < len(self._last_target_poses):
            self._last_target_poses[env_i] = target_pose
        self._last_joints[env_i] = np.asarray(joints, dtype=np.float32).reshape(-1)[:7] if joints is not None else None

        step_done = bool(metadata.get("done", False))
        truncated = bool(metadata.get("truncated", False))
        if step_done:
            self._success_latched[slot] = True
        success = self._success_latched[slot] if self._terminate_on_success else step_done

        oracle = metadata.get("oracle") if isinstance(metadata.get("oracle"), dict) else None
        achieved = [bool(v) for v in oracle.get("achieved", [])] if oracle else []
        n_goals = len(achieved)
        completed = int(metadata.get("num_subgoals_complete", sum(achieved)))
        if success and n_goals > 0:
            completed = max(completed, n_goals)
        progress = float(completed) / float(n_goals) if n_goals > 0 else float(success)

        obs_out: dict[str, Any] = {
            "images": images,
            "task_description": str(metadata.get("instruction", "")),
            "task_name": self._current_task_name,
        }
        if self._send_state:
            if joints is not None and self.SUITE_NAME == "ladder_robolab":
                j = np.asarray(joints, dtype=np.float32).reshape(-1)[:7]
                # DROID convention: gripper 0.0 = open, 1.0 = closed (LADDER stores 1.0 = open).
                obs_out["state"] = np.concatenate([j, np.array([1.0 - gripper], dtype=np.float32)])
                obs_out["states"] = obs_out["state"]
            else:
                rotvec = quat_to_axisangle(matrix_to_quat(eef_pose[:3, :3]))
                state_vec = np.concatenate([eef_pose[:3, 3], rotvec, np.array([gripper], dtype=np.float32)])
                obs_out["state"] = state_vec
                obs_out["states"] = state_vec
                obs_out["base_pose"] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                obs_out["tcp_pose"] = np.concatenate(
                    [eef_pose[:3, 3], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
                )
        if self._expose_oracle and oracle is not None:
            obs_out["oracle"] = oracle
        if self._expose_privileged and "privileged_state" in metadata:
            obs_out["privileged_state"] = metadata["privileged_state"]

        frame = images.get("main")
        if frame is None and images:
            frame = next(iter(images.values()))
        done = truncated or (self._terminate_on_success and step_done)
        metrics: EpisodeResult = {
            "success": bool(success),
            "subgoal_progress": round(progress, 6),
            "num_subgoals_complete": int(completed),
            "_truncated": bool(truncated),
        }
        return obs_out, frame, metrics, done

    def _decode_action(self, env_i: int, raw: np.ndarray) -> Any:
        """Convert a flat 7-D EE or 8-D joint action vector into a ``ladder`` ``RobotAction``."""
        from ladder.components.inference.api.core.action import (
            EndEffectorPoseAction,
            JointAnglesAction,
            ManipulatorAction,
        )

        if raw.size == 8 and (self._action_mode in ("joint_pos", "joint_vel") or self.SUITE_NAME == "ladder_robolab"):
            arm = raw[:7].astype(np.float32)
            cur_j = self._last_joints[env_i] if env_i < len(self._last_joints) else None
            if self._action_mode == "joint_vel" and cur_j is not None:
                joints = (cur_j + np.clip(arm, -1.0, 1.0) * 0.2).astype(np.float32)
            else:
                joints = arm
            # DROID convention: action[-1] > 0.5 means closed -> LADDER gripper 0.0 (closed), else 1.0 (open).
            gripper = 0.0 if float(raw[7]) > 0.5 else 1.0
            return ManipulatorAction(arm_action=JointAnglesAction(angles=joints), gripper_action=gripper)

        if raw.size < 7:
            raise ValueError(f"{type(self).__name__} expected at least a 7-D action, got shape {raw.shape}")
        pos = raw[:3].astype(np.float32)
        rot_in = raw[3:6].astype(np.float64)
        gripper = float(1.0 if raw[6] > 0.0 else 0.0) if raw[6] < 0.0 else float(np.clip(raw[6], 0.0, 1.0))
        target_eef = (
            self._last_target_poses[env_i] if env_i < len(self._last_target_poses) else self._last_eef_poses[env_i]
        )
        if self._action_mode == "delta_ee":
            translation = (target_eef[:3, 3] + pos).astype(np.float32)
            r_delta = axisangle_to_matrix(rot_in)
            rotation = (r_delta @ target_eef[:3, :3]).astype(np.float32)
            if env_i < len(self._last_target_poses):
                new_target = np.eye(4, dtype=np.float32)
                new_target[:3, 3] = translation
                new_target[:3, :3] = rotation
                self._last_target_poses[env_i] = new_target
        elif self._control_mode and "target_base_pose" in self._control_mode:
            # SimplerEnv absolute base-pose controller (e.g. X-VLA simpler_widowx): rot_in is WidowX
            # controller-frame intrinsic XYZ Euler; convert to LADDER canonical frame.
            translation = pos
            r_widowx = _intrinsic_xyz_to_matrix(rot_in)
            rotation = (r_widowx @ _CANONICAL_TO_WIDOWX_ROTATION.T).astype(np.float32)
        else:
            translation = pos
            rotation = _intrinsic_xyz_to_matrix(rot_in).astype(np.float32)
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(translation=translation, rotation=rotation),
            gripper_action=gripper,
        )

    def _hold_robot_action(self, eef_pose: np.ndarray, joints: np.ndarray | None = None) -> Any:
        from ladder.components.inference.api.core.action import (
            EndEffectorPoseAction,
            JointAnglesAction,
            ManipulatorAction,
        )

        if joints is not None and self.SUITE_NAME == "ladder_robolab":
            return ManipulatorAction(
                arm_action=JointAnglesAction(angles=np.asarray(joints, dtype=np.float32)[:7]),
                gripper_action=1.0,
            )
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=eef_pose[:3, 3].copy(),
                rotation=eef_pose[:3, :3].copy(),
            ),
            gripper_action=1.0,
        )

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        raise NotImplementedError

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        raise NotImplementedError


class LadderSimplerBenchmark(_LadderVectorBenchmark):
    """LADDER ``SimplerEnv-Long`` benchmark (WidowX on SAPIEN 3 / ManiSkill 3, 54 tasks × 24 episodes)."""

    SUITE_NAME: ClassVar[str] = "ladder_simpler"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = SIMPLER_TASKS
    LITE_TASKS: ClassVar[tuple[str, ...]] = SIMPLER_LITE_TASKS
    DEFAULT_MAX_STEPS: ClassVar[int] = 1000
    step_dt = 0.2

    render_backends = frozenset({"gpu", "cpu"})
    _render_mode: ClassVar[str] = DEFAULT_RENDER_MODE

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        applied = configure_sapien_render(mode, SIMPLER_LAVAPIPE_ICD_ENV_VAR)
        cls._render_mode = mode
        return applied

    def __init__(
        self,
        num_envs: int = 24,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "delta_ee",
        control_mode: str | None = None,
        max_episode_steps: int | None = None,
        success_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
        expose_oracle: bool = False,
        expose_privileged: bool = False,
        publish_depth: bool = False,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            tasks=tasks,
            root=root,
            assets_dir=assets_dir,
            action_mode=action_mode,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
            success_mode=success_mode,
            terminate_on_success=terminate_on_success,
            send_state=send_state,
            send_wrist_image=send_wrist_image,
            expose_oracle=expose_oracle,
            expose_privileged=expose_privileged,
            publish_depth=publish_depth,
        )
        self._env: Any = None
        self._current_key: tuple[str, int] | None = None

    def _ensure_assets_env(self) -> None:
        if "MS_ASSET_DIR" in os.environ:
            return
        candidates = [
            self._assets_dir,
            Path(os.environ["LADDER_SIMPLER_ASSETS"]) if "LADDER_SIMPLER_ASSETS" in os.environ else None,
            self._root / "ladder" / "components" / "inference" / "robots" / "simpler" / "maniskill3",
            assets_cache("ladder_simpler"),
        ]
        for cand in candidates:
            if cand is not None and cand.is_dir():
                os.environ["MS_ASSET_DIR"] = str(cand)
                return

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        if self._render_mode == "cpu":
            assert_lavapipe_vulkan(type(self).__name__)
        _ensure_ladder_on_path(self._root)
        self._ensure_assets_env()

        import gymnasium as gym
        import torch

        from ladder.components.inference.robots.simpler.env_builder import ABS_CONTROL_MODE
        from ladder.components.inference.robots.simpler.simpler_env import SimplerEnv, SimplerEnvConfig

        n_envs = len(episode_ids)
        key = (task_name, n_envs)
        if self._env is None or self._current_key != key:
            if self._env is not None:
                self._env.close()
                self._env = None
            cfg = SimplerEnvConfig(
                env_name=f"{task_name}-v0",
                robot_name="widowx",
                scene_name="bridge_table_1_v1",
                rgb_overlay_path="",
                sim_freq=500,
                control_mode=ABS_CONTROL_MODE,
                control_freq=5,
                max_episode_steps=0,
                raytracing=False,
                n_envs=n_envs,
                episode_layouts=max(24, max(episode_ids, default=0) + 1),
                expose_privileged=self._expose_privileged,
                publish_depth=self._publish_depth,
            )
            if self._render_mode == "cpu" or not torch.cuda.is_available():
                from mani_skill.utils.structs.types import SimConfig

                def _create_cpu_env(self_env: Any) -> Any:
                    return gym.make(
                        f"{task_name}-v1",
                        num_envs=n_envs,
                        obs_mode="rgb+depth+segmentation" if self._publish_depth else "rgb+segmentation",
                        sim_backend="cpu",
                        control_mode=ABS_CONTROL_MODE,
                        max_episode_steps=None,
                        sim_config=SimConfig(sim_freq=500, control_freq=5),
                        n_layouts=cfg.episode_layouts,
                    )

                SimplerEnv._create_env = _create_cpu_env  # type: ignore[method-assign]
            self._env = SimplerEnv(cfg)
            self._current_key = key

        obs_list = self._env.reset(episode_ids)
        self._last_eef_poses = [np.asarray(o.state.eef_pose, dtype=np.float32) for o in obs_list]
        self._last_target_poses = [
            np.asarray(getattr(o.state, "eef_pose_from_target", o.state.eef_pose), dtype=np.float32) for o in obs_list
        ]
        self._last_joints = [
            np.asarray(o.state.joints, dtype=np.float32)[:7] if getattr(o.state, "joints", None) is not None else None
            for o in obs_list
        ]
        self._last_actions = [self._hold_robot_action(p) for p in self._last_target_poses]
        return obs_list

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        assert self._env is not None
        return self._env.step(actions)

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "main": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = RAW
        return spec

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            self._current_key = None


class LadderRobotArenaBenchmark(_LadderVectorBenchmark):
    """LADDER ``RobotArena-∞-Long`` benchmark (WidowX on Genesis, 47 tasks × 8 episodes)."""

    SUITE_NAME: ClassVar[str] = "ladder_robot_arena"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = ROBOT_ARENA_TASKS
    LITE_TASKS: ClassVar[tuple[str, ...]] = ROBOT_ARENA_LITE_TASKS
    DEFAULT_MAX_STEPS: ClassVar[int] = 1000
    step_dt = 0.2

    def __init__(
        self,
        num_envs: int = 8,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "delta_ee",
        control_mode: str | None = None,
        max_episode_steps: int | None = None,
        success_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
        expose_oracle: bool = False,
        expose_privileged: bool = False,
        publish_depth: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            tasks=tasks,
            root=root,
            assets_dir=assets_dir,
            action_mode=action_mode,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
            success_mode=success_mode,
            terminate_on_success=terminate_on_success,
            send_state=send_state,
            send_wrist_image=send_wrist_image,
            expose_oracle=expose_oracle,
            expose_privileged=expose_privileged,
            publish_depth=publish_depth,
        )
        self._seed = int(seed)
        self._env: Any = None
        self._padded_single = False

    def _resolve_data_root(self) -> str:
        candidates = [
            self._assets_dir,
            Path(os.environ["LADDER_ROBOT_ARENA_ASSETS"]) if "LADDER_ROBOT_ARENA_ASSETS" in os.environ else None,
            self._root
            / "ladder"
            / "components"
            / "inference"
            / "robots"
            / "robot_arena"
            / "robot_arena_data"
            / "examples"
            / "data",
            assets_cache("ladder_robot_arena") / "examples" / "data",
            assets_cache("ladder_robot_arena"),
        ]
        for cand in candidates:
            if cand is not None and ((cand / "bridge").is_dir() or (cand / "scenes").is_dir()):
                return str(cand)
        raise FileNotFoundError(
            "RobotArena assets not found (expected a directory containing 'bridge/'). "
            "Set LADDER_ROBOT_ARENA_ASSETS or mount to ~/.cache/vla-eval/assets/ladder_robot_arena."
        )

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
        from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnv, RobotArenaEnvConfig
        from ladder.components.inference.robots.robot_arena.scene_builder import set_genesis_seed
        from ladder.core.compute.seeder import SEEDER

        SEEDER.seed_everything(self._seed)
        set_genesis_seed(self._seed)

        comp_task = COMPOSITION_TASKS[task_name]
        offsets = [ROBOT_ARENA_K8_OFFSETS[ep % len(ROBOT_ARENA_K8_OFFSETS)] for ep in episode_ids]
        # RobotArenaEnv's K>=2 path requires len(per_env_object_offsets) >= 2 and skips overlap
        # resolution to match the published K=8 protocol; pad a 1-slot wave to 2 and slice slot 0.
        self._padded_single = len(offsets) == 1
        if self._padded_single:
            offsets = [offsets[0], ROBOT_ARENA_K8_OFFSETS[0]]

        cfg = RobotArenaEnvConfig(
            data_root=self._resolve_data_root(),
            scene_name=comp_task.scene,
            robot_name="widowx",
            max_episode_steps=comp_task.max_episode_steps,
            control_freq=5,
            sim_freq=200,
            sim_substeps=1,
            rigid_constraint_timeconst=0.02,
            expose_privileged=self._expose_privileged,
            publish_depth=self._publish_depth,
            camera_width=640,
            camera_height=480,
            per_env_object_offsets=offsets,
            composition_task=task_name,
        )
        if self._env is not None and self._env.can_reuse_for(cfg):
            self._env.reconfigure(cfg)
        else:
            if self._env is not None:
                self._env.close()
            self._env = RobotArenaEnv(config=cfg)

        obs_list = self._env.reset()
        if self._padded_single:
            obs_list = obs_list[:1]
        self._last_eef_poses = [np.asarray(o.state.eef_pose, dtype=np.float32) for o in obs_list]
        self._last_target_poses = [np.asarray(o.state.eef_pose, dtype=np.float32) for o in obs_list]
        self._last_joints = [None] * len(obs_list)
        self._last_actions = [self._hold_robot_action(p) for p in self._last_target_poses]
        return obs_list

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        assert self._env is not None
        step_actions = [actions[0], actions[0]] if self._padded_single else actions
        obs_list = self._env.step(step_actions)
        return obs_list[:1] if self._padded_single else obs_list

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "main": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = RAW
        return spec

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


class _RoboLabKit:
    """Process-wide Isaac Sim SimulationApp and single worker thread (Kit's async_engine needs its own thread)."""

    app: Any = None
    thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-robolab")
    env: Any = None


class LadderRoboLabBenchmark(_LadderVectorBenchmark):
    """LADDER ``RoboLab-Long`` benchmark (Franka DROID on Isaac Sim 6 / Isaac Lab 3, 44 tasks × 8 episodes)."""

    SUITE_NAME: ClassVar[str] = "ladder_robolab"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = ROBOLAB_TASKS
    LITE_TASKS: ClassVar[tuple[str, ...]] = ROBOLAB_LITE_TASKS
    DEFAULT_MAX_STEPS: ClassVar[int] = 1500
    hard_exit: ClassVar[bool] = True
    step_dt = 1.0 / 15.0

    def __init__(
        self,
        num_envs: int = 8,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "joint_pos",
        control_mode: str | None = None,
        max_episode_steps: int | None = None,
        success_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
        expose_oracle: bool = False,
        expose_privileged: bool = False,
        publish_depth: bool = False,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            tasks=tasks,
            root=root,
            assets_dir=assets_dir,
            action_mode=action_mode,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
            success_mode=success_mode,
            terminate_on_success=terminate_on_success,
            send_state=send_state,
            send_wrist_image=send_wrist_image,
            expose_oracle=expose_oracle,
            expose_privileged=expose_privileged,
            publish_depth=publish_depth,
        )
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._env: Any = None
        self._env_cfg: Any = None
        self._current_key: tuple[str, int] | None = None
        self._converter: Any = None
        self._raw_obs: Any = None
        self._oracle: Any = None
        self._step_count = 0

    def _on_kit_thread(self, fn: Callable[..., _T], *args: Any) -> _T:
        return _RoboLabKit.thread.submit(fn, *args).result()

    def _resolve_assets_dir(self) -> str:
        if "ROBOLAB_ASSET_DIR" in os.environ and Path(os.environ["ROBOLAB_ASSET_DIR"]).is_dir():
            return os.environ["ROBOLAB_ASSET_DIR"]
        candidates = [
            self._assets_dir,
            Path(os.environ["LADDER_ROBOLAB_ASSETS"]) if "LADDER_ROBOLAB_ASSETS" in os.environ else None,
            self._root / "ladder" / "components" / "inference" / "robots" / "robolab" / "assets",
            assets_cache("ladder_robolab") / "assets",
            assets_cache("ladder_robolab"),
        ]
        for cand in candidates:
            if cand is not None and cand.is_dir():
                os.environ["ROBOLAB_ASSET_DIR"] = str(cand)
                return str(cand)
        raise FileNotFoundError(
            "RoboLab assets not found. Set LADDER_ROBOLAB_ASSETS or mount to ~/.cache/vla-eval/assets/ladder_robolab."
        )

    def _launch_app(self) -> None:
        if _RoboLabKit.app is not None:
            return
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        self._resolve_assets_dir()
        _ensure_ladder_on_path(self._root)

        import signal
        import threading

        import cv2  # noqa: F401 -- must precede isaaclab import
        from isaaclab.app import AppLauncher

        orig_signal = signal.signal
        if threading.current_thread() is not threading.main_thread():
            setattr(signal, "signal", lambda *a, **kw: None)
        try:
            launcher = AppLauncher(headless=True, enable_cameras=True)
        finally:
            setattr(signal, "signal", orig_signal)
        _RoboLabKit.app = launcher.app

        import carb.settings
        import robolab.constants
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        from ladder.components.inference.robots.robolab.composition.specs import SPECS, task_target

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)
        auto_register_droid_envs(
            task_dirs=None,
            task=[task_target(t) if t in SPECS else t for t in self._task_names],
            cameras=None,
        )

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        return self._on_kit_thread(self._reset_wave_sync, task_name, len(episode_ids))

    def _reset_wave_sync(self, task_name: str, n_envs: int) -> list[Any]:
        self._launch_app()
        from robolab.core.environments.runtime import create_env

        from ladder.components.inference.robots.robolab.composition.specs import SPECS, task_class_name
        from ladder.components.inference.robots.robolab.droid_conversions import DroidJointposConverter
        from ladder.components.inference.robots.robolab.oracle import build_oracle

        cls_name = task_class_name(task_name) if task_name in SPECS else task_name
        key = (cls_name, n_envs)
        if self._env is None or self._current_key != key:
            if _RoboLabKit.env is not None:
                _RoboLabKit.env.close()
                _RoboLabKit.env = None
            self._env, self._env_cfg = create_env(
                cls_name,
                device="cuda:0",
                num_envs=n_envs,
                instruction_type="default",
                policy="ladder_runner",
                renderer="realtime",
                rendering_mode="balanced",
            )
            _RoboLabKit.env = self._env
            self._current_key = key
            self._oracle = build_oracle(getattr(self._env.cfg, "subtasks", None), self._env_cfg.instruction)
        else:
            self._env.reset_eval_state()

        obs, _ = self._env.reset()
        obs, _ = self._env.reset()  # native double reset
        self._raw_obs = obs
        self._step_count = 0
        self._converter = DroidJointposConverter(num_envs=n_envs, image_size=self._image_size)
        obs_list = self._build_obs_list()
        self._last_eef_poses = [np.asarray(o["state"]["eef_pose"], dtype=np.float32) for o in obs_list]
        self._last_target_poses = [np.asarray(o["state"]["eef_pose"], dtype=np.float32) for o in obs_list]
        self._last_joints = [np.asarray(o["state"]["joints"], dtype=np.float32)[:7] for o in obs_list]
        self._last_actions = [
            self._hold_robot_action(self._last_eef_poses[i], o["state"]["joints"]) for i, o in enumerate(obs_list)
        ]
        return obs_list

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        return self._on_kit_thread(self._step_wave_sync, actions)

    def _step_wave_sync(self, actions: list[Any]) -> list[Any]:
        import omni.kit.app
        import omni.timeline
        import torch

        assert self._env is not None and self._raw_obs is not None
        joints = self._raw_obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()
        commands = self._converter.actions_to_jointpos(actions, joints)
        timeline = omni.timeline.get_timeline_interface()
        kit_app = omni.kit.app.get_app()
        while not timeline.is_playing():
            kit_app.update()
        obs, _, _, _, _ = self._env.step(torch.as_tensor(commands, device=self._env.device))
        self._raw_obs = obs
        self._step_count += 1
        return self._build_obs_list()

    def _build_obs_list(self) -> list[dict[str, Any]]:
        from ladder.components.inference.robots.robolab.composition.specs import (
            SPECS,
            decomp_cache_entries,
            task_class_name,
        )

        image_obs = self._raw_obs["image_obs"]
        proprio = self._raw_obs["proprio_obs"]
        results = self._env.get_env_results()
        active = set(self._env.active_env_ids)
        max_steps_reached = self._step_count >= int(self._env.max_episode_length)
        cls_name = self._current_key[0] if self._current_key else ""
        plan_entries = decomp_cache_entries()
        ordered_map = {task_class_name(k): s.ordered for k, s in SPECS.items()}
        revisit_map = {task_class_name(k): bool(s.via_legs) for k, s in SPECS.items()}

        achieved_all: list[list[bool]] | None = None
        plan_subtasks: list[str] | None = None
        if self._oracle:
            commands = [g.command for g in self._oracle]
            cached = (plan_entries.get(cls_name) or {}).get("subtasks")
            plan_subtasks = list(cached) if cached and len(cached) == len(commands) else commands
            try:
                achieved_all = [
                    [bool(g.check(self._env, env_id=i)) for g in self._oracle] for i in range(self._env.num_envs)
                ]
            except Exception:
                achieved_all = None

        out: list[dict[str, Any]] = []
        for i in range(self._env.num_envs):
            success = bool(results[i]["success"])
            truncated = max_steps_reached or i not in active
            oracle_block = (
                {
                    "subtasks": plan_subtasks,
                    "achieved": achieved_all[i],
                    "ordered": ordered_map.get(cls_name, False),
                    "revisits": revisit_map.get(cls_name, False),
                }
                if achieved_all is not None
                else {"subtasks": [self._env_cfg.instruction], "achieved": [success]}
            )
            metadata = {
                "env_name": cls_name,
                "robot_name": "franka",
                "instruction": self._env_cfg.instruction,
                "done": success,
                "truncated": truncated,
                "oracle": oracle_block,
            }
            out.append(
                self._converter.observation_dict(
                    env_i=i,
                    main_image=image_obs["over_shoulder_left_camera"][i].detach().cpu().numpy(),
                    wrist_image=image_obs["wrist_cam"][i].detach().cpu().numpy(),
                    joints=proprio["arm_joint_pos"][i].detach().cpu().numpy(),
                    gripper_position=proprio["gripper_pos"][i].detach().cpu().numpy(),
                    metadata=metadata,
                )
            )
        return out

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "main": IMAGE_RGB, "wrist": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = RAW
        return spec

    def cleanup(self) -> None:
        self._env = None
