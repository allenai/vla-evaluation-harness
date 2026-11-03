"""Shared base class and kinematics helpers for the LADDER benchmark adapters."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, ClassVar

import numpy as np

from vla_eval import columns
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.rotation import axisangle_to_matrix, matrix_to_quat, quat_to_axisangle
from vla_eval.types import Batch, EpisodeResult, Task

DEFAULT_LADDER_ROOT = "/app/ladder"
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
    DEFAULT_MAX_STEPS: ClassVar[int] = 1000

    def __init__(
        self,
        num_envs: int = 8,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "delta_ee",
        control_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
    ) -> None:
        super().__init__(num_envs)
        if isinstance(tasks, str):
            resolved_tasks = [tasks]
        elif tasks is None:
            resolved_tasks = list(self.DEFAULT_TASKS)
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
        self._terminate_on_success = bool(terminate_on_success)
        self._send_state = bool(send_state)
        self._send_wrist_image = bool(send_wrist_image)

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
