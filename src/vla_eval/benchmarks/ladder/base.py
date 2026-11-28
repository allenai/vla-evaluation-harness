"""Shared wave execution and scoring for the LADDER benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from vla_eval import columns
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.rotation import axisangle_to_matrix, euler_xyz_to_matrix, matrix_to_quat, quat_to_axisangle
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Batch, EpisodeResult, Task

# The pinned LADDER checkout; the Dockerfiles put it here and on PYTHONPATH.
LADDER_ROOT = Path(os.environ.get("LADDER_ROOT", "/app/ladder"))
# LADDER's CANONICAL_TO_WIDOWX_ROTATION (robots/simpler/proprio.py): canonical EE frame -> WidowX controller frame.
_CANONICAL_TO_WIDOWX = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


@dataclass
class EnvStep:
    """One environment's observation and score after a reset or step."""

    obs: dict[str, Any]
    done: bool  # every subgoal satisfied
    truncated: bool
    num_goals: int
    num_complete: int


class LadderBenchmark(VectorStepBenchmark):
    """Runs one task per wave of ``num_envs`` episodes and scores success and subgoal progress.

    Subclasses provide the task list file and the simulator: ``_reset_wave``, ``_step_wave`` and ``_decode_action``.
    """

    partial_reset: ClassVar[bool] = False
    task_exclusive: ClassVar[bool] = True
    _ALL_RECORD_FIELDS = frozenset(
        {"reward", "done", "truncated", "success", "subgoal_progress", "num_subgoals_complete"}
    )

    SUITE_NAME: ClassVar[str]
    TASK_FILE: ClassVar[str]  # under scripts/tasks/ in the LADDER checkout
    MAX_STEPS: ClassVar[int]

    def __init__(self, num_envs: int = 8, tasks: list[str] | str | None = None, send_state: bool = True) -> None:
        super().__init__(num_envs)
        suite_tasks = (LADDER_ROOT / "scripts" / "tasks" / self.TASK_FILE).read_text().split()
        self._task_names = suite_tasks if tasks is None else [tasks] if isinstance(tasks, str) else list(tasks)
        unknown = sorted(set(self._task_names) - set(suite_tasks))
        if unknown:
            raise ValueError(f"{type(self).__name__}: unknown task(s) {unknown}")
        self._send_state = send_state
        self._task_name = ""
        self._env_index: dict[int, int] = {}
        self._recorders: dict[int, EpisodeRecorder] = {}
        self._results: dict[int, EpisodeResult] = {}
        self._actions: list[Any] = []  # last action per env; envs whose episode ended keep repeating it

    def get_tasks(self) -> list[Task]:
        return [{"name": name, "suite": self.SUITE_NAME} for name in self._task_names]

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "subgoal_progress": "mean", "num_subgoals_complete": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {"suite": self.SUITE_NAME, "max_steps": self.MAX_STEPS, "num_envs": self.num_envs}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = RAW
        return spec

    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch:
        names = {str(t["name"]) for t in tasks}
        if len(names) != 1:
            raise RuntimeError(f"{type(self).__name__}: one wave must run one task, got {sorted(names)}")
        self._task_name = names.pop()
        self._env_index = {slot: i for i, slot in enumerate(slots)}
        self._recorders = dict(zip(slots, recorders))
        steps = self._reset_wave(self._task_name, [int(t.get("episode_idx", i)) for i, t in enumerate(tasks)])
        return columns.stack([self._record(slot, steps[i], first=True) for i, slot in enumerate(slots)])

    def step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]:
        for slot, row in zip(slots, columns.split(actions, len(slots))):
            i = self._env_index[slot]
            self._actions[i] = self._decode_action(i, np.asarray(row["actions"], dtype=np.float32).reshape(-1))
        steps = self._step_wave(self._actions)
        obs = [self._record(slot, steps[self._env_index[slot]], first=False) for slot in slots]
        done = np.array([steps[self._env_index[s]].done or steps[self._env_index[s]].truncated for s in slots])
        return columns.stack(obs), done

    def result(self, slot: int) -> EpisodeResult:
        return self._results[slot]

    def _record(self, slot: int, step: EnvStep, first: bool) -> dict[str, Any]:
        completed = step.num_goals if step.done else step.num_complete
        result: EpisodeResult = {
            "success": step.done,
            "subgoal_progress": round(completed / step.num_goals, 6) if step.num_goals else float(step.done),
            "num_subgoals_complete": completed,
        }
        self._results[slot] = result
        recorder = self._recorders[slot]
        recorder.record_video(step.obs["images"]["main"])
        if not first:
            recorder.record_step(reward=float(step.done), done=step.done, truncated=step.truncated, **result)
        return step.obs

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[EnvStep]:
        """Start ``task_name`` in every env, set ``self._actions`` to a hold action per env."""
        raise NotImplementedError

    def _step_wave(self, actions: list[Any]) -> list[EnvStep]:
        raise NotImplementedError

    def _decode_action(self, env_i: int, raw: np.ndarray) -> Any:
        raise NotImplementedError


class WidowXBenchmark(LadderBenchmark):
    """WidowX suites (SimplerEnv-Long, RobotArena-Long): one camera, 7-D delta end-effector actions.

    An action is ``[dx, dy, dz, axis-angle (3), gripper]`` with gripper 1 = open, applied to the env's
    commanded target pose (``eef_pose_from_target``) in the robot base frame, as LADDER's own runner does.
    A ``control_mode`` with ``target_base_pose`` (X-VLA's SimplerEnv profile) instead takes absolute
    ``[x, y, z, intrinsic XYZ Euler (3), gripper]`` in the WidowX controller frame, as ManiSkill's base-pose controller.
    """

    MAX_STEPS: ClassVar[int] = 1000  # longest task budget: 100 steps per goal x 10 goals

    def __init__(self, num_envs: int = 8, control_mode: str | None = None, **kwargs: Any) -> None:
        super().__init__(num_envs=num_envs, **kwargs)
        self._absolute = control_mode is not None and "target_base_pose" in control_mode
        self._targets: list[np.ndarray] = []

    def _to_steps(self, robot_obs: list[Any]) -> list[EnvStep]:
        """``ladder`` ``RobotObservation``s -> :class:`EnvStep`s."""
        self._targets = [np.asarray(o.state.eef_pose_from_target, dtype=np.float32) for o in robot_obs]
        steps = []
        for o in robot_obs:
            meta = o.metadata or {}
            achieved = (meta.get("oracle") or {}).get("achieved", [])
            obs: dict[str, Any] = {
                "images": {"main": np.asarray(o.images["main"], dtype=np.uint8)},
                "task_description": str(meta.get("instruction", "")),
                "task_name": self._task_name,
            }
            if self._send_state:
                pose = np.asarray(o.state.eef_pose, dtype=np.float32)
                gripper = np.asarray(o.state.gripper_state, dtype=np.float32).reshape(-1)[:1]
                rotvec = quat_to_axisangle(matrix_to_quat(pose[:3, :3]))
                obs["states"] = np.concatenate([pose[:3, 3], rotvec, gripper]).astype(np.float32)
            steps.append(
                EnvStep(
                    obs=obs,
                    done=bool(meta.get("done", False)),
                    truncated=bool(meta.get("truncated", False)),
                    num_goals=len(achieved),
                    num_complete=int(meta.get("num_subgoals_complete", sum(map(bool, achieved)))),
                )
            )
        return steps

    def _hold_actions(self) -> list[Any]:
        from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction

        return [
            ManipulatorAction(
                arm_action=EndEffectorPoseAction(translation=t[:3, 3].copy(), rotation=t[:3, :3].copy()),
                gripper_action=1.0,
            )
            for t in self._targets
        ]

    def _decode_action(self, env_i: int, raw: np.ndarray) -> Any:
        from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction

        if self._absolute:  # intrinsic XYZ = euler_xyz_to_matrix(-e).T; the env maps it back to the same Euler
            rotation = euler_xyz_to_matrix(-raw[3:6].astype(np.float64)).T @ _CANONICAL_TO_WIDOWX.T
            return ManipulatorAction(
                arm_action=EndEffectorPoseAction(translation=raw[:3].copy(), rotation=rotation.astype(np.float32)),
                gripper_action=float(np.clip(raw[6], 0.0, 1.0)),
            )
        target = self._targets[env_i]
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=(target[:3, 3] + raw[:3]).astype(np.float32),
                rotation=(axisangle_to_matrix(raw[3:6].astype(np.float64)) @ target[:3, :3]).astype(np.float32),
            ),
            gripper_action=float(np.clip(raw[6], 0.0, 1.0)),
        )
