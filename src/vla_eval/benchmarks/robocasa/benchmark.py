"""RoboCasa365 benchmark adapter.

The adapter intentionally uses RoboCasa's registered Gymnasium environment
instead of rebuilding the Panda-Omron observation and action contract.  The
upstream wrapper is the behavioral reference used by RoboCasa365's GR00T
evaluation code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import (
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")

OFFICIAL_TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")
VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)
ACTION_COMPONENTS = (
    # Official LeRobot modality.json flat-action order.
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
)
ACTION_DIM = sum(width for _, width in ACTION_COMPONENTS)
BASE_MOTION = DimSpec("base_motion", 4, "panda_omron_base_velocity3_torso", (-1, 1))
CONTROL_MODE_01 = DimSpec("control_mode", 1, "panda_omron_control_mode_01", (0, 1))
GRIPPER_BINARY_CLOSE_01 = DimSpec("gripper", 1, "binary_close_01", (0, 1))


def _task_registry() -> Mapping[str, Sequence[str]]:
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    return TASK_SET_REGISTRY


def _task_horizon(task_name: str) -> int:
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    return int(get_task_horizon(task_name))


def _decode_panda_omron_action(action: Action) -> dict[str, np.ndarray]:
    """Decode the flat vla-eval wire action into RoboCasa's named action dict."""
    raw = action.get("actions", action.get("action"))
    if raw is None:
        raise ValueError("RoboCasa365 requires an 'actions' vector")
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (ACTION_DIM,):
        raise ValueError(f"RoboCasa365 expected a {ACTION_DIM}-D action, got {raw.shape}")

    named = {}
    offset = 0
    for key, width in ACTION_COMPONENTS:
        named[key] = raw[offset : offset + width]
        offset += width
    return named


class RoboCasaBenchmark(StepBenchmark):
    """Official-protocol RoboCasa365 environment adapter.

    ``tasks=None`` resolves the 50 official multi-task evaluation tasks from
    RoboCasa's registry.  Per-task horizons also come from that registry.
    ``max_steps`` is only an explicit debugging override.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int | None = None,
        split: str = "pretrain",
        seed: int | None = 0,
        enable_render: bool = True,
        success_check_interval: int = 16,
    ) -> None:
        super().__init__()
        if split not in {"pretrain", "target"}:
            raise ValueError("split must be 'pretrain' or 'target'")
        if camera_size <= 0:
            raise ValueError("camera_size must be positive")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if success_check_interval <= 0:
            raise ValueError("success_check_interval must be positive")

        self._explicit_tasks = list(tasks) if tasks is not None else None
        self._camera_size = camera_size
        self._max_steps_override = max_steps
        self._split = split
        self._seed = seed
        self._enable_render = enable_render
        self._success_check_interval = success_check_interval
        self._resolved_tasks: list[Task] | None = None
        self._env: Any = None
        self._current_task: str | None = None
        self._current_horizon = 0
        self._steps = 0
        self._episode_success = False

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                # Cleanup must not mask an episode or runner failure.
                pass
            finally:
                self._env = None

    def _resolve_tasks(self) -> list[Task]:
        registry = _task_registry()
        task_to_suite = {task: suite for suite in OFFICIAL_TASK_SETS for task in registry[suite]}
        if self._explicit_tasks is not None:
            unknown = sorted(set(self._explicit_tasks) - set(task_to_suite))
            if unknown:
                raise ValueError(f"tasks are not in the official target50 registry: {unknown}")
            return [{"name": task, "suite": task_to_suite[task]} for task in self._explicit_tasks]

        return [{"name": task, "suite": suite} for suite in OFFICIAL_TASK_SETS for task in registry[suite]]

    def get_tasks(self) -> list[Task]:
        if self._resolved_tasks is None:
            self._resolved_tasks = self._resolve_tasks()
        return [dict(task) for task in self._resolved_tasks]

    def _make_env(self, task_name: str) -> Any:
        import gymnasium as gym
        import robocasa  # noqa: F401  # registers robocasa/* Gym environments

        return gym.make(
            f"robocasa/{task_name}",
            split=self._split,
            enable_render=self._enable_render,
            camera_widths=self._camera_size,
            camera_heights=self._camera_size,
        )

    def reset(self, task: Task) -> Any:
        task_name = task["name"]
        episode_idx = int(task.get("episode_idx", 0))
        episode_seed = None if self._seed is None else self._seed + episode_idx
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                self._env.close()
            self._env = self._make_env(task_name)
            self._current_task = task_name

        obs, _ = self._env.reset(seed=episode_seed)
        self._steps = 0
        self._episode_success = False
        self._current_horizon = self._max_steps_override or _task_horizon(task_name)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        named_action = _decode_panda_omron_action(action)
        obs, _, terminated, truncated, info = self._env.step(named_action)
        self._steps += 1
        time_limit_reached = self._steps >= self._current_horizon
        done = bool(terminated or truncated or time_limit_reached)
        # Upstream MultiStepWrapper exposes success once per action chunk and still runs to the task horizon.
        at_chunk_boundary = self._steps % self._success_check_interval == 0
        if at_chunk_boundary or done:
            self._episode_success |= bool(info.get("success", False) or self._env.unwrapped.env._check_success())
        info = {**info, "success": self._episode_success}
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(self._episode_success), done=done, success=self._episode_success)
        return StepResult(obs=obs, reward=float(self._episode_success), done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, Mapping):
            return None
        frame = raw_obs.get(VIDEO_KEYS[0])
        return None if frame is None else np.ascontiguousarray(frame)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        missing = [key for key in (*VIDEO_KEYS, *STATE_KEYS) if key not in raw_obs]
        if missing:
            raise KeyError(f"RoboCasa Gym observation is missing official fields: {missing}")
        return {
            "images": {key: np.ascontiguousarray(raw_obs[key]) for key in VIDEO_KEYS},
            "state": {key: np.asarray(raw_obs[key]) for key in STATE_KEYS},
            "task_description": str(raw_obs.get("annotation.human.task_description", task["name"])),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": bool(step_result.info.get("success", False))}

    def get_metadata(self) -> dict[str, Any]:
        tasks = self.get_tasks()
        max_steps = self._max_steps_override or max(_task_horizon(task["name"]) for task in tasks)
        return {
            "max_steps": max_steps,
            "environment_split": self._split,
            "environment_seed": self._seed,
            "success_check_interval": self._success_check_interval,
            "task_horizon_source": "robocasa.utils.dataset_registry_utils.get_task_horizon",
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_BINARY_CLOSE_01,
            "base_motion": BASE_MOTION,
            "control_mode": CONTROL_MODE_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def render(self) -> np.ndarray | None:
        if self._env is None:
            return None
        return self._env.render()
