"""RoboCasa and RoboCasa365 benchmark adapter.

The legacy protocol preserves the original arbitrary-task, configurable-camera,
7-D RoboCasa adapter.  The ``rc365`` protocol uses RoboCasa's registered
Gymnasium environment, which is the behavioral reference for the official
Panda-Omron observation and action contract.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import (
    GRIPPER_RAW,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    ROTATION_EULER,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_TASKS = [
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToSink",
    "OpenSingleDoor",
    "CloseDoubleDoor",
    "TurnOnSinkFaucet",
    "PreheatOven",
]
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
    """RoboCasa adapter with legacy and official RoboCasa365 protocols.

    ``protocol="legacy"`` preserves the original adapter API and behavior.
    ``protocol="rc365"`` resolves the official target50 tasks and per-task
    horizons from RoboCasa's registry and uses the 12-D Panda-Omron contract.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "PandaOmron",
        camera_names: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int | None = None,
        split: str = "pretrain",
        seed: int | None = None,
        enable_render: bool = True,
        success_check_interval: int = 16,
        protocol: str = "legacy",
    ) -> None:
        super().__init__()
        if protocol not in {"legacy", "rc365"}:
            raise ValueError("protocol must be 'legacy' or 'rc365'")
        if split not in {"pretrain", "target"}:
            raise ValueError("split must be 'pretrain' or 'target'")
        if camera_size <= 0:
            raise ValueError("camera_size must be positive")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if success_check_interval <= 0:
            raise ValueError("success_check_interval must be positive")

        self._protocol = protocol
        self._explicit_tasks = list(tasks) if tasks is not None else None
        self._robot = robot
        self._camera_names = camera_names or [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ]
        self._camera_size = camera_size
        self._max_steps_override = max_steps
        self._legacy_max_steps = 500 if max_steps is None else max_steps
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
        self._lang = ""

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
        if self._protocol == "legacy":
            return [{"name": task} for task in (self._explicit_tasks or DEFAULT_TASKS)]

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
        if self._protocol == "legacy":
            from robocasa.utils.env_utils import create_env

            return create_env(
                env_name=task_name,
                robots=self._robot,
                camera_names=self._camera_names,
                camera_widths=self._camera_size,
                camera_heights=self._camera_size,
                render_onscreen=False,
                split=self._split,
                seed=self._seed,
            )

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
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                self._env.close()
            self._env = self._make_env(task_name)
            self._current_task = task_name

        if self._protocol == "legacy":
            obs = self._env.reset()
            self._lang = self._env.get_ep_meta().get("lang", task_name)
            self._recorder.record_video(self._extract_frame(obs))
            return obs

        episode_idx = int(task.get("episode_idx", 0))
        episode_seed = None if self._seed is None else self._seed + episode_idx
        obs, _ = self._env.reset(seed=episode_seed)
        self._steps = 0
        self._episode_success = False
        self._current_horizon = self._max_steps_override or _task_horizon(task_name)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        if self._protocol == "legacy":
            return self._step_legacy(action)

        named_action = _decode_panda_omron_action(action)
        obs, _, _, _, info = self._env.step(named_action)
        self._steps += 1
        done = self._steps >= self._current_horizon
        # Match the official MultiStepWrapper: sample success at each policy
        # chunk endpoint (including the final partial chunk) and always run to
        # the registry horizon. RoboCasa environments use ignore_done=True, so
        # inner termination signals do not define the official episode length.
        at_chunk_endpoint = self._steps % self._success_check_interval == 0 or done
        if at_chunk_endpoint:
            self._episode_success |= bool(info.get("success", False) or self._env.unwrapped.env._check_success())
        info = {**info, "success": self._episode_success}
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(self._episode_success), done=done, success=self._episode_success)
        return StepResult(obs=obs, reward=float(self._episode_success), done=done, info=info)

    def _step_legacy(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(7)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        assert raw_action.shape[-1] == 7, f"Action dimension mismatch: got {raw_action.shape[-1]}, expected 7"

        act_dim = self._env.action_spec[0].shape[0]
        if raw_action.shape[0] < act_dim:
            raw_action = np.concatenate([raw_action, np.zeros(act_dim - raw_action.shape[0])])
        elif raw_action.shape[0] > act_dim:
            raw_action = raw_action[:act_dim]

        obs, _, done, info = self._env.step(raw_action)
        success = bool(self._env._check_success())
        info["success"] = success
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(success), done=bool(done), success=success)
        return StepResult(obs=obs, reward=float(success), done=done, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, Mapping):
            return None
        if self._protocol == "legacy":
            for camera_name in self._camera_names:
                frame = raw_obs.get(f"{camera_name}_image")
                if frame is not None:
                    return np.ascontiguousarray(frame[::-1])
            return None
        frame = raw_obs.get(VIDEO_KEYS[0])
        return None if frame is None else np.ascontiguousarray(frame)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        if self._protocol == "legacy":
            images = {}
            for camera_name in self._camera_names:
                frame = raw_obs.get(f"{camera_name}_image")
                if frame is not None:
                    images[camera_name] = np.ascontiguousarray(frame[::-1])
            return {"images": images, "task_description": self._lang}

        missing = [key for key in (*VIDEO_KEYS, *STATE_KEYS) if key not in raw_obs]
        if missing:
            raise KeyError(f"RoboCasa Gym observation is missing official fields: {missing}")
        return {
            "images": {key: np.ascontiguousarray(raw_obs[key]) for key in VIDEO_KEYS},
            "state": {key: np.asarray(raw_obs[key]) for key in STATE_KEYS},
            "task_description": str(raw_obs.get("annotation.human.task_description", task["name"])),
        }

    def check_done(self, step_result: StepResult) -> bool:
        if self._protocol == "legacy":
            return step_result.done or step_result.info.get("success", False)
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        if self._protocol == "legacy":
            return {"success": step_result.info.get("success", False)}
        return {"success": bool(step_result.info.get("success", False))}

    def get_metadata(self) -> dict[str, Any]:
        if self._protocol == "legacy":
            return {"max_steps": self._legacy_max_steps}

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
        if self._protocol == "legacy":
            return {
                "position": POSITION_DELTA,
                "rotation": ROTATION_EULER,
                "gripper": GRIPPER_RAW,
            }
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_BINARY_CLOSE_01,
            "base_motion": BASE_MOTION,
            "control_mode": CONTROL_MODE_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        if self._protocol == "legacy":
            return {"robot0_agentview_left": IMAGE_RGB, "language": LANGUAGE}
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def render(self) -> np.ndarray | None:
        if self._env is None:
            return None
        try:
            return self._env.render()
        except Exception:
            return None
