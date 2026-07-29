"""RoboCasa365 multi-task benchmark implementation.

RoboCasa365 is the successor to RoboCasa: policies pretrain on 300 kitchen
tasks (65 atomic + 235 composite) across 2500+ procedurally-generated
scenes, and are evaluated on a 50-task target split.  This adapter drives
robocasa's own Panda-Omron Gymnasium wrapper, so task membership, the
pretraining-kitchen split, per-task horizons, the observation schema and
the success predicate all come from upstream.

The official multi-task protocol evaluates the 50 target tasks
(``atomic_seen`` + ``composite_seen`` + ``composite_unseen``) in pretrain
scenes for 50 episodes each, running every episode to its registry horizon
while sampling success at the end of each action chunk.

Actions are the wrapper's full 12-D mobile-manipulation vector, flattened in
the order of robocasa's own ``env_utils.convert_action``; nothing is dropped
or padded.

The original pre-v1 benchmark lives in :mod:`vla_eval.benchmarks.robocasa`
and runs on its own image; the two upstream releases are not API-compatible.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import (
    BASE_MOTION,
    CONTROL_MODE_01,
    GRIPPER_CLOSE_01,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

# The three task sets whose union is robocasa's ``target50`` evaluation split.
TARGET_TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")

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
# Flat action layout of robocasa's ``env_utils.convert_action``.
ACTION_COMPONENTS = (
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
)
ACTION_DIM = sum(width for _, width in ACTION_COMPONENTS)


def decode_action(action: Action) -> dict[str, np.ndarray]:
    """Split a flat 12-D action into the wrapper's named action dict."""
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


def _task_registry() -> Mapping[str, Sequence[str]]:
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    return TASK_SET_REGISTRY


def _task_horizon(task_name: str) -> int:
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    return int(get_task_horizon(task_name))


class RoboCasa365Benchmark(StepBenchmark):
    """RoboCasa365 multi-task benchmark on the official target50 split.

    Args:
        tasks: Task names to evaluate, which must be in the target50 registry.
            Defaults to all 50.
        camera_size: Camera resolution (square).  256 is the protocol value.
        max_steps: Override the per-task registry horizon.  Leave unset for
            the official protocol; use it to cut episodes short in smoke runs.
        split: Kitchen split — ``"pretrain"`` (the multi-task protocol) or
            ``"target"``.
        seed: Base seed; episode ``i`` of each task runs at ``seed + i``.
            ``None`` leaves the environment unseeded.
        success_check_interval: Sample success every N steps, matching the
            official evaluator's one success check per action chunk.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int | None = None,
        split: str = "pretrain",
        seed: int | None = None,
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
        self._max_steps = max_steps
        self._split = split
        self._seed = seed
        self._success_check_interval = success_check_interval
        self._resolved_tasks: list[Task] | None = None
        self._env: Any = None
        self._current_task: str | None = None
        self._horizon = 0
        self._steps = 0
        self._success = False

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                # Cleanup must not mask an episode or runner failure.
                pass
            finally:
                self._env = None

    def get_tasks(self) -> list[Task]:
        if self._resolved_tasks is None:
            registry = _task_registry()
            suite_of = {task: suite for suite in TARGET_TASK_SETS for task in registry[suite]}
            names = self._explicit_tasks if self._explicit_tasks is not None else list(suite_of)
            unknown = sorted(set(names) - set(suite_of))
            if unknown:
                raise ValueError(f"tasks are not in the official target50 registry: {unknown}")
            self._resolved_tasks = [{"name": name, "suite": suite_of[name]} for name in names]
        return [dict(task) for task in self._resolved_tasks]

    def _make_env(self, task_name: str) -> Any:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import gymnasium as gym
        import robocasa.wrappers.gym_wrapper  # noqa: F401  # registers the robocasa/* Gym environments

        return gym.make(
            f"robocasa/{task_name}",
            split=self._split,
            enable_render=True,
            camera_widths=self._camera_size,
            camera_heights=self._camera_size,
            disable_env_checker=True,
        )

    def reset(self, task: Task) -> Any:
        task_name = task["name"]
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                self._env.close()
            self._env = self._make_env(task_name)
            self._current_task = task_name

        # The wrapper reseeds the simulator's rng from ``seed`` before sampling
        # the scene, so episodes are reproducible without rebuilding the env.
        episode_seed = None if self._seed is None else self._seed + int(task.get("episode_idx", 0))
        obs, _ = self._env.reset(seed=episode_seed)
        self._steps = 0
        self._success = False
        self._horizon = self._max_steps or _task_horizon(task_name)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        obs, _, _, _, info = self._env.step(decode_action(action))
        self._steps += 1
        done = self._steps >= self._horizon
        # The official evaluator reads success once per action chunk and
        # latches it, running the episode out to the registry horizon.
        if self._steps % self._success_check_interval == 0 or done:
            self._success |= bool(info.get("success", False))
        info = {**info, "success": self._success}
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(self._success), done=done, success=self._success)
        return StepResult(obs=obs, reward=float(self._success), done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, Mapping):
            return None
        frame = raw_obs.get(VIDEO_KEYS[0])
        return None if frame is None else np.ascontiguousarray(frame)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        missing = [key for key in (*VIDEO_KEYS, *STATE_KEYS) if key not in raw_obs]
        if missing:
            raise KeyError(f"RoboCasa365 observation is missing official fields: {missing}")
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
        # Episodes stop at their own registry horizon; the runner-wide cap has
        # to clear the longest of them.
        horizons = [self._max_steps] if self._max_steps else [_task_horizon(t["name"]) for t in self.get_tasks()]
        return {
            "max_steps": max(horizons),
            "split": self._split,
            "seed": self._seed,
            "success_check_interval": self._success_check_interval,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_01,
            "base_motion": BASE_MOTION,
            "control_mode": CONTROL_MODE_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def render(self) -> np.ndarray | None:
        if self._env is None:
            return None
        try:
            return self._env.render()
        except Exception:
            return None
