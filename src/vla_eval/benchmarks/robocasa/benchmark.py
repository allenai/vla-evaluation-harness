"""RoboCasa benchmark implementation.

RoboCasa is a large-scale simulation framework for kitchen manipulation
tasks built on top of robosuite v2 and MuJoCo.  It provides 365 tasks
(atomic + composite) across 2500+ procedurally-generated kitchen scenes.

Actions are 7-D by default when using the ``PandaOmron`` robot with a
standard ``OSC_POSE`` composite controller: ``[dx, dy, dz, drx, dry, drz,
gripper]``.

Observations expose RGB images from configurable cameras (default:
``robot0_agentview_left`` and ``robot0_eye_in_hand``) plus a natural
language task description obtained via ``env.get_ep_meta()["lang"]``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")

# Subset of atomic tasks suitable for quick evaluation.
DEFAULT_TASKS = [
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToSink",
    "OpenSingleDoor",
    "CloseDoubleDoor",
    "TurnOnSinkFaucet",
    "PreheatOven",
]


class RoboCasaBenchmark(StepBenchmark):
    """RoboCasa kitchen manipulation benchmark.

    Args:
        tasks: List of RoboCasa environment names to evaluate.
            Defaults to a small atomic-task subset.
        robot: Robot model name (default ``"PandaOmron"``).
        camera_names: Camera names for observations.
        camera_size: Camera resolution (square, default 256).
        max_steps: Maximum steps per episode (default 500).
        split: Dataset split — ``"pretrain"`` or ``"target"``.
        seed: Random seed for environment creation.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "PandaOmron",
        camera_names: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int = 500,
        split: str = "pretrain",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._robot = robot
        self._camera_names = camera_names or [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ]
        self._camera_size = camera_size
        self._max_steps = max_steps
        self._split = split
        self._seed = seed
        self._env: Any = None
        self._current_task: str | None = None
        self._lang: str = ""

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def get_tasks(self) -> list[Task]:
        return [{"name": t} for t in self._task_names]

    def reset(self, task: Task) -> Any:
        from robocasa.utils.env_utils import create_env

        task_name = task["name"]

        # Reuse env for same task across episodes
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                self._env.close()
            self._env = create_env(
                env_name=task_name,
                robots=self._robot,
                camera_names=self._camera_names,
                camera_widths=self._camera_size,
                camera_heights=self._camera_size,
                render_onscreen=False,
                split=self._split,
                seed=self._seed,
            )
            self._current_task = task_name

        obs = self._env.reset()
        self._lang = self._env.get_ep_meta().get("lang", task_name)
        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(7)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        assert raw_action.shape[-1] == 7, f"Action dimension mismatch: got {raw_action.shape[-1]}, expected 7"

        # Pad or truncate to match env action dimension
        act_dim = self._env.action_spec[0].shape[0]
        if raw_action.shape[0] < act_dim:
            raw_action = np.concatenate([raw_action, np.zeros(act_dim - raw_action.shape[0])])
        elif raw_action.shape[0] > act_dim:
            raw_action = raw_action[:act_dim]

        obs, reward, done, info = self._env.step(raw_action)
        success = bool(self._env._check_success())
        info["success"] = success
        return StepResult(obs=obs, reward=float(success), done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images: dict[str, Any] = {}
        for cam in self._camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                # RoboCasa images are upside-down — flip vertically
                images[cam] = np.ascontiguousarray(raw_obs[key][::-1])
        return {
            "images": images,
            "task_description": self._lang,
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done or step_result.info.get("success", False)

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "robot0_agentview_left": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def render(self) -> np.ndarray | None:
        try:
            return self._env.render()
        except Exception:
            return None
