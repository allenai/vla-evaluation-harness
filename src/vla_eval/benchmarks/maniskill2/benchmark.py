"""ManiSkill2 benchmark implementation.

Evaluates 5 ManiSkill2 tasks:
PickCube-v0, StackCube-v0, PickSingleYCB-v0, PickSingleEGAD-v0, PickClutterYCB-v0.

Key details:
- gymnasium API with obs_mode="rgbd", control_mode="pd_ee_delta_pose"
- Gripper state tracking: self.gripper_state = -action[-1]
- Gripper binarization: < 0.5 → 1 (open), >= 0.5 → -1 (close)
- Done condition: terminated or truncated
- Success: info.get("success", False)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_CLOSE_NEG, IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

# Prevent display issues in headless environments
os.environ.setdefault("DISPLAY", "")

# Task name → goal description
TASK_GOALS: dict[str, str] = {
    "PickCube-v0": "pick up a cube and move it to the green point",
    "StackCube-v0": "pick up a red cube and place it on a green cube",
    "PickSingleYCB-v0": "pick up the {} and move it to the green point",
    "PickSingleEGAD-v0": "pick up the {} and move it to the green point",
    "PickClutterYCB-v0": "pick up the {} and move it to the green point",
}

DEFAULT_TASKS = list(TASK_GOALS.keys())


class ManiSkill2Benchmark(StepBenchmark):
    """ManiSkill2 manipulation benchmark (SAPIEN physics).

    Non-obvious behaviors:
        - **Goal site visibility**: ManiSkill2 hides the goal sphere by
          default.  This benchmark explicitly makes it visible before each
          step, matching the training setup where models see the green target.
        - **Gripper binarization**: Threshold is 0.5 (not 0.0).
          ``raw[-1] < 0.5 → 1.0 (close), ≥ 0.5 → -1.0 (open)``.
        - **Camera names**: ``enabled_cameras`` values must exactly match
          camera names defined in the ManiSkill2 environment.

    Args:
        tasks: List of ManiSkill2 task IDs (e.g. ``["PickCube-v0"]``).
        max_episode_steps: Max steps per episode (default 400).
        image_size: Camera resolution as ``[width, height]`` (default [256, 256]).
        enabled_cameras: Camera names to include (default ``["base_camera"]``).
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        episodes_per_task: int = 50,
        max_episode_steps: int = 400,
        image_size: list[int] | tuple[int, int] = (256, 256),
        enabled_cameras: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.tasks = tasks or DEFAULT_TASKS
        self.episodes_per_task = episodes_per_task
        self.max_episode_steps = max_episode_steps
        self.image_size = tuple(image_size)
        self.enabled_cameras = enabled_cameras or ["base_camera"]

        self._env = None
        self._current_task: str | None = None
        self._goal: str = ""
        self.gripper_state: float = -1.0

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def get_tasks(self) -> list[Task]:
        return [{"name": t, "env_name": t} for t in self.tasks]

    def reset(self, task: Task) -> Any:
        import gymnasium as gym
        import mani_skill2.envs  # noqa: F401 — register envs

        env_name = task["env_name"]

        # Recreate env when task changes
        if self._env is None or self._current_task != env_name:
            if self._env is not None:
                self._env.close()
            self._env = gym.make(
                env_name,
                obs_mode="rgbd",
                control_mode="pd_ee_delta_pose",
                render_mode="cameras",
                camera_cfgs=dict(width=self.image_size[0], height=self.image_size[1]),
                max_episode_steps=self.max_episode_steps,
            )
            self._current_task = env_name

        obs, info = self._env.reset()
        self.gripper_state = -1.0

        # Resolve goal description
        goal_template = TASK_GOALS.get(env_name, "complete the task")
        if "{}" in goal_template:
            obj_name = self._get_obj_name()
            self._goal = goal_template.format(obj_name)
        else:
            self._goal = goal_template

        return obs

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        assert len(raw) == 7, f"Action dimension mismatch: got {len(raw)}, expected 7"

        # Gripper binarization: < 0.5 → open, >= 0.5 → close
        gripper = 1.0 if raw[-1] < 0.5 else -1.0
        env_action = np.array(raw[:6] + [gripper], dtype=np.float32)

        # Render goal site so the green target sphere is visible in camera
        # observations. ManiSkill2 hides it by default (_hidden_objects).
        # render_goal=True before env.step() so the green target is visible.
        assert self._env is not None
        self._render_goal_site(self._env)

        obs, reward, terminated, truncated, info = self._env.step(env_action)

        # Track gripper state for next observation (sign inversion)
        self.gripper_state = -env_action[-1]

        done = terminated or truncated
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # Extract camera images
        images: dict[str, np.ndarray] = {}
        for cam in self.enabled_cameras:
            images[cam] = raw_obs["image"][cam]["rgb"]

        return {
            "images": images,
            "task_description": self._goal,
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": bool(step_result.info.get("success", False))}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self.max_episode_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "base_camera": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def _get_obj_name(self) -> str:
        """Extract object name from the environment for goal description."""
        try:
            assert self._env is not None
            obj = self._env.unwrapped.obj
            return " ".join(obj.name.split("_")[1:])
        except (AttributeError, IndexError):
            return "object"

    @staticmethod
    def _render_goal_site(env: Any) -> None:
        """Make the goal_site sphere visible in camera observations.

        ManiSkill2 adds goal_site to _hidden_objects, which calls
        hide_visual() during env reconfiguration. The model was trained
        with the green goal sphere visible, so we must restore visibility.
        """
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(unwrapped, "goal_site"):
            for v in unwrapped.goal_site.get_visual_bodies():
                v.set_visibility(1.0)
