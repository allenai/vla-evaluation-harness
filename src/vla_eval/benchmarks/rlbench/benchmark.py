"""RLBench benchmark implementation.

Uses CoppeliaSim 4.1.0 + PyRep for simulation, shipped in the
``rlbench`` Docker image.  Xvfb is started by the entrypoint for
headless rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

# The Docker entrypoint starts Xvfb and sets DISPLAY=:99.
# Do NOT set QT_QPA_PLATFORM=offscreen — CoppeliaSim needs xcb+Xvfb.

DEFAULT_TASKS = [
    "reach_target",
    "pick_up_cup",
    "push_button",
    "close_drawer",
    "open_door",
]


class RLBenchBenchmark(StepBenchmark):
    """RLBench manipulation benchmark (CoppeliaSim / PyRep).

    Args:
        tasks: List of RLBench task file names (snake_case).
        image_size: Camera resolution (square).
        max_steps: Maximum steps per episode.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        image_size: int = 256,
        max_steps: int = 200,
    ) -> None:
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._image_size = image_size
        self._max_steps = max_steps
        self._env = None  # rlbench.environment.Environment
        self._task_env = None  # rlbench.task_environment.TaskEnvironment
        self._descriptions: list[str] = []

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.shutdown()
            except Exception:
                pass
            self._env = None
            self._task_env = None

    # ------------------------------------------------------------------ #
    # lazy init
    # ------------------------------------------------------------------ #
    def _ensure_env(self):
        if self._env is not None:
            return
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import CameraConfig, ObservationConfig

        cam = CameraConfig(
            rgb=True,
            depth=False,
            point_cloud=False,
            mask=False,
            image_size=(self._image_size, self._image_size),
        )
        cam_off = CameraConfig()
        cam_off.set_all(False)

        obs_cfg = ObservationConfig(
            front_camera=cam,
            wrist_camera=cam,
            left_shoulder_camera=cam_off,
            right_shoulder_camera=cam_off,
            overhead_camera=cam_off,
            joint_positions=True,
            gripper_open=True,
        )

        action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        )
        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_cfg,
            headless=True,
        )
        self._env.launch()

    # ------------------------------------------------------------------ #
    # StepBenchmark interface
    # ------------------------------------------------------------------ #
    def get_tasks(self) -> list[Task]:
        return [{"name": t, "task_file": t} for t in self._task_names]

    def reset(self, task: Task) -> Any:
        from rlbench import utils as rlbench_utils

        self._ensure_env()
        assert self._env is not None

        task_class = rlbench_utils.name_to_task_class(task["task_file"])
        self._task_env = self._env.get_task(task_class)
        self._task_env.sample_variation()
        self._descriptions, obs = self._task_env.reset()
        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        act = np.asarray(raw_action, dtype=np.float64)
        assert act.shape[-1] == 8, f"Action dimension mismatch: got {act.shape[-1]}, expected 8"

        # Expected: 8D (7 joint vel + 1 gripper discrete)
        if act.shape[0] < 8:
            act = np.pad(act, (0, 8 - act.shape[0]))
        act = act[:8]

        assert self._task_env is not None
        obs, reward, terminate = self._task_env.step(act)
        return StepResult(obs=obs, reward=reward, done=terminate, info={})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images = {}
        if raw_obs.front_rgb is not None:
            images["front"] = np.asarray(raw_obs.front_rgb, dtype=np.uint8)
        if raw_obs.wrist_rgb is not None:
            images["wrist"] = np.asarray(raw_obs.wrist_rgb, dtype=np.uint8)

        description = self._descriptions[0] if self._descriptions else task["name"]
        return {
            "images": images,
            "task_description": description,
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.reward > 0.99}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        # 8D: 7 joint velocities + 1 discrete gripper
        return {
            "joints": DimSpec("joints", 7, "joint_velocity"),
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "front": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "language": LANGUAGE,
        }
