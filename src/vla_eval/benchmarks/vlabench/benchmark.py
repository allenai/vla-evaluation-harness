"""VLABench benchmark implementation.

VLABench is a large-scale benchmark for language-conditioned robotics
manipulation with long-horizon reasoning tasks, built on dm_control (MuJoCo).

Actions from the model server are 7-D: ``[dx, dy, dz, droll, dpitch, dyaw,
gripper]``.  These deltas are added to the current end-effector pose, then
converted to joint-space via inverse kinematics before being sent to the
dm_control environment as ``[7D qpos, 2D gripper]``.

Success is detected via dm_control's ``timestep.last()`` termination signal.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_TASKS = [
    "select_fruit",
    "select_toy",
    "select_drink",
    "select_book",
    "select_painting",
]

DEFAULT_MAX_STEPS = 200


class VLABenchBenchmark(StepBenchmark):
    """VLABench manipulation benchmark (dm_control / MuJoCo).

    Args:
        tasks: List of VLABench task names to evaluate.
        robot: Robot name (default ``"franka"``).
        max_steps: Maximum steps per episode (default 200).
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "franka",
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._robot = robot
        self._max_steps = max_steps
        self._env: Any = None
        self._current_task: str | None = None
        self._instruction: str = ""
        self._last_ee_state: np.ndarray | None = None

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _ensure_vlabench(self) -> None:
        """Lazy-import VLABench and register robots/tasks."""
        import VLABench  # noqa: F401 — triggers VLABENCH_ROOT
        import VLABench.robots  # noqa: F401 — registers robot classes
        import VLABench.tasks  # noqa: F401 — registers task classes

        # Monkey-patch to skip PCD generator (Open3D segfaults in headless
        # containers and we never request point clouds).
        from VLABench.envs.dm_env import LM4ManipDMEnv

        if not hasattr(LM4ManipDMEnv, "_pcd_patched"):
            # Stub that satisfies `self.pcd_generator.physics = ...`
            class _PcdStub:
                physics = None

            LM4ManipDMEnv.register_pcd_generator = lambda self: setattr(self, "pcd_generator", _PcdStub())
            LM4ManipDMEnv._pcd_patched = True

    def get_tasks(self) -> list[Task]:
        return [{"name": t} for t in self._task_names]

    def reset(self, task: Task) -> Any:
        self._ensure_vlabench()
        from VLABench.envs import load_env

        task_name = task["name"]

        # Close previous env and create new one (task may change scene layout)
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning("Failed to close VLABench environment: %s", e)
        self._env = load_env(task_name, robot=self._robot)
        self._current_task = task_name

        obs = self._env.get_observation(require_pcd=False)
        self._instruction = self._env.task.get_instruction()
        self._last_ee_state = obs.get("ee_state", None)
        return obs

    def step(self, action: Action) -> StepResult:
        from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(7, dtype=np.float32)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        assert raw_action.shape[-1] == 7, f"Action dimension mismatch: got {raw_action.shape[-1]}, expected 7"

        # Interpret 7D action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        delta_pos = raw_action[:3]
        delta_euler = raw_action[3:6]
        gripper_cmd = raw_action[6] if len(raw_action) > 6 else 0.0

        # Get current EE state for absolute target computation
        ee_state = self._last_ee_state
        if ee_state is None:
            ee_state = np.concatenate([self._env.get_ee_pos(), self._env.get_ee_quat(), [0.0]])
        current_pos = ee_state[:3]
        current_quat = ee_state[3:7]
        current_euler = np.array(quaternion_to_euler(current_quat))

        target_pos = current_pos + delta_pos
        target_euler = current_euler + delta_euler
        target_quat = euler_to_quaternion(*target_euler)

        # Inverse kinematics: ee pose → joint positions
        _, qpos = self._env.robot.get_qpos_from_ee_pos(
            physics=self._env.physics,
            pos=target_pos,
            quat=target_quat,
        )

        # Gripper: >0 → open (0.04), ≤0 → closed (0)
        grip_val = 0.04 if gripper_cmd > 0 else 0.0
        gripper_state = np.array([grip_val, grip_val])
        env_action = np.concatenate([qpos, gripper_state])

        timestep = self._env.step(env_action)
        success = bool(timestep.last())

        # Update cached EE state
        obs = self._env.get_observation(require_pcd=False)
        self._last_ee_state = obs.get("ee_state", None)
        self._instruction = self._env.task.get_instruction()

        return StepResult(
            obs=obs,
            reward=1.0 if success else 0.0,
            done=success,
            info={"success": success, "timestep": timestep},
        )

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images: dict[str, np.ndarray] = {}
        rgb = raw_obs.get("rgb", None)
        if rgb is not None and len(rgb) > 0:
            images["primary"] = rgb[0]  # (N_cams, H, W, 3) — take first camera

        return {
            "images": images,
            "task_description": self._instruction,
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

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
            "primary": IMAGE_RGB,
            "language": LANGUAGE,
        }
