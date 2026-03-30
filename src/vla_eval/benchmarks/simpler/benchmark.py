"""SimplerEnv benchmark implementation using ManiSkill2.

Creates a fresh environment per episode.  Each episode corresponds
to a distinct ``obj_episode_id`` from ``obj_episode_range``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import (
    GRIPPER_CLOSE_NEG,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_EULER,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task


def _euler2axangle(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to compact axis-angle vector."""
    from transforms3d.euler import euler2axangle as _e2a

    axis, angle = _e2a(float(euler[0]), float(euler[1]), float(euler[2]))
    return np.asarray(axis) * angle


class SimplerEnvBenchmark(StepBenchmark):
    """SimplerEnv (ManiSkill2 real2sim) benchmark (SAPIEN + Vulkan).

    Non-obvious behaviors:
        - **Vulkan required**: SAPIEN rendering needs Vulkan drivers.  Docker
          configs set ``NVIDIA_DRIVER_CAPABILITIES=all`` and mount the Vulkan
          ICD for this reason.
        - **New env per episode**: Unlike other benchmarks, a fresh environment
          is created for each episode (matching the reference implementation).
        - **RGB overlay + scene must match**: ``rgb_overlay_path`` and
          ``scene_name`` are paired (e.g. ``bridge_real_eval_1.png`` with
          ``bridge_table_1_v1``).  Mismatched pairs cause domain gap.
        - **obj_episode_range and episodes_per_task**: The range ``[0, 24]``
          means episode indices 0–23.  Set ``episodes_per_task: 24`` to cover
          all variations.
        - **Success semantics**: Runs until truncation (``max_episode_steps``).
          Success = ``terminated`` on the final step.  Early termination is
          ignored because success can flip back to False.

    Args:
        env_name: SimplerEnv environment ID.
        scene_name: SAPIEN scene identifier.
        robot: Robot model name (e.g. "widowx").
        control_freq: Control frequency in Hz.
        sim_freq: Simulation frequency in Hz.
        max_episode_steps: Max steps per episode.
        rgb_overlay_path: Path to real-world inpainting overlay image.
        robot_init_x, robot_init_y: Robot base position.
        robot_init_rot_quat_center: Center quaternion ``[x, y, z, w]``.
        robot_init_rot_rpy_range: RPY range as 9 floats.
        obj_variation_mode: Object variation selection mode.
        obj_episode_range: ``[start, end)`` range of object variation IDs.
        seed: Random seed for ``env.reset()``.  ``None`` → no seed.
    """

    def __init__(
        self,
        env_name: str = "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        scene_name: str = "bridge_table_1_v1",
        robot: str = "widowx",
        control_freq: int = 5,
        sim_freq: int = 500,
        max_episode_steps: int = 120,
        robot_init_x: float = 0.147,
        robot_init_y: float = 0.028,
        robot_init_rot_quat_center: list[float] | None = None,
        robot_init_rot_rpy_range: list[float] | None = None,
        obj_variation_mode: str = "episode",
        obj_episode_range: list[int] | None = None,
        rgb_overlay_path: str | None = None,
        seed: int | None = None,
        send_state: bool = False,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.send_state = send_state
        self.env_name = env_name
        self.scene_name = scene_name
        self.robot = robot
        self.control_freq = control_freq
        self.sim_freq = sim_freq
        self.max_episode_steps = max_episode_steps
        self.robot_init_x = robot_init_x
        self.robot_init_y = robot_init_y
        self.obj_variation_mode = obj_variation_mode
        self.obj_episode_range = obj_episode_range or [0, 24]

        # Compute single robot init quaternion from config
        quat_center = robot_init_rot_quat_center or [0, 0, 0, 1]
        rpy_range = robot_init_rot_rpy_range or [0, 0, 1, 0, 0, 1, 0, 0, 1]
        self._robot_init_quat = self._compute_init_quat(quat_center, rpy_range)

        self.rgb_overlay_path = rgb_overlay_path

        self._env = None
        self._task_description: str = ""

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_init_quat(center: list[float], rpy_range: list[float]) -> np.ndarray:
        """Compute robot init quaternion (matches reference Args._process_robot_position_args)."""
        from transforms3d.euler import euler2quat
        from sapien.core import Pose

        r, p, y = float(rpy_range[0]), float(rpy_range[3]), float(rpy_range[6])
        return (Pose(q=euler2quat(r, p, y)) * Pose(q=center)).q

    def _build_obs_dict(self, image: np.ndarray) -> dict[str, Any]:
        """Wrap image and task description into an Observation dict."""
        return {"images": {"primary": image}, "task_description": self._task_description}

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        return [
            {
                "name": self.env_name,
                "env_name": self.env_name,
                "scene_name": self.scene_name,
            }
        ]

    def reset(self, task: Task) -> Any:
        from simpler_env.utils.env.env_builder import (
            build_maniskill2_env,
            get_robot_control_mode,
        )

        # Close previous env — new env per episode (matches reference)
        if self._env is not None:
            self._env.close()

        control_mode = get_robot_control_mode(self.robot, "vla")
        build_kwargs: dict[str, Any] = dict(
            obs_mode="rgbd",
            robot=self.robot,
            scene_name=task.get("scene_name", self.scene_name),
            control_freq=self.control_freq,
            sim_freq=self.sim_freq,
            max_episode_steps=self.max_episode_steps,
            control_mode=control_mode,
            camera_cfgs={"add_segmentation": True},
        )
        if self.rgb_overlay_path is not None:
            build_kwargs["rgb_overlay_path"] = self.rgb_overlay_path

        self._env = build_maniskill2_env(
            task.get("env_name", self.env_name),
            **build_kwargs,
        )

        # Reset with robot init + object variation (matches reference)
        episode_idx = task.get("episode_idx", 0)
        obj_episode_id = self.obj_episode_range[0] + episode_idx

        env_reset_options = {
            "robot_init_options": {
                "init_xy": np.array([self.robot_init_x, self.robot_init_y]),
                "init_rot_quat": self._robot_init_quat,
            },
            "obj_init_options": {"episode_id": obj_episode_id},
        }
        reset_kwargs: dict[str, Any] = {"options": env_reset_options}
        if self.seed is not None:
            reset_kwargs["seed"] = self.seed
        obs, _ = self._env.reset(**reset_kwargs)

        # Task description from environment
        try:
            self._task_description = self._env.unwrapped.get_language_instruction()
        except AttributeError:
            self._task_description = self._env.get_wrapper_attr("get_language_instruction")()

        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"Action dimension mismatch: got {len(raw_action)}, expected 7"

        # [x, y, z, roll, pitch, yaw, gripper] -> ManiSkill2 format
        # Rotation passed directly (not converted to axis-angle) matching
        # official eval pipelines which pass model output as-is to env.step().
        pos = np.array(raw_action[:3])
        rot = np.array(raw_action[3:6])
        gripper = 1.0 if raw_action[6] > 0.5 else -1.0

        env_action = np.concatenate([pos, rot, [gripper]])
        assert self._env is not None
        obs, reward, done, truncated, info = self._env.step(env_action)

        info["truncated"] = truncated
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        image = get_image_from_maniskill2_obs_dict(self._env, raw_obs)
        obs = self._build_obs_dict(image)
        if self.send_state:
            eef = raw_obs.get("agent", {}).get("eef_pos")
            if eef is not None:
                obs["states"] = np.asarray(eef, dtype=np.float32)
        return obs

    def check_done(self, step_result: StepResult) -> bool:
        # Run until truncated (max_episode_steps), never stop early on
        # terminated.  The success condition can flip back to False if the
        # robot disturbs the object after a momentary success.
        return step_result.info.get("truncated", False)

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        # Success = terminated on the final step (evaluated at truncation).
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_episode_steps,
            "env_name": self.env_name,
            "robot": self.robot,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "primary": IMAGE_RGB,
            "language": LANGUAGE,
        }
