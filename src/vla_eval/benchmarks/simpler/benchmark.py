"""SimplerEnv benchmark implementation using ManiSkill2.

Creates a fresh environment per episode.  Each episode corresponds
to a distinct ``obj_episode_id`` from ``obj_episode_range``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_CLOSE_POS, IMAGE_RGB, LANGUAGE, POSITION_DELTA, RAW, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task


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
        control_mode: str | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        pass_rotation_raw: bool = False,
        accumulate_success: bool = False,
        prepackaged_config: bool = False,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.send_state = send_state
        self._control_mode_override = control_mode
        self._pass_rotation_raw = pass_rotation_raw
        self._accumulate_success = accumulate_success
        self._prepackaged_config = prepackaged_config
        self._success_seen = False
        self.image_size = tuple(image_size) if image_size is not None else None
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
        if self.image_size is not None and image.shape[:2] != self.image_size:
            import cv2

            image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
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
        self._success_seen = False
        if self._env is not None:
            self._env.close()

        if self._prepackaged_config:
            build_kwargs: dict[str, Any] = dict(
                obs_mode="rgbd",
                prepackaged_config=True,
                max_episode_steps=self.max_episode_steps,
            )
        else:
            control_mode = self._control_mode_override or get_robot_control_mode(self.robot, "vla")
            build_kwargs = dict(
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
        pos = np.array(raw_action[:3])
        if self._control_mode_override or self._pass_rotation_raw:
            # Absolute EE control (X-VLA) or raw pass-through (GR00T):
            # rotation passed directly without euler→axangle conversion.
            rot = np.array(raw_action[3:6])
        else:
            # Default delta control: convert euler → axis-angle for ManiSkill2
            from vla_eval.rotation import euler_xyz_to_matrix, matrix_to_quat, quat_to_axisangle

            mat = euler_xyz_to_matrix(np.array(raw_action[3:6]))
            rot = quat_to_axisangle(matrix_to_quat(mat))
        gripper = 1.0 if raw_action[6] > 0.5 else -1.0

        env_action = np.concatenate([pos, rot, [gripper]])
        assert self._env is not None
        obs, reward, done, truncated, info = self._env.step(env_action)

        info["truncated"] = truncated
        if self._accumulate_success and done:
            self._success_seen = True
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        image = get_image_from_maniskill2_obs_dict(self._env, raw_obs)
        obs = self._build_obs_dict(image)
        if self.send_state:
            agent = raw_obs.get("agent", {})
            extra = raw_obs.get("extra", {})
            # Send base_pose + tcp_pose so model servers can compute
            # base-relative EE pose (required by X-VLA, GR00T, etc.)
            base_pose = agent.get("base_pose")
            tcp_pose = extra.get("tcp_pose")
            if base_pose is not None and tcp_pose is not None:
                obs["base_pose"] = np.asarray(base_pose, dtype=np.float32)
                obs["tcp_pose"] = np.asarray(tcp_pose, dtype=np.float32)
            # Compute base-relative EE pose (8D: pos3+quat4_wxyz+gripper_openness).
            # Matches NVIDIA's ManiSkill2 fork (youliangtan/ManiSkill2_real2sim).
            eef = agent.get("eef_pos")
            if eef is not None:
                obs["states"] = np.asarray(eef, dtype=np.float32)
            elif base_pose is not None and tcp_pose is not None:
                from vla_eval.rotation import quat_to_matrix, matrix_to_quat

                bp = np.asarray(base_pose, dtype=np.float64).flatten()
                tp = np.asarray(tcp_pose, dtype=np.float64).flatten()

                # Build 4x4 transforms (ManiSkill2 quaternion: wxyz)
                def _pose7_to_mat4(p):
                    m = np.eye(4)
                    q_wxyz = p[3:7]
                    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                    m[:3, :3] = quat_to_matrix(q_xyzw)
                    m[:3, 3] = p[:3]
                    return m

                base_mat = _pose7_to_mat4(bp)
                tcp_mat = _pose7_to_mat4(tp)
                ee_in_base = np.linalg.inv(base_mat) @ tcp_mat
                pos = ee_in_base[:3, 3]
                q_xyzw = matrix_to_quat(ee_in_base[:3, :3])
                q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                # Gripper openness: 1 - closedness. Use env's get_gripper_closedness if available.
                assert self._env is not None
                try:
                    closedness = self._env.unwrapped.agent.get_gripper_closedness()
                    gripper_open = 1.0 - float(closedness)
                except Exception:
                    qpos = agent.get("qpos")
                    gripper_open = float(qpos[-1]) if qpos is not None else 0.0
                obs["states"] = np.concatenate([pos, q_wxyz, [gripper_open]]).astype(np.float32)
        return obs

    def check_done(self, step_result: StepResult) -> bool:
        # Run until truncated (max_episode_steps), never stop early on
        # terminated.  The success condition can flip back to False if the
        # robot disturbs the object after a momentary success.
        return step_result.info.get("truncated", False)

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        # Default: success = terminated on the final step (at truncation).
        # accumulate_success: success if terminated at any point during the episode
        # (matches GR00T official eval which OR-accumulates success).
        success = self._success_seen if self._accumulate_success else step_result.done
        return {"success": success}

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
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "primary": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_state:
            spec["state"] = RAW
        return spec
