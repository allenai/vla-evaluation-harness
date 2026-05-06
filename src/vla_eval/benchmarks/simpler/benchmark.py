"""SimplerEnv benchmark implementation.

Two evaluation protocols:

**Visual Matching (VM)** -- the default.  Uses
``simpler_env.make(task_name)`` which internally calls
``gym.make(env_name, obs_mode="rgbd", prepackaged_config=True)``.
The prepackaged config sets the correct control mode, scene, robot,
camera, RGB overlay, and robot init position for each task.

**Variant Aggregation (VA)** -- activated when *env_name* is set.
Calls ``gym.make(env_name, ...)`` directly with explicit scene,
lighting, distractor, and camera overrides.  Robot/object init
positions are controlled via *init_config* to reproduce the exact
grids from the official SimplerEnv evaluation scripts.

Success modes (set via model server ``get_observation_params()``):
    - ``truncation``: Run until ``max_episode_steps``.  Success =
      ``terminated`` on the final step.
    - ``early_stop``: Stop on the first ``terminated=True``.  Matches
      X-VLA official eval (``if done: break``).
    - ``accumulate``: Run until ``max_episode_steps``.  Success =
      ``terminated`` at any point during the episode.  Matches GR00T
      official eval (OR-accumulation).
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_CLOSE_POS, IMAGE_RGB, LANGUAGE, POSITION_DELTA, RAW, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task


def _linspace_from_range(spec: list[float]) -> np.ndarray:
    """Convert ``[start, end, n]`` to ``np.linspace(start, end, int(n))``."""
    return np.linspace(spec[0], spec[1], int(spec[2]))


class SimplerEnvBenchmark(StepBenchmark):
    """SimplerEnv (ManiSkill2 real2sim) benchmark.

    Args:
        task_name: SimplerEnv task identifier (e.g. ``"widowx_stack_cube"``).
            Must be a key in ``simpler_env.ENVIRONMENT_MAP``.
        max_episode_steps: Override environment's default episode length.
            ``None`` keeps the prepackaged default.  X-VLA uses 1200,
            GR00T uses 10000, starVLA/DB-CogACT use 120.
        success_mode: How to determine episode success:
            ``"truncation"`` -- success = terminated on the final step.
            ``"early_stop"`` -- stop on first terminated, count as success.
            ``"accumulate"`` -- run to end, success if ever terminated.
        send_state: Include proprioceptive state (base_pose, tcp_pose,
            EE pose) in observations for models that need it.
        seed: Random seed for ``env.reset()``.
        env_name: Gymnasium env ID for VA evaluation.  When set, calls
            ``gym.make()`` directly instead of ``simpler_env.make()``.
        scene_name: Scene name passed to ``gym.make()`` (VA only).
        env_build_kwargs: Extra kwargs forwarded to ``gym.make()``
            (VA only).  Lighting, distractors, shader, etc.
        init_config: Position grid config for VA evaluation.  Keys:
            ``robot_x/robot_y`` ([start, end, n] linspace),
            ``robot_rot_quat`` ([x, y, z, w] quaternion),
            ``obj_x/obj_y`` ([start, end, n] linspace, xy mode),
            ``obj_episode_range`` ([start, end) end-exclusive, episode mode).
    """

    def __init__(
        self,
        task_name: str = "widowx_stack_cube",
        max_episode_steps: int | None = None,
        success_mode: str = "truncation",
        send_state: bool = False,
        seed: int | None = None,
        deterministic_episodes: bool = True,
        control_mode: str | None = None,
        gripper_mode: str = "binary",
        env_name: str | None = None,
        scene_name: str | None = None,
        env_build_kwargs: dict[str, Any] | None = None,
        init_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert success_mode in ("truncation", "early_stop", "accumulate"), (
            f"Invalid success_mode={success_mode!r}. Expected: truncation, early_stop, accumulate"
        )
        assert gripper_mode in ("binary", "sticky"), f"Invalid gripper_mode={gripper_mode!r}. Expected: binary, sticky"
        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.success_mode = success_mode
        self.send_state = send_state
        self.seed = seed
        self.deterministic_episodes = deterministic_episodes
        self.control_mode = control_mode
        self.gripper_mode = gripper_mode
        self.env_name = env_name
        self.scene_name = scene_name
        self.env_build_kwargs = env_build_kwargs or {}

        self._env: Any = None
        self._task_description: str = ""
        self._success_seen: bool = False
        # Sticky gripper state (Google Robot)
        self._sticky_action_is_on: bool = False
        self._sticky_gripper_action: float = 0.0
        self._gripper_action_repeat: int = 0
        self._sticky_gripper_num_repeat: int = 15

        # Precompute VA position grid
        self._va_grid: list[tuple[np.ndarray, dict[str, Any]]] | None = None
        if init_config is not None:
            self._va_grid = self._build_position_grid(init_config)

    @staticmethod
    def _build_position_grid(cfg: dict[str, Any]) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Build a flat list of ``(robot_xy, obj_options)`` from *init_config*.

        Each entry maps to one episode.  The grid order matches the
        nested-for-loop order in SimplerEnv's evaluation scripts:
        outer = robot positions, inner = object positions / episode ids.
        """
        robot_xs = _linspace_from_range(cfg["robot_x"]) if "robot_x" in cfg else np.array([0.0])
        robot_ys = _linspace_from_range(cfg["robot_y"]) if "robot_y" in cfg else np.array([0.0])
        rot_quat = np.asarray(cfg.get("robot_rot_quat", [0, 0, 0, 1]), dtype=np.float64)

        grid: list[tuple[np.ndarray, dict[str, Any]]] = []

        if "obj_episode_range" in cfg:
            ep_start, ep_end = int(cfg["obj_episode_range"][0]), int(cfg["obj_episode_range"][1])
            for rx, ry in product(robot_xs, robot_ys):
                robot_xy = np.array([rx, ry])
                for ep_id in range(ep_start, ep_end):
                    grid.append(
                        (
                            robot_xy,
                            {"robot_rot_quat": rot_quat, "obj_init_options": {"episode_id": ep_id}},
                        )
                    )
        else:
            obj_xs = _linspace_from_range(cfg["obj_x"]) if "obj_x" in cfg else np.array([0.0])
            obj_ys = _linspace_from_range(cfg["obj_y"]) if "obj_y" in cfg else np.array([0.0])
            for rx, ry in product(robot_xs, robot_ys):
                robot_xy = np.array([rx, ry])
                for ox, oy in product(obj_xs, obj_ys):
                    grid.append(
                        (
                            robot_xy,
                            {"robot_rot_quat": rot_quat, "obj_init_options": {"init_xy": np.array([ox, oy])}},
                        )
                    )
        return grid

    def _build_va_reset_kwargs(self, task: Task) -> dict[str, Any]:
        """Map ``task["episode_idx"]`` to a position in the precomputed grid."""
        assert self._va_grid is not None
        idx = task.get("episode_idx", 0) % len(self._va_grid)
        robot_xy, extra = self._va_grid[idx]
        return {
            "options": {
                "robot_init_options": {
                    "init_xy": robot_xy,
                    "init_rot_quat": extra["robot_rot_quat"],
                },
                "obj_init_options": extra["obj_init_options"],
            },
        }

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        return [{"name": self.task_name, "task_name": self.task_name}]

    def reset(self, task: Task) -> Any:
        # Close previous env -- new env per episode (matches reference)
        self._success_seen = False
        self._sticky_action_is_on = False
        self._sticky_gripper_action = 0.0
        self._gripper_action_repeat = 0
        if self._env is not None:
            self._env.close()

        if self.env_name is not None:
            self._env, reset_kwargs = self._make_va_env(task)
        else:
            self._env, reset_kwargs = self._make_vm_env(task)
        if self.seed is not None:
            reset_kwargs["seed"] = self.seed

        obs, _ = self._env.reset(**reset_kwargs)

        # Task description from environment
        try:
            self._task_description = self._env.unwrapped.get_language_instruction()
        except AttributeError:
            self._task_description = self._env.get_wrapper_attr("get_language_instruction")()

        return obs

    def _common_make_kwargs(self) -> dict[str, Any]:
        """Build kwargs shared by both VA and VM env creation."""
        kw: dict[str, Any] = {}
        if self.control_mode is not None:
            kw["control_mode"] = self.control_mode
        if self.max_episode_steps is not None:
            kw["max_episode_steps"] = self.max_episode_steps
        return kw

    def _make_va_env(self, task: Task) -> tuple[Any, dict[str, Any]]:
        """VA path: ``gym.make()`` with explicit scene/lighting/kwargs."""
        import simpler_env  # noqa: F401 -- registers ManiSkill2 envs
        import gymnasium as gym

        make_kwargs = {"obs_mode": "rgbd", **self.env_build_kwargs, **self._common_make_kwargs()}
        if self.scene_name is not None:
            make_kwargs["scene_name"] = self.scene_name
        env = gym.make(self.env_name, **make_kwargs)

        reset_kwargs = self._build_va_reset_kwargs(task) if self._va_grid is not None else {}
        return env, reset_kwargs

    def _make_vm_env(self, task: Task) -> tuple[Any, dict[str, Any]]:
        """VM path: ``simpler_env.make()`` with prepackaged config."""
        import simpler_env

        env = simpler_env.make(self.task_name, **self._common_make_kwargs())

        # deterministic_episodes=True: pass episode_id for reproducible
        # object placement (matches X-VLA, starVLA reference evals).
        # deterministic_episodes=False: random placement each reset
        # (matches GR00T with vectorized envs + auto-reset).
        reset_kwargs: dict[str, Any] = {}
        if self.deterministic_episodes:
            episode_idx = task.get("episode_idx", 0)
            reset_kwargs["options"] = {"obj_init_options": {"episode_id": episode_idx}}
        return env, reset_kwargs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"Action dimension mismatch: got {len(raw_action)}, expected 7"

        # [pos3, rot3, gripper] — pass directly to env.step().
        # No rotation conversion: the controller (arm_pd_ee_target_delta_pose_align2)
        # uses Rotation.from_rotvec() which interprets action[3:6] as a rotation vector.
        # All reference implementations feed their rotation values straight through.
        pos = np.array(raw_action[:3])
        rot = np.array(raw_action[3:6])
        if self.gripper_mode == "sticky":
            # Google Robot: relative gripper with sticky repeat (15 steps)
            g = float(raw_action[6]) * 2 - 1  # [0,1] → [-1,1]
            relative = -g
            if abs(relative) > 0.5 and not self._sticky_action_is_on:
                self._sticky_action_is_on = True
                self._sticky_gripper_action = relative
            if self._sticky_action_is_on:
                self._gripper_action_repeat += 1
                relative = self._sticky_gripper_action
            if self._gripper_action_repeat == self._sticky_gripper_num_repeat:
                self._sticky_action_is_on = False
                self._gripper_action_repeat = 0
                self._sticky_gripper_action = 0.0
            gripper = relative
        else:
            gripper = 1.0 if raw_action[6] > 0.5 else -1.0
        env_action = np.concatenate([pos, rot, [gripper]])

        assert self._env is not None
        obs, reward, done, truncated, info = self._env.step(env_action)

        info["truncated"] = truncated
        if done:
            self._success_seen = True

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        image = get_image_from_maniskill2_obs_dict(self._env, raw_obs)

        obs: Observation = {
            "images": {"primary": image},
            "task_description": self._task_description,
            "task_name": self.task_name,
        }

        if self.send_state:
            agent = raw_obs.get("agent", {})
            extra = raw_obs.get("extra", {})

            # Send base_pose + tcp_pose for model servers that compute
            # base-relative EE pose (X-VLA, GR00T, etc.)
            base_pose = agent.get("base_pose")
            tcp_pose = extra.get("tcp_pose")
            if base_pose is not None and tcp_pose is not None:
                obs["base_pose"] = np.asarray(base_pose, dtype=np.float32)
                obs["tcp_pose"] = np.asarray(tcp_pose, dtype=np.float32)

            # Send pre-computed EE state if available (8D: pos3 + quat4_wxyz + gripper)
            eef = agent.get("eef_pos")
            if eef is not None:
                obs["states"] = np.asarray(eef, dtype=np.float32)
            elif base_pose is not None and tcp_pose is not None:
                from vla_eval.rotation import matrix_to_quat, pose7_wxyz_to_mat4, quat_xyzw_to_wxyz

                bp = np.asarray(base_pose, dtype=np.float64).flatten()
                tp = np.asarray(tcp_pose, dtype=np.float64).flatten()

                base_mat = pose7_wxyz_to_mat4(bp)
                tcp_mat = pose7_wxyz_to_mat4(tp)
                ee_in_base = np.linalg.inv(base_mat) @ tcp_mat
                pos = ee_in_base[:3, 3]
                q_wxyz = quat_xyzw_to_wxyz(matrix_to_quat(ee_in_base[:3, :3]))

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
        if self.success_mode == "early_stop":
            return step_result.done or step_result.info.get("truncated", False)
        # truncation and accumulate: run until max_episode_steps
        return step_result.info.get("truncated", False)

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        if self.success_mode == "accumulate":
            return {"success": self._success_seen}
        # truncation: success = terminated on final step
        # early_stop: success = terminated (which triggered the stop)
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "task_name": self.task_name,
            "success_mode": self.success_mode,
            "max_steps": self.max_episode_steps,
        }
        if self.env_name is not None:
            meta["env_name"] = self.env_name
            if self.scene_name is not None:
                meta["scene_name"] = self.scene_name
            if self.env_build_kwargs:
                meta["env_build_kwargs"] = self.env_build_kwargs
        return meta

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
