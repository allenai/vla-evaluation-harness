"""SimplerEnv benchmark implementation.

Uses ``simpler_env.make(task_name)`` which internally calls
``gym.make(env_name, obs_mode="rgbd", prepackaged_config=True)``.
The prepackaged config sets the correct control mode, scene, robot,
camera, RGB overlay, and robot init position for each task.

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

from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_CLOSE_POS, IMAGE_RGB, LANGUAGE, POSITION_DELTA, RAW, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task


class SimplerEnvBenchmark(StepBenchmark):
    """SimplerEnv (ManiSkill2 real2sim) benchmark.

    Args:
        task_name: SimplerEnv task identifier (e.g. ``"widowx_stack_cube"``).
            Must be a key in ``simpler_env.ENVIRONMENT_MAP``.
        max_episode_steps: Override environment's default episode length.
            ``None`` keeps the prepackaged default.  X-VLA uses 1200,
            GR00T uses 10000, starVLA/DB-CogACT use 120.
        success_mode: How to determine episode success:
            ``"truncation"`` — success = terminated on the final step.
            ``"early_stop"`` — stop on first terminated, count as success.
            ``"accumulate"`` — run to end, success if ever terminated.
        send_state: Include proprioceptive state (base_pose, tcp_pose,
            EE pose) in observations for models that need it.
        seed: Random seed for ``env.reset()``.
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
    ) -> None:
        super().__init__()
        assert success_mode in ("truncation", "early_stop", "accumulate"), (
            f"Invalid success_mode={success_mode!r}. Expected: truncation, early_stop, accumulate"
        )
        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.success_mode = success_mode
        self.send_state = send_state
        self.seed = seed
        self.deterministic_episodes = deterministic_episodes
        self.control_mode = control_mode

        self._env: Any = None
        self._task_description: str = ""
        self._success_seen: bool = False

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
        import simpler_env

        # Close previous env — new env per episode (matches reference)
        self._success_seen = False
        if self._env is not None:
            self._env.close()

        make_kwargs: dict[str, Any] = {}
        if self.control_mode is not None:
            make_kwargs["control_mode"] = self.control_mode
        if self.max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = self.max_episode_steps
        self._env = simpler_env.make(self.task_name, **make_kwargs)

        # Reset — robot init is handled by prepackaged_config internally.
        # deterministic_episodes=True: pass episode_id for reproducible object placement
        #   (matches X-VLA, starVLA reference evals).
        # deterministic_episodes=False: no obj_init_options, random placement each reset
        #   (matches GR00T reference eval which uses vectorized envs with auto-reset).
        reset_kwargs: dict[str, Any] = {}
        if self.deterministic_episodes:
            episode_idx = task.get("episode_idx", 0)
            reset_kwargs["options"] = {"obj_init_options": {"episode_id": episode_idx}}
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

        # [pos3, rot3, gripper] — pass directly to env.step().
        # No rotation conversion: the controller (arm_pd_ee_target_delta_pose_align2)
        # uses Rotation.from_rotvec() which interprets action[3:6] as a rotation vector.
        # All reference implementations feed their rotation values straight through.
        pos = np.array(raw_action[:3])
        rot = np.array(raw_action[3:6])
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
                from vla_eval.rotation import quat_to_matrix, matrix_to_quat

                bp = np.asarray(base_pose, dtype=np.float64).flatten()
                tp = np.asarray(tcp_pose, dtype=np.float64).flatten()

                def _pose7_to_mat4(p: np.ndarray) -> np.ndarray:
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
        return {
            "task_name": self.task_name,
            "success_mode": self.success_mode,
            "max_steps": self.max_episode_steps,
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
