"""MolmoSpaces-Bench benchmark implementation.

Wraps MolmoSpaces's JSON-based evaluation pipeline (JsonEvalTaskSampler →
BaseMujocoTask) so VLA model servers can evaluate on MolmoSpaces-Bench via
the vla-eval WebSocket/msgpack protocol.

This implementation matches the paper's evaluation path
(``olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig``) using:
- ``action_type = joint_pos`` (absolute joint positions, NOT relative)
- ``policy_dt_ms = 66.0``, command_mode = ``joint_position``
- ``task_horizon = 600`` (for pick-and-place tasks)

Camera name mapping:
- Primary camera: exo_camera_1 (maps to MolmoSpaces's
  ``droid_shoulder_light_randomization`` in the sensor suite output)
- Wrist camera: wrist_camera (maps to ``wrist_camera_zed_mini``)

Action format (over the wire from the model server):
- ``obs["actions"]`` is an 8-dim vector: 7 absolute arm joint positions +
  1 gripper command (0 or 255 after clamping, done by the model server).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_01, IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MUJOCO_GL", "egl")

# Fallback max steps used if ``task_horizon`` is not set in the config.
# The MolmoBot paper's README specifies ``task_horizon=600`` for pick-and-place
# (https://github.com/allenai/MolmoBot/blob/main/MolmoBot/README.md). Other task
# types have not been end-to-end verified through this harness yet; users should
# set ``task_horizon`` explicitly in the benchmark YAML when running them.
DEFAULT_TASK_HORIZON = 600

# Camera name aliases: primary = exo_camera_1, wrist = wrist_camera.
# MolmoSpaces env emits them under different names depending on the camera system.
PRIMARY_CAM_ALIASES = ("droid_shoulder_light_randomization", "exo_camera_1", "exo_camera")
WRIST_CAM_ALIASES = ("wrist_camera_zed_mini", "wrist_camera")

# Canonical wire names (what the model server expects).
PRIMARY_CAM = "exo_camera_1"
WRIST_CAM = "wrist_camera"


class MolmoSpacesBenchmark(StepBenchmark):
    """MolmoSpaces-Bench manipulation benchmark.

    Args:
        benchmark_dir: Path to a benchmark directory containing
            ``benchmark.json`` (or ``house_*/episode_*.json`` layout).
        eval_config_cls: Import string ``module:Class`` for the
            MlSpacesExpConfig subclass. Defaults to the paper's Franka
            state-8 clamped absolute-joint-position config.
        task_horizon: Override max steps per episode. If ``None``, uses
            the per-task defaults above.
        send_wrist_image: Include wrist camera in wire observations.
        send_state: Include proprioceptive state (qpos) in wire observations.
    """

    def __init__(
        self,
        benchmark_dir: str,
        eval_config_cls: str = "olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig",
        task_horizon: int | None = None,
        send_wrist_image: bool = True,
        send_state: bool = True,
    ) -> None:
        super().__init__()
        self.benchmark_dir = Path(benchmark_dir)
        self.eval_config_cls = eval_config_cls
        self.task_horizon = task_horizon
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state

        self._episodes: list[Any] = []
        self._exp_config: Any = None
        self._sampler: Any = None
        self._task: Any = None
        self._task_description: str = ""
        self._step_count: int = 0

    # -- data -------------------------------------------------------------

    def cleanup(self) -> None:
        if self._sampler is not None:
            try:
                if hasattr(self._sampler, "close"):
                    self._sampler.close()
                elif hasattr(self._sampler, "_env") and self._sampler._env is not None:
                    if hasattr(self._sampler._env, "close"):
                        self._sampler._env.close()
            except Exception:
                pass
            self._sampler = None
        self._task = None

    def get_tasks(self) -> list[Task]:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

        self._episodes = load_all_episodes(self.benchmark_dir)
        if not self._episodes:
            raise RuntimeError(f"No episodes found in {self.benchmark_dir}")
        logger.info("Loaded %d episodes from %s", len(self._episodes), self.benchmark_dir)

        tasks: list[Task] = []
        for i, ep in enumerate(self._episodes):
            task_cls = ep.get_task_cls().rsplit(".", 1)[-1]
            desc = ep.language.task_description[:60].replace("/", "_")
            tasks.append(
                {
                    "name": f"{task_cls}_{i:04d}_{desc}",
                    "episode_index": i,
                    "task_cls": task_cls,
                }
            )
        return tasks

    # -- episode lifecycle ------------------------------------------------

    def reset(self, task: Task) -> Any:
        from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

        # Build the eval exp_config lazily so MuJoCo/JAX aren't imported at class load time.
        if self._exp_config is None:
            self._exp_config = self._build_exp_config()

        ep_idx = task["episode_index"]
        episode_spec = self._episodes[ep_idx]
        self._task_description = episode_spec.language.task_description

        # Apply task_horizon (configured via YAML, or fall back to 600).
        horizon = self.task_horizon or DEFAULT_TASK_HORIZON
        self._exp_config.task_horizon = int(horizon)

        # JsonEvalRunner.patch_config → adds EvalRuntimeParams so _sample_task works.
        from molmo_spaces.evaluation.eval_main import EvalRuntimeParams

        self._exp_config.eval_runtime_params = EvalRuntimeParams()

        # Clean up any previous sampler before creating a new one.
        if self._sampler is not None:
            try:
                if hasattr(self._sampler, "_env") and self._sampler._env is not None:
                    if hasattr(self._sampler._env, "close"):
                        self._sampler._env.close()
            except Exception:
                pass

        # Create the per-episode task sampler and materialize the MuJoCo task.
        self._sampler = JsonEvalTaskSampler(self._exp_config, episode_spec)
        mujoco_task = self._sampler.sample_task(house_index=episode_spec.house_index)
        if mujoco_task is None:
            raise RuntimeError(f"Failed to create task for episode {ep_idx}")
        self._task = mujoco_task
        self._step_count = 0

        # BaseMujocoTask.reset() returns (obs_list, info) per gym convention.
        reset_output = self._task.reset()
        if isinstance(reset_output, tuple):
            raw_obs = reset_output[0]
        else:
            raw_obs = reset_output
        return self._unwrap_batch(raw_obs)

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        raw = np.asarray(raw, dtype=np.float32).flatten()
        if raw.size < 8:
            raise ValueError(f"Expected 8D action, got {raw.size}D: {raw}")

        # Split into MolmoSpaces's per-move-group action dict.
        env_action = {
            "arm": raw[:7].astype(np.float32).copy(),
            "gripper": raw[7:8].astype(np.float32).copy(),
        }

        assert self._task is not None
        step_output = self._task.step(env_action)
        self._step_count += 1

        # task.step() returns (obs, reward, terminated, truncated, info).
        obs, reward, terminated, truncated, info = step_output
        obs = self._unwrap_batch(obs)
        reward = self._scalar(reward, default=0.0)
        terminated = self._boolean(terminated)
        truncated = self._boolean(truncated)
        if isinstance(info, (list, tuple)):
            info = info[0] if info else {}

        done = bool(terminated or truncated)
        return StepResult(obs=obs, reward=float(reward), done=done, info=info or {})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        if not isinstance(raw_obs, dict):
            raw_obs = self._unwrap_batch(raw_obs)
        if not isinstance(raw_obs, dict):
            return {"images": {}, "task_description": self._task_description}

        images: dict[str, np.ndarray] = {}

        primary = self._find_camera(raw_obs, PRIMARY_CAM_ALIASES)
        if primary is not None:
            images[PRIMARY_CAM] = primary

        if self.send_wrist_image:
            wrist = self._find_camera(raw_obs, WRIST_CAM_ALIASES)
            if wrist is not None:
                images[WRIST_CAM] = wrist

        result: Observation = {
            "images": images,
            "task_description": self._task_description,
        }

        if self.send_state:
            qpos = self._extract_qpos(raw_obs)
            if qpos is not None:
                result["states"] = qpos

        return result

    def check_done(self, step_result: StepResult) -> bool:
        if step_result.done:
            return True
        if self._task is not None:
            try:
                return bool(self._scalar(self._task.is_done(), default=False))
            except Exception:
                pass
        return False

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        success = False
        if self._task is not None:
            try:
                judged = self._task.judge_success()
                success = bool(self._scalar(judged, default=False))
            except Exception:
                success = bool(step_result.info.get("success", False))
        else:
            success = bool(step_result.info.get("success", False))
        return {"success": success, "steps": self._step_count}

    # -- specs / metadata -------------------------------------------------

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self.task_horizon or 600,
            "eval_config_cls": self.eval_config_cls,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "joints": DimSpec("joints", 7, "joint_pos_abs", (-3.15, 3.15)),
            "gripper": GRIPPER_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            PRIMARY_CAM: IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec[WRIST_CAM] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_JOINT
        return spec

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean"}

    # -- private helpers --------------------------------------------------

    def _build_exp_config(self) -> Any:
        """Instantiate the eval config class specified by eval_config_cls."""
        import importlib

        module_path, class_name = self.eval_config_cls.split(":")
        module = importlib.import_module(module_path)
        eval_config_cls = getattr(module, class_name)
        exp_config = eval_config_cls()

        # Ensure no action noise for evaluation (the Franka configs already do
        # this, but harmless to enforce).
        if hasattr(exp_config.robot_config, "action_noise_config"):
            exp_config.robot_config.action_noise_config.enabled = False

        # Match the production eval pipeline defaults.
        exp_config.filter_for_successful_trajectories = False
        exp_config.num_workers = 1
        exp_config.seed = 42
        return exp_config

    @staticmethod
    def _unwrap_batch(obs: Any) -> Any:
        """Sensor suite returns list[dict] per batch; take the first."""
        if isinstance(obs, (list, tuple)) and obs:
            head = obs[0]
            if isinstance(head, dict):
                return head
        return obs

    @staticmethod
    def _scalar(value: Any, default: Any = None) -> Any:
        """Collapse per-batch scalars/arrays down to a single Python value."""
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        if isinstance(value, np.ndarray):
            return value.flatten()[0] if value.size else default
        return value

    @staticmethod
    def _boolean(value: Any) -> bool:
        val = MolmoSpacesBenchmark._scalar(value, default=False)
        try:
            return bool(val)
        except Exception:
            return False

    @staticmethod
    def _find_camera(obs: dict[str, Any], aliases: tuple[str, ...]) -> np.ndarray | None:
        """Find an RGB camera image under any of the given aliases."""
        for key in aliases:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                    return np.asarray(img, dtype=np.uint8)
        # Fallback: scan for any (H,W,3) image in the obs
        for key, val in obs.items():
            if (
                isinstance(val, np.ndarray)
                and val.ndim == 3
                and val.shape[-1] == 3
                and val.dtype == np.uint8
                and "camera" in key
            ):
                return val
        return None

    @staticmethod
    def _extract_qpos(obs: dict[str, Any]) -> np.ndarray | None:
        """Build the 8-dim proprioceptive state: 7 arm joints + 1 gripper."""
        robot_state = obs.get("robot_state")
        if isinstance(robot_state, dict) and "qpos" in robot_state:
            qpos = robot_state["qpos"]
            if isinstance(qpos, dict):
                arm = qpos.get("arm")
                gripper = qpos.get("gripper")
                parts: list[np.ndarray] = []
                if arm is not None:
                    parts.append(np.asarray(arm, dtype=np.float32).flatten())
                if gripper is not None:
                    # gripper_representation_count=1 in SynthVLAPolicyConfig
                    parts.append(np.asarray(gripper, dtype=np.float32).flatten()[:1])
                if parts:
                    return np.concatenate(parts)
        # Fallback: flat qpos array
        qpos = obs.get("qpos")
        if qpos is not None:
            return np.asarray(qpos, dtype=np.float32).flatten()
        return None
