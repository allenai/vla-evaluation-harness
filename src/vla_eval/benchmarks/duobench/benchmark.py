"""DuoBench benchmark implementation.

11 MuJoCo sim tasks from RobotControlStack's ``rcs`` Gym stack
(https://github.com/RobotControlStack/duobench).

Action: the model sends a flat vector ``[left_arm_action, right_arm_action,
left_gripper, right_gripper]`` which ``step()`` packs into RCS's nested
``{"left": ..., "right": ...}`` dict.  Each per-arm action depends on the control mode —
``joints`` (7 joint angles), ``cartesian_trpy`` (pose xyz+rpy, 6) or
``cartesian_tquat`` (pose xyz+quat, 7) — absolute, or deltas with
``relative_to=last_step``.  The gripper is a single ``[0, 1]`` value (0 = closed,
1 = open); ``binary_gripper`` decides whether the env rounds it to a grasp/open
command or applies it as a continuous width.

Observation: ``make_obs`` returns ``{"images": {<cam>: HWC uint8 RGB},
"task_description": str}`` and, when ``send_state`` is set, a flat ``"states"``
vector carrying DuoBench's full per-arm proprio (joints + EE pose + gripper).

The env is rebuilt on each task switch; results group under a single
``duobench/<task_id>/...`` namespace.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.duobench.utils import ensure_mujoco_arena_memory, extract_rgb, resolve_enum
from vla_eval.specs import (
    GRIPPER_01,
    IMAGE_RGB,
    JOINT_ABSOLUTE,
    JOINT_DELTA,
    LANGUAGE,
    POSITION_ABSOLUTE,
    POSITION_DELTA,
    ROTATION_EULER,
    ROTATION_QUAT,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

logger = logging.getLogger(__name__)


# Per-task episode horizons (https://arxiv.org/pdf/2606.11901, Table 4).
TASK_HORIZONS: dict[str, int] = {
    "ball_maze": 350,
    "bin_sort": 216,
    "block_balance": 766,
    "carry_pot": 450,
    "hinge_chest": 441,
    "join_blocks": 829,
    "pour_marbles": 442,
    "spring_door": 807,
    "transfer_cube": 441,
    "transfer_gate": 523,
    "transfer_reorient": 505,
}
DUOBENCH_TASK_IDS = tuple(TASK_HORIZONS.keys())

# Per control mode, the key under which each arm's action goes.
ARM_ACTION_BY_CONTROL_MODE: dict[str, str] = {
    "joints": "joints",
    "cartesian_trpy": "xyzrpy",
    "cartesian_tquat": "tquat",
}

# Dimension of per-arm proprioception.
PROPRIO_DIMS: dict[str, int] = {
    "joints": 7,  # joint angles
    "xyzrpy": 6,  # EE pose: position(3) + euler(3)
    "tquat": 7,  # EE pose: position(3) + quaternion(4)
    "gripper": 1,  # gripper
}


class DuoBenchBenchmark(StepBenchmark):
    """DuoBench bimanual Franka Research 3 Duo benchmark (MuJoCo/RobotControlStack).

    Non-obvious behaviors:
        - **Headless rendering**: sets ``MUJOCO_GL=egl`` / ``PYOPENGL_PLATFORM=egl``
          on import for GPU-accelerated headless rendering.
        - **MuJoCo arena bump**: raises each scene's arena/stack memory before compile
          (:func:`ensure_mujoco_arena_memory`) to preempt fatal ``mj_stackAlloc``
          crashes on contact-heavy scenes.
        - **Per-task horizons**: the env never truncates, so each task is capped at its
          99th-percentile length (:data:`TASK_HORIZONS`); ``max_steps`` overrides.
        - **Env rebuilt per task**: rebuilt on each task switch (a few seconds, cheap
          next to inference).
        - **Eval sim config**: fixed control period, run flat out (``async_control``,
          ``realtime=False``) — overrides DuoBench's teleop defaults.

    Args:
        task_ids: Subset of :data:`DUOBENCH_TASK_IDS` to evaluate (default: all 11).
        seed: Base seed; episode ``i`` resets with ``seed + i``.
        frequency: Sim control frequency in Hz (default 30).
        max_steps: Uniform per-episode cap; ``None`` (default) uses the per-task
            horizons in :data:`TASK_HORIZONS`.
        send_state: Include the flat 42-D bimanual proprio state (21 per arm; default True).
        camera_resolution: Per-camera *native* render resolution ``{cam: [H, W]}`` set
            on the RCS scene (``None`` = each camera's DuoBench default).
        control_mode: RCS control mode (case-insensitive) — ``"joints"``,
            ``"cartesian_trpy"``, or ``"cartesian_tquat"`` (default).
        relative_to: RCS action frame (case-insensitive) — ``"none"`` (absolute),
            ``"last_step"`` (delta, default), or ``"configured_origin"``.
        binary_gripper: Round the gripper command to grasp/open instead of a
            continuous width (default False).
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success", "stage"})

    def __init__(
        self,
        task_ids: list[str] | tuple[str, ...] | None = None,
        seed: int = 0,
        frequency: int = 30,
        max_steps: int | None = None,
        send_state: bool = True,
        camera_resolution: Mapping[str, Sequence[int]] | None = None,
        control_mode: str = "cartesian_tquat",
        relative_to: str = "last_step",
        binary_gripper: bool = False,
    ) -> None:
        super().__init__()
        ids = tuple(task_ids) if task_ids else DUOBENCH_TASK_IDS
        unknown = [t for t in ids if t not in DUOBENCH_TASK_IDS]
        if unknown:
            raise ValueError(f"Unknown duobench task_id(s) {unknown}.  Known: {DUOBENCH_TASK_IDS}")
        self.task_ids: tuple[str, ...] = ids
        self.seed = seed
        self.frequency = frequency
        self._max_steps = max_steps
        self.send_state = send_state
        self._camera_resolution = camera_resolution
        self.control_mode = control_mode
        if self.control_mode not in ARM_ACTION_BY_CONTROL_MODE:
            raise ValueError(f"control_mode must be one of {sorted(ARM_ACTION_BY_CONTROL_MODE)}, got {control_mode!r}")
        self._arm_action_key = ARM_ACTION_BY_CONTROL_MODE[self.control_mode]
        self._arm_action_dim = PROPRIO_DIMS[self._arm_action_key]
        self.relative_to = relative_to
        self.binary_gripper = binary_gripper

        self._env = None
        self._current_task_id: str | None = None
        self._instruction: str = ""
        self._max_stage_reached = 0
        self._elapsed = 0
        self._cap = 0

    @staticmethod
    def _task_config(task_id: str) -> Any:
        """Build the RCS ``EnvConfig`` for ``task_id`` (registers its gym id)."""
        task_module = __import__(f"duobench.tasks.{task_id}", fromlist=["*"])
        env_config_cls = next((cls for name, cls in vars(task_module).items() if name.endswith("EnvConfig")), None)
        if env_config_cls is None:
            raise RuntimeError(f"duobench.tasks.{task_id} defines no *EnvConfig class")
        return env_config_cls().config()

    def _build_env(self, task_id: str):
        """Construct the duobench Gym env for ``task_id`` in the configured mode."""
        import gymnasium as gym
        from duobench import tasks as _duobench_tasks  # noqa: F401  (registers gym env ids).
        from rcs.envs.base import ControlMode, RelativeTo
        from rcs.envs.scenes import SimConfig

        cfg = self._task_config(task_id)
        cfg.headless = True
        cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=self.frequency)
        cfg.control_mode = resolve_enum(ControlMode, self.control_mode, "control_mode")
        cfg.relative_to = resolve_enum(RelativeTo, self.relative_to, "relative_to")
        cfg.wrapper_cfg.binary_gripper = self.binary_gripper
        self._apply_camera_resolution(cfg)
        ensure_mujoco_arena_memory()  # preempt mj_stackAlloc overflow on contact-heavy scenes.
        return gym.make(f"duobench/{task_id}", cfg=cfg)

    def _apply_camera_resolution(self, cfg: Any) -> None:
        """Set each camera's native render resolution on RCS's ``camera_cfgs``.

        ``camera_resolution`` is ``{cam: (height, width)}`` (harness convention); RCS
        stores each as ``resolution_height`` / ``resolution_width`` on its
        ``SimCameraConfig``.  No-op when unset; raises on an unknown camera name.
        """
        if not self._camera_resolution:
            return
        available = cfg.camera_cfgs or {}
        for cam, hw in self._camera_resolution.items():
            cam_cfg = available.get(cam)
            if cam_cfg is None:
                raise ValueError(f"camera_resolution camera {cam!r} not in scene cameras {sorted(available)}")
            cam_cfg.resolution_height, cam_cfg.resolution_width = int(hw[0]), int(hw[1])

    def cleanup(self) -> None:
        self._close_env()

    def _close_env(self) -> None:
        """Close and drop the current env, logging (not raising) on failure."""
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception:
            logger.exception("duobench env close failed")
        self._env = None

    def get_tasks(self) -> list[Task]:
        return [{"name": tid, "task_id": tid} for tid in self.task_ids]

    def reset(self, task: Task) -> Any:
        # Rebuild env on task switch.
        if task["task_id"] != self._current_task_id:
            self._close_env()

        if self._env is None:
            self._env = self._build_env(task["task_id"])
            self._current_task_id = task["task_id"]

        self._max_stage_reached = 0
        self._elapsed = 0
        self._cap = self._max_steps if self._max_steps is not None else TASK_HORIZONS[task["task_id"]]
        obs, info = self._env.reset(seed=self.seed + task["episode_idx"])
        self._instruction = info["instruction"]
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        flat = np.asarray(raw, dtype=np.float64).flatten()
        arm_action_key, arm_action_dim = self._arm_action_key, self._arm_action_dim
        expected = 2 * arm_action_dim + 2
        if flat.shape[0] != expected:
            raise ValueError(
                f"DuoBench in {self.control_mode!r} mode expects a flat {expected}-D action "
                f"[left_{arm_action_key}({arm_action_dim}), right_{arm_action_key}({arm_action_dim}), "
                f"left_gripper(1), right_gripper(1)]; got shape {flat.shape}"
            )

        def _arm(arm_action: np.ndarray, gripper: float) -> dict[str, np.ndarray]:
            return {
                arm_action_key: arm_action,
                "gripper": np.array([float(np.clip(gripper, 0.0, 1.0))], dtype=np.float32),
            }

        action_dict = {
            "left": _arm(flat[0:arm_action_dim], flat[2 * arm_action_dim]),
            "right": _arm(flat[arm_action_dim : 2 * arm_action_dim], flat[2 * arm_action_dim + 1]),
        }

        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(action_dict)
        self._elapsed += 1
        done = bool(terminated or truncated or self._elapsed >= self._cap)
        stage = int(info["stage"])
        self._max_stage_reached = max(self._max_stage_reached, stage)
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=done, success=bool(info["success"]), stage=stage)
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        return extract_rgb(raw_obs["frames"]["head"])

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {
            "success": bool(step_result.info["success"]),
            "progress": self._max_stage_reached / step_result.info["max_stage"],
        }

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images = {cam: rgb for cam, payload in raw_obs["frames"].items() if (rgb := extract_rgb(payload)) is not None}
        obs_dict: dict[str, Any] = {
            "images": images,
            "task_description": self._instruction,
        }
        if self.send_state:
            obs_dict["states"] = np.concatenate(
                [
                    np.asarray(raw_obs[side][field], dtype=np.float32).flatten()
                    for side in ("left", "right")
                    for field in PROPRIO_DIMS
                ]
            )
        return obs_dict

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "head": IMAGE_RGB,
            "left_wrist": IMAGE_RGB,
            "right_wrist": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_state:
            spec["state"] = DimSpec(
                "state",
                2 * sum(PROPRIO_DIMS.values()),  # 42 = left(21) + right(21)
                "duobench_bimanual_left_then_right[joints7_xyzrpy6_tquat7_gripper1]",
                description=(
                    "Flat bimanual proprio.  Left arm then right arm; each 21-dim block is: "
                    "joints[0:7] (joint angles); xyzrpy[7:13] (EE pose as position(3)+euler(3)); "
                    "tquat[13:20] (same EE pose as position(3)+quaternion(4)); gripper[20:21] (width).  "
                    "Right arm repeats at +21."
                ),
            )
        return spec

    def get_action_spec(self) -> dict[str, DimSpec]:
        delta = self.relative_to == "last_step"
        if self.control_mode == "joints":
            segments = {"joints": replace(JOINT_DELTA if delta else JOINT_ABSOLUTE, dims=7)}
        else:  # cartesian_trpy / cartesian_tquat
            position = POSITION_DELTA if delta else POSITION_ABSOLUTE
            rotation = ROTATION_QUAT if self.control_mode == "cartesian_tquat" else ROTATION_EULER
            segments = {"position": position, "rotation": rotation}
        spec = {f"{side}_{seg}": s for side in ("left", "right") for seg, s in segments.items()}
        spec["left_gripper"] = GRIPPER_01
        spec["right_gripper"] = GRIPPER_01
        return spec

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "progress": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        ceiling = self._max_steps if self._max_steps is not None else max(TASK_HORIZONS.values())
        return {"max_steps": ceiling}
