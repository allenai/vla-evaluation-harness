"""LIBERO benchmark implementation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.libero.utils import preprocess_libero_image
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

# EGL for headless rendering
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEP_MAPPING = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
    "libero_90": 400,
}


class LIBEROBenchmark(StepBenchmark):
    """LIBERO tabletop manipulation benchmark (MuJoCo/robosuite).

    Non-obvious behaviors:
        - **PyTorch compat**: Patches ``torch.load`` to use
          ``weights_only=False`` for PyTorch ≥2.6 compatibility with LIBERO's
          initial-state files (numpy arrays stored via ``torch.save``).
        - **Headless rendering**: Sets ``EGL_PLATFORM=device`` and
          ``PYOPENGL_PLATFORM=egl`` on import for GPU-accelerated headless
          rendering.
        - **Dummy wait steps**: At episode start, ``num_steps_wait`` steps
          (default 10) are executed with a fixed open-gripper action to let
          objects settle in the physics simulation.
        - **Suite-specific max_steps**: libero_spatial=220, libero_object=280,
          libero_goal=300, libero_10=520, libero_90=400.
        - **Image preprocessing**: robosuite renders images with inverted axes.
          Both agentview and wrist images are flipped ``[::-1, ::-1]`` to
          correct orientation, then resized to 256×256 with padding.

    Args:
        suite: LIBERO suite name (e.g. "libero_spatial", "libero_10").
        seed: Random seed for environment initialization.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive 8-D state
            ``[pos3, axisangle3, gripper2]`` in observations.
        absolute_action: Use absolute (world-frame) actions instead of delta.
            When True, sets ``robot.controller.use_delta = False`` after the
            initial dummy-wait steps.
        max_steps: Override the default suite-specific max step count.
            When None, uses ``MAX_STEP_MAPPING[suite]``.
    """

    def __init__(
        self,
        suite: str = "libero_spatial",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.suite = suite
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.absolute_action = absolute_action
        self._max_steps = max_steps
        self._env = None
        self._task_suite = None
        self._current_task_id: int | None = None

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _init_libero(self) -> None:
        """Lazily initialize LIBERO (heavy imports)."""
        if self._task_suite is not None:
            return
        # LIBERO init states use torch.save with numpy arrays.
        # PyTorch ≥2.6 defaults weights_only=True which blocks numpy globals.
        # Patch torch.load to default weights_only=False for LIBERO compatibility.
        import functools

        import torch

        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_load

        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[self.suite]()

    def get_tasks(self) -> list[Task]:
        self._init_libero()
        assert self._task_suite is not None
        tasks = []
        for task_id in range(self._task_suite.n_tasks):
            task = self._task_suite.get_task(task_id)
            tasks.append(
                {
                    "name": task.language,
                    "suite": self.suite,
                    "task_id": task_id,
                    "task_obj": task,
                }
            )
        return tasks

    def reset(self, task: Task) -> Any:
        from pathlib import Path

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_obj = task["task_obj"]
        task_id = task["task_id"]
        episode_idx = task.get("episode_idx", 0)

        # Only create a new env when the task changes (reuse across episodes)
        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._env.close()

            bddl_file = Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
            env_args = {
                "bddl_file_name": str(bddl_file),
                "camera_heights": LIBERO_ENV_RESOLUTION,
                "camera_widths": LIBERO_ENV_RESOLUTION,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(self.seed)
            self._env = env
            self._current_task_id = task_id

        # Reset env before setting init state (matches reference)
        self._env.reset()

        # Set initial state
        assert self._task_suite is not None
        initial_states = self._task_suite.get_task_init_states(task_id)
        obs = self._env.set_init_state(initial_states[episode_idx])

        # Run dummy action wait steps (always in delta mode to avoid slamming to origin)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        # Switch to absolute action mode after settling (e.g. for X-VLA)
        if self.absolute_action:
            for robot in self._env.robots:
                robot.controller.use_delta = False

        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"Action dimension mismatch: got {len(raw_action)}, expected 7"

        # Discretize gripper
        if raw_action[-1] < 0:
            gripper = -1.0
        else:
            gripper = 1.0
        processed_action = raw_action[:-1] + [gripper]

        assert self._env is not None
        obs, reward, done, info = self._env.step(processed_action)
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        img = preprocess_libero_image(raw_obs["agentview_image"], LIBERO_ENV_RESOLUTION)

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
        }

        if self.send_wrist_image:
            wrist = preprocess_libero_image(raw_obs["robot0_eye_in_hand_image"], LIBERO_ENV_RESOLUTION)
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            # Both sources: observation (default) and controller.
            # Most models (Pi0, OFT, GR00T) use obs; X-VLA uses controller.
            obs_dict["states"] = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    quat_to_axisangle(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            assert self._env is not None
            robot = self._env.robots[0]
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            ee_ori_mat = np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
            ee_aa = quat_to_axisangle(matrix_to_quat(ee_ori_mat))
            obs_dict["controller_states"] = np.concatenate(
                [ee_pos, ee_aa, np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)]
            )

        return obs_dict

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_steps or MAX_STEP_MAPPING.get(self.suite, 300),
            "max_episodes_per_task": 50,  # bounded by initial_states per task
            "suite": self.suite,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_EEF_POS_AA_GRIP
        return spec

    def render(self) -> np.ndarray | None:
        try:
            assert self._env is not None
            return self._env.render()
        except Exception:
            return None
