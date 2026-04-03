"""RoboCerebra benchmark — long-horizon manipulation on LIBERO/robosuite."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.rotation import quat_to_axisangle
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

logger = logging.getLogger(__name__)

os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_DUMMY_ACTION = [0.0] * 6 + [-1.0]


class RoboCerebraBenchmark(StepBenchmark):
    """RoboCerebra long-horizon manipulation benchmark.

    Each task is a directory under ``robocerebra_root/<task_type>/<case>/``
    containing a ``.bddl`` file (environment spec), ``demo.hdf5`` (initial
    states), ``task_description.txt``, and ``goal.json`` (success criteria).

    Args:
        robocerebra_root: Path to the downloaded RoboCerebra_Bench data.
        task_types: Which task-type folders to include (default ``["Ideal"]``).
        seed: Random seed for env initialization.
        num_steps_wait: Dummy action steps at episode start.
        send_wrist_image: Include wrist camera in observations.
        send_state: Include proprioceptive state in observations.
    """

    def __init__(
        self,
        robocerebra_root: str = "/workspace/RoboCerebra_Bench",
        task_types: list[str] | None = None,
        seed: int = 7,
        num_steps_wait: int = 15,
        send_wrist_image: bool = False,
        send_state: bool = False,
    ) -> None:
        super().__init__()
        self.robocerebra_root = robocerebra_root
        self.task_types = task_types or ["Ideal"]
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self._env: Any = None
        self._current_goal: dict | None = None
        self._libero_inited = False

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ------------------------------------------------------------------
    def _ensure_libero(self) -> None:
        if self._libero_inited:
            return
        # Trigger all TASK_MAPPING registrations
        import libero.libero.envs  # noqa: F401

        self._libero_inited = True

    # ------------------------------------------------------------------
    def get_tasks(self) -> list[Task]:
        self._ensure_libero()
        root = Path(self.robocerebra_root)
        tasks: list[Task] = []
        for task_type in self.task_types:
            type_dir = root / task_type
            if not type_dir.is_dir():
                continue
            for case_dir in sorted(type_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                bddl_files = list(case_dir.glob("*.bddl"))
                if not bddl_files:
                    continue
                description = f"{task_type}/{case_dir.name}"
                desc_file = case_dir / "task_description.txt"
                if desc_file.exists():
                    for line in desc_file.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("Task:"):
                            description = line.split(":", 1)[1].strip()
                            break
                tasks.append(
                    {
                        "name": description,
                        "task_type": task_type,
                        "case_name": case_dir.name,
                        "task_dir": str(case_dir),
                        "bddl_file": str(bddl_files[0]),
                    }
                )
        return tasks

    # ------------------------------------------------------------------
    def reset(self, task: Task) -> Any:
        import h5py
        import libero.libero.envs.bddl_utils as BDDLUtils
        from libero.libero.envs import TASK_MAPPING
        from robosuite import load_controller_config

        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

        bddl_file = task["bddl_file"]
        task_dir = Path(task["task_dir"])

        problem_info = BDDLUtils.get_problem_info(bddl_file)
        problem_name = problem_info["problem_name"]
        controller_config = load_controller_config(default_controller="OSC_POSE")

        self._env = TASK_MAPPING[problem_name](
            bddl_file_name=bddl_file,
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        self._env.seed(self.seed)
        obs = self._env.reset()

        # Apply initial state from demo.hdf5
        h5_path = task_dir / "demo.hdf5"
        if h5_path.exists():
            with h5py.File(str(h5_path), "r") as h5f:
                init_state = h5f["data"]["demo_1"]["states"][0]
            self._env.sim.set_state_from_flattened(init_state)
            self._env.sim.forward()
            self._env._post_process()
            self._env._update_observables(force=True)
            obs = self._env._get_observations()

        # Load goal for success checking
        goal_path = task_dir / "goal.json"
        if goal_path.exists():
            raw = json.loads(goal_path.read_text())
            # Convert goal.json to monitor_dict format expected by _check_success
            goal: dict[str, list] = {}
            for obj_id, relations in raw.items():
                processed = []
                for item in relations:
                    if isinstance(item, dict) and "state_pair" in item:
                        triple = item["state_pair"]
                    elif isinstance(item, list):
                        triple = item
                    else:
                        continue
                    processed.append([t.lower() if i == 0 else t for i, t in enumerate(triple)])
                goal[obj_id] = processed
            self._current_goal = goal
        else:
            self._current_goal = None

        # Dummy wait steps to let physics settle
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(_DUMMY_ACTION)

        return obs

    # ------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"Action dimension mismatch: got {len(raw_action)}, expected 7"

        gripper = -1.0 if raw_action[-1] < 0 else 1.0
        processed = raw_action[:6] + [gripper]

        obs, reward, done, info = self._env.step(processed)

        # Check success via _check_success if goal is available
        success = False
        if self._current_goal is not None:
            try:
                _, _, all_done = self._env._check_success(self._current_goal)
                success = bool(all_done)
                if success:
                    done = True
            except Exception as e:
                logger.warning("Failed to check success criteria: %s", e)
        info["success"] = success
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        img = raw_obs["agentview_image"]
        img = img[::-1, ::-1].copy()

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
        }

        if self.send_wrist_image:
            wrist = raw_obs["robot0_eye_in_hand_image"]
            wrist = wrist[::-1, ::-1].copy()
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            state = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    quat_to_axisangle(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            obs_dict["states"] = state

        return obs_dict

    # ------------------------------------------------------------------
    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 400}

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
            return self._env.render()
        except Exception:
            return None
