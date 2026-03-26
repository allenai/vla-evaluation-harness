"""RoboTwin 2.0 benchmark — dual-arm manipulation on SAPIEN/CuRobo.

Ported from the existing ``vla_evaluation_harness`` implementation
shipped in the ``robotwin`` Docker image.

Non-obvious behaviors:
    - **Expert check**: ``get_tasks()`` optionally runs the oracle planner
      per seed to verify solvability (``skip_expert_check=False``).
    - **Lazy init**: Heavy imports happen on first use, not at construction.
    - **14D action**: dual-arm qpos; 16D inputs are trimmed to 14D.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

ROBOTWIN_ROOT = "/app/RoboTwin"


class _EvalGripperPlanner:
    """Minimal planner shim for eval-only RoboTwin startup.

    RoboTwin's qpos evaluation path still calls ``plan_grippers()`` during
    env setup, but it never uses CuRobo path planning afterwards.  This shim
    keeps gripper interpolation working while avoiding the expensive CuRobo
    warmup in ``Robot.set_planner()``.
    """

    def plan_grippers(self, now_val: float, target_val: float) -> dict[str, Any]:
        num_step = 200
        per_step = (target_val - now_val) / num_step
        vals = np.linspace(now_val, target_val, num_step)
        return {"num_step": num_step, "per_step": per_step, "result": vals}

    def update_point_cloud(self, world_pcd: Any, resolution: float = 0.02) -> None:
        return None

    def plan_path(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo path planning during episode execution.")

    def plan_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo batch planning during episode execution.")


def _install_open3d_stub() -> None:
    """Skip importing heavy open3d when the config never requests pointclouds."""
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))


def _install_fast_eval_planner_patch() -> None:
    """Monkeypatch RoboTwin's planner setup to skip CuRobo in qpos eval mode."""
    import envs.robot.robot as robot_mod

    if getattr(robot_mod.Robot, "_vla_eval_fast_patch", False):
        return

    def _set_planner_fast(self: Any, scene: Any = None) -> None:
        self.communication_flag = False
        self.left_planner = _EvalGripperPlanner()
        self.right_planner = _EvalGripperPlanner()

        if self.need_topp:
            self.left_mplib_planner = robot_mod.MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = robot_mod.MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    robot_mod.Robot.set_planner = _set_planner_fast
    robot_mod.Robot._vla_eval_fast_patch = True


def _install_fast_render_patch() -> None:
    """Use SAPIEN's default camera shader instead of ray tracing for faster startup."""
    import sapien.render as sapien_render

    if getattr(sapien_render, "_vla_eval_fast_render_patch", False):
        return

    original_set_camera_shader_dir = sapien_render.set_camera_shader_dir

    def _set_camera_shader_dir_fast(shader_dir: str) -> None:
        original_set_camera_shader_dir("default")

    sapien_render.set_camera_shader_dir = _set_camera_shader_dir_fast
    sapien_render.set_ray_tracing_samples_per_pixel = lambda spp: None
    sapien_render.set_ray_tracing_path_depth = lambda depth: None
    sapien_render.set_ray_tracing_denoiser = lambda name: None
    sapien_render._vla_eval_fast_render_patch = True


class RoboTwinBenchmark(StepBenchmark):
    """RoboTwin dual-arm manipulation benchmark (SAPIEN/CuRobo).

    Args:
        task_name: RoboTwin task (e.g. ``"grab_roller"``).
        task_config: Config name under ``task_config/`` (default ``"demo_clean"``).
        seed: Base seed index.  Starting seed = ``100000 * (1 + seed)``.
        instruction_type: Instruction variant (``"seen"`` or ``"unseen"``).
        test_num: Number of valid episodes to evaluate.
        skip_expert_check: If ``True``, skip oracle planner verification in
            ``get_tasks()`` (useful for quick smoke tests).
        fast_init: If ``True``, skip CuRobo planner warmup for qpos evaluation
            episodes after task discovery. This preserves the eval path used by
            the harness while substantially reducing cold-start time.
        fast_render: If ``True``, use SAPIEN's default camera shader instead of
            RoboTwin's ray-traced renderer. Faster, but observation fidelity may
            differ from the reference benchmark.
    """

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean",
        seed: int = 0,
        instruction_type: str = "seen",
        test_num: int = 100,
        skip_expert_check: bool = False,
        fast_init: bool = True,
        fast_render: bool = False,
    ) -> None:
        import re

        super().__init__()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task_name):
            raise ValueError(f"Invalid task_name: {task_name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_config):
            raise ValueError(f"Invalid task_config: {task_config!r}")
        self.task_name = task_name
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.test_num = test_num
        self.skip_expert_check = skip_expert_check
        self.fast_init = fast_init
        self.fast_render = fast_render
        self._env: Any = None
        self._env_class: Any = None
        self._args: dict[str, Any] | None = None

    # -----------------------------------------------------------------
    # Lazy init
    # -----------------------------------------------------------------

    def _init_robotwin(self) -> None:
        """Add RoboTwin paths, load YAML configs, resolve embodiment."""
        if self._args is not None:
            return

        for p in [ROBOTWIN_ROOT, f"{ROBOTWIN_ROOT}/policy", f"{ROBOTWIN_ROOT}/description/utils"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.chdir(ROBOTWIN_ROOT)
        import yaml

        config_path = os.path.join(
            ROBOTWIN_ROOT,
            "task_config",
            f"{self.task_config}.yml",
        )
        with open(config_path) as f:
            args: dict[str, Any] = yaml.safe_load(f)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config
        if not args.get("data_type", {}).get("pointcloud", False):
            _install_open3d_stub()

        from envs import CONFIGS_PATH

        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
            _embodiment_types = yaml.safe_load(f)

        def _get_file(etype: str) -> str:
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        def _get_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml")) as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_config(args["left_robot_file"])
        args["right_embodiment_config"] = _get_config(args["right_robot_file"])

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
            _camera_config = yaml.safe_load(f)

        hcam = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[hcam]["h"]
        args["head_camera_w"] = _camera_config[hcam]["w"]
        args["eval_mode"] = True

        self._args = args
        envs_module = importlib.import_module(f"envs.{self.task_name}")
        if self.fast_render:
            _install_fast_render_patch()
        self._env_class = getattr(envs_module, self.task_name)
        logger.info("RoboTwin initialised: task=%s", self.task_name)

    def _create_env(self) -> Any:
        assert self._env_class is not None
        return self._env_class()

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close_env(clear_cache=True)
            except Exception:
                pass
            self._env = None

    # -----------------------------------------------------------------
    # StepBenchmark interface
    # -----------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        self._init_robotwin()
        assert self._args is not None
        st_seed = 100000 * (1 + self.seed)

        if self.skip_expert_check:
            if self.fast_init:
                _install_fast_eval_planner_patch()
            return [
                {
                    "name": self.task_name,
                    "suite": "robotwin",
                    "seed": st_seed + i,
                    "episode_idx": i,
                    "instruction": f"Perform the {self.task_name} task.",
                }
                for i in range(self.test_num)
            ]

        # Full expert check — run oracle planner per seed
        from generate_episode_instructions import generate_episode_descriptions

        env = self._create_env()
        tasks: list[Task] = []
        now_seed = st_seed
        episode_idx = 0
        logger.info("Running expert checks from seed %d ...", st_seed)

        while len(tasks) < self.test_num:
            try:
                env.setup_demo(
                    now_ep_num=episode_idx,
                    seed=now_seed,
                    is_test=True,
                    **self._args,
                )
                episode_info = env.play_once()
                env.close_env()
                if env.plan_success and env.check_success():
                    results = generate_episode_descriptions(
                        self.task_name,
                        [episode_info["info"]],
                        self.test_num,
                    )
                    instruction = np.random.choice(
                        results[0][self.instruction_type],
                    )
                    tasks.append(
                        {
                            "name": self.task_name,
                            "suite": "robotwin",
                            "seed": now_seed,
                            "episode_idx": episode_idx,
                            "instruction": instruction,
                        }
                    )
                    episode_idx += 1
            except Exception as e:
                logger.warning("Expert check failed for seed %d: %s", now_seed, e)
                try:
                    env.close_env()
                except Exception:
                    pass
            now_seed += 1
        if self.fast_init:
            _install_fast_eval_planner_patch()
        return tasks

    def reset(self, task: Task) -> Any:
        self._init_robotwin()
        assert self._args is not None
        if self.fast_init:
            _install_fast_eval_planner_patch()

        if self._env is not None:
            try:
                self._env.close_env(clear_cache=True)
            except Exception as e:
                logger.warning("Failed to close previous RoboTwin env: %s", e)
            self._env = None

        self._env = self._create_env()
        self._env.setup_demo(
            now_ep_num=task.get("episode_idx", 0),
            seed=task["seed"],
            is_test=True,
            **self._args,
        )
        self._env.set_instruction(instruction=task["instruction"])
        raw_obs = self._env.get_obs()
        return raw_obs

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        act = np.asarray(raw, dtype=np.float64).flatten()
        if len(act) > 14:
            act = act[:14]
        elif len(act) < 14:
            act = np.pad(act, (0, 14 - len(act)))
        assert act.shape[-1] == 14, f"Action dimension mismatch: got {act.shape[-1]}, expected 14"

        self._env.take_action(act, action_type="qpos")
        raw_obs = self._env.get_obs()
        success = bool(self._env.eval_success)
        done = success or (self._env.take_action_cnt >= self._env.step_lim)
        return StepResult(obs=raw_obs, reward=1.0 if success else 0.0, done=done, info={"success": success})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        return {
            "images": {
                "head_camera": raw_obs["observation"]["head_camera"]["rgb"],
                "left_camera": raw_obs["observation"]["left_camera"]["rgb"],
                "right_camera": raw_obs["observation"]["right_camera"]["rgb"],
            },
            "task_description": raw_obs.get(
                "language",
                task.get("instruction", ""),
            ),
            "joint_state": np.array(raw_obs["joint_action"]["vector"]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": 400,
            "task_name": self.task_name,
            "action_dim": 14,
            "max_episodes_per_task": self.test_num,
        }
