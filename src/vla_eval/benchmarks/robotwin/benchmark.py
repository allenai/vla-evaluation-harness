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
import json
import logging
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

ROBOTWIN_ROOT = "/app/RoboTwin"
# Expert-check results shipped per task config (gen_expert_seeds.py): the oracle's accepted seeds from 100000 on.
EXPERT_SEEDS_DIR = Path(__file__).parent / "expert_seeds"
BUNDLED_SEED_BASE = 100000


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


class _LazyOpen3D(types.ModuleType):
    """Import open3d only when one of its attributes is first accessed."""

    def __init__(self) -> None:
        super().__init__("open3d")
        self._real_module: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._real_module is not None:
            return self._real_module

        if sys.modules.get("open3d") is self:
            sys.modules.pop("open3d", None)
        try:
            module = importlib.import_module("open3d")
        except Exception:
            sys.modules["open3d"] = self
            raise
        self.__dict__.update(module.__dict__)
        self._real_module = module
        sys.modules["open3d"] = module
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


@contextmanager
def _defer_open3d_import(enabled: bool):
    """Defer open3d import during RoboTwin module import when pointclouds are unused."""
    if not enabled:
        yield
        return

    previous = sys.modules.get("open3d")
    proxy = _LazyOpen3D()
    sys.modules["open3d"] = proxy
    try:
        yield
    finally:
        if sys.modules.get("open3d") is proxy:
            if previous is None:
                sys.modules.pop("open3d", None)
            else:
                sys.modules["open3d"] = previous


def _make_fast_set_planner(robot_mod: Any):
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

    return _set_planner_fast


@contextmanager
def _patched_robot_set_planner(enabled: bool):
    """Temporarily skip CuRobo planner warmup during env setup."""
    if not enabled:
        yield
        return

    import envs.robot.robot as robot_mod

    original = robot_mod.Robot.set_planner
    robot_mod.Robot.set_planner = _make_fast_set_planner(robot_mod)
    try:
        yield
    finally:
        robot_mod.Robot.set_planner = original


@contextmanager
def _patched_render_setup(enabled: bool):
    """Temporarily use SAPIEN's default shader during env setup."""
    if not enabled:
        yield
        return

    import sapien.render as sapien_render

    originals = {
        "set_camera_shader_dir": sapien_render.set_camera_shader_dir,
        "set_ray_tracing_samples_per_pixel": sapien_render.set_ray_tracing_samples_per_pixel,
        "set_ray_tracing_path_depth": sapien_render.set_ray_tracing_path_depth,
        "set_ray_tracing_denoiser": sapien_render.set_ray_tracing_denoiser,
    }

    def _set_camera_shader_dir_fast(shader_dir: str) -> None:
        originals["set_camera_shader_dir"]("default")

    sapien_render.set_camera_shader_dir = _set_camera_shader_dir_fast
    sapien_render.set_ray_tracing_samples_per_pixel = lambda spp: None
    sapien_render.set_ray_tracing_path_depth = lambda depth: None
    sapien_render.set_ray_tracing_denoiser = lambda name: None
    try:
        yield
    finally:
        for name, func in originals.items():
            setattr(sapien_render, name, func)


def _restore_play_once_attributes(env: Any, task_name: str) -> None:
    """Set what these tasks' check_success reads but only play_once assigns (eval_policy.py runs play_once on the same
    env first); play_once derives it from the initial scene, so the same expressions after setup_demo match."""
    if task_name == "open_laptop":
        from envs.utils import ArmTag, get_face_prod

        env.arm_tag = ArmTag("left" if get_face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0]) > 0 else "right")
    elif task_name in ("place_object_scale", "put_object_cabinet"):
        from envs.utils import ArmTag

        env.arm_tag = ArmTag("right" if env.object.get_pose().p[0] > 0 else "left")
        if task_name == "put_object_cabinet":
            env.origin_z = env.object.get_pose().p[2]


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
        use_bundled_expert_seeds: When seeds start at 100000 (``seed=0``, the
            official protocol), take the verified seeds and instructions from
            ``expert_seeds/`` if it covers ``test_num`` and matches the configs and
            task code. ``False`` always verifies at startup.
        fast_init: If ``True``, skip CuRobo planner warmup for qpos evaluation
            episodes after task discovery. This preserves the eval path used by
            the harness while substantially reducing cold-start time.
        fast_render: If ``True``, use SAPIEN's default camera shader instead of
            RoboTwin's ray-traced renderer. Faster, but observation fidelity may
            differ from the reference benchmark.
        render_every: Render the cameras only every this many steps and at the
            episode's end; the other steps' observations have no ``images``, so
            a server that predicts from one fails. Set it to the server's action
            chunk length when the whole chunk runs open loop. Videos then get a
            frame every this many steps.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

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
        use_bundled_expert_seeds: bool = True,
        render_every: int = 1,
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
        self.use_bundled_expert_seeds = use_bundled_expert_seeds
        self.render_every = render_every
        self._steps = 0  # in the current episode
        self._env: Any = None
        self._env_done = False  # the env's last episode ended through done, so setup_demo can reuse it
        self._episodes_since_clear = 0
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

        from envs import CONFIGS_PATH

        embodiment_type = args.get("embodiment")
        if not isinstance(embodiment_type, list) or not all(isinstance(item, str) for item in embodiment_type):
            raise ValueError(f"RoboTwin config {config_path!r} must define embodiment as a list of strings")
        embodiment_type = cast(list[str], embodiment_type)
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
        with _defer_open3d_import(enabled=not args.get("data_type", {}).get("pointcloud", False)):
            envs_module = importlib.import_module(f"envs.{self.task_name}")
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

        bundled = self._bundled_tasks(st_seed)
        if bundled is not None:
            return bundled

        env = self._create_env()
        tasks: list[Task] = []
        now_seed = st_seed
        logger.info("Running expert checks from seed %d ...", st_seed)

        # a broken planner install fails every seed; the worst task (put_object_cabinet, demo_randomized) needs ~21
        max_seeds = 50 * self.test_num
        while len(tasks) < self.test_num:
            if now_seed - st_seed >= max_seeds:
                raise RuntimeError(
                    f"{self.task_name}: {len(tasks)}/{self.test_num} solvable seeds in {max_seeds} tries"
                )
            info = self._expert_check(env, now_seed, len(tasks))
            if info is not None:
                tasks.append(
                    self._task_entry(now_seed, len(tasks), self._instructions(info, now_seed)[self.instruction_type])
                )
            now_seed += 1
        return tasks

    def _task_entry(self, seed: int, episode_idx: int, instruction: str) -> Task:
        return {
            "name": self.task_name,
            "suite": "robotwin",
            "seed": seed,
            "episode_idx": episode_idx,
            "instruction": instruction,
        }

    def _expert_check(self, env: Any, seed: int, episode_idx: int) -> dict[str, Any] | None:
        """Run the oracle on ``seed``; its episode info if the plan and the task succeed, else ``None``."""
        assert self._args is not None
        try:
            env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **self._args)
            episode_info = env.play_once()
            env.close_env()
            return episode_info["info"] if env.plan_success and env.check_success() else None
        except Exception as e:
            logger.warning("Expert check failed for seed %d: %s", seed, e)
            try:
                env.close_env()
            except Exception:
                pass
            return None

    def _instructions(self, episode_info: dict[str, Any], seed: int) -> dict[str, str]:
        """One ``seen`` and one ``unseen`` instruction, drawn with ``seed`` (RoboTwin's own draw is unseeded)."""
        import random

        from generate_episode_instructions import generate_episode_descriptions

        state = random.getstate()
        random.seed(seed)  # generate_episode_descriptions samples with the global ``random``
        try:
            results = generate_episode_descriptions(self.task_name, [episode_info], 100)[0]  # independent of test_num
        finally:
            random.setstate(state)
        rng = np.random.default_rng(seed)
        return {kind: str(rng.choice(results[kind])) for kind in ("seen", "unseen")}

    def fingerprint(self) -> str:
        """What the expert check depends on: the resolved configs, the task's code and the base task's code."""
        import hashlib

        assert self._args is not None
        h = hashlib.sha256(json.dumps(self._args, sort_keys=True, default=str).encode())
        for name in (self.task_name, "_base_task"):
            h.update(Path(ROBOTWIN_ROOT, "envs", f"{name}.py").read_bytes())
        return h.hexdigest()[:16]

    def _bundled_tasks(self, st_seed: int) -> list[Task] | None:
        """Tasks from the shipped expert-check list (``expert_seeds/<task_config>.json``) when it covers this run and
        was built from the same configs and task code; otherwise ``None`` (verify at startup)."""
        path = EXPERT_SEEDS_DIR / f"{self.task_config}.json"
        if not self.use_bundled_expert_seeds or st_seed != BUNDLED_SEED_BASE or not path.is_file():
            return None
        entry = json.loads(path.read_text())["tasks"].get(self.task_name)
        if entry is None or len(entry["episodes"]) < self.test_num:
            return None
        if entry["fingerprint"] != self.fingerprint():
            logger.warning(
                "%s: configs or task code differ from %s; running the expert check", self.task_name, path.name
            )
            return None
        logger.info("Expert check from %s (%d of %d seeds)", path.name, self.test_num, len(entry["episodes"]))
        return [
            self._task_entry(e["seed"], i, e[self.instruction_type])
            for i, e in enumerate(entry["episodes"][: self.test_num])
        ]

    def reset(self, task: Task) -> Any:
        self._init_robotwin()
        assert self._args is not None

        # As eval_policy.py: one env, SAPIEN's asset cache cleared every clear_cache_freq episodes (task configs: 5).
        if self._env is not None:
            clear = not self._env_done or self._episodes_since_clear >= self._args["clear_cache_freq"]
            try:
                self._env.close_env(clear_cache=clear)
            except Exception as e:
                logger.warning("Failed to close previous RoboTwin env: %s", e)
                self._env_done = False
            if clear:
                self._episodes_since_clear = 0
            if not self._env_done:
                self._env = None
        if self._env is None:
            self._env = self._create_env()
        self._env_done = False
        self._episodes_since_clear += 1
        with _patched_robot_set_planner(self.fast_init), _patched_render_setup(self.fast_render):
            self._env.setup_demo(
                now_ep_num=task.get("episode_idx", 0),
                seed=task["seed"],
                is_test=True,
                **self._args,
            )
        _restore_play_once_attributes(self._env, self.task_name)
        self._env.set_instruction(instruction=task["instruction"])
        self._steps = 0
        raw_obs = self._env.get_obs()
        self._recorder.record_video(self._extract_frame(raw_obs))
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
        self._steps += 1
        success = bool(self._env.eval_success)
        done = success or (self._env.take_action_cnt >= self._env.step_lim)
        self._env_done = done
        # get_obs renders three ray-traced cameras, most of a step's time
        raw_obs = self._env.get_obs() if done or self._steps % self.render_every == 0 else None
        self._recorder.record_video(self._extract_frame(raw_obs))
        self._recorder.record_step(reward=1.0 if success else 0.0, done=done, success=success)
        return StepResult(obs=raw_obs, reward=1.0 if success else 0.0, done=done, info={"success": success})

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        try:
            return np.asarray(raw_obs["observation"]["head_camera"]["rgb"])
        except (KeyError, TypeError):
            return None

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        if raw_obs is None:  # a step render_every skipped
            return {"task_description": task.get("instruction", "")}
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
            "max_steps": 1700,  # step() ends each episode at its task's step_lim, at most 1700
            "task_name": self.task_name,
            "action_dim": 14,
            "max_episodes_per_task": self.test_num,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        # 14D dual-arm joint positions
        return {
            "joints": DimSpec("joints", 14, "joint_positions"),
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head_camera": IMAGE_RGB,
            "left_camera": IMAGE_RGB,
            "right_camera": IMAGE_RGB,
            "state": STATE_JOINT,
            "language": LANGUAGE,
        }
