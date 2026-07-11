"""RoboDojo benchmark adapter.

RoboDojo is an Isaac Lab manipulation benchmark.  This adapter deliberately
reuses its native task, scene, reward, and observation managers while leaving
model communication to vla-eval.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

DEFAULT_TASKS = ["stack_blocks"]
DEFAULT_ACTION_DIMS = [6, 1, 6, 1]
_ISAAC_LIFETIME_GUARD: list[Any] = []


class _NoopModelClient:
    """Replace RoboDojo's legacy policy client; vla-eval owns that connection."""

    def __init__(self, **_: Any) -> None:
        pass

    def call(self, **_: Any) -> None:
        return None


class RoboDojoBenchmark(StepBenchmark):
    """Run RoboDojo tasks through the vla-eval step interface.

    Args:
        root: RoboDojo checkout inside the container.
        tasks: Task modules under ``task/RoboDojo/tasks``.
        env_config: YAML basename under ``env_cfg``.
        action_dims: Flattened absolute joint-action layout.  The default is
            dual X5: left arm (6), left gripper (1), right arm (6),
            right gripper (1).
        camera_names: Optional subset of RoboDojo camera keys to expose.
        send_state: Include concatenated proprioceptive state.
        max_steps: Harness-side safety cap; RoboDojo also enforces its task cap.
        seed: Published Eval_Layout group to use. Episodes select numbered
            layouts from that group using ``episode_idx``.
        enable_planner: Build RoboDojo's cuRobo IK/trajectory planners. VLA
            evaluation applies joint actions directly, so this is disabled by
            default to avoid large unused planner caches.
    """

    def __init__(
        self,
        root: str = "/workspace/RoboDojo",
        tasks: list[str] | None = None,
        env_config: str = "arx_x5",
        action_dims: list[int] | None = None,
        camera_names: list[str] | None = None,
        send_state: bool = True,
        max_steps: int = 100,
        seed: int = 0,
        enable_planner: bool = False,
    ) -> None:
        super().__init__()
        self._root = Path(root).resolve()
        self._task_names = list(tasks or DEFAULT_TASKS)
        self._env_config = env_config
        self._action_dims = list(action_dims or DEFAULT_ACTION_DIMS)
        if len(self._action_dims) not in (2, 4):
            raise ValueError(
                "action_dims must describe one arm [arm, gripper] or two arms [arm, gripper, arm, gripper]"
            )
        if any(dim <= 0 for dim in self._action_dims):
            raise ValueError("all action_dims values must be positive")
        self._action_dim = sum(self._action_dims)
        self._camera_names = camera_names
        self._send_state = send_state
        self._max_steps = int(max_steps)
        self._seed = int(seed)
        self._enable_planner = bool(enable_planner)
        self._step_count = 0
        self._env: Any = None
        self._app: Any = None
        self._current_task_name = ""

    def get_tasks(self) -> list[Task]:
        missing = [
            name
            for name in self._task_names
            if not (self._root / "task" / "RoboDojo" / "tasks" / f"{name}.py").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"RoboDojo task modules not found under {self._root}: {missing}")
        return [{"name": name, "suite": "robodojo"} for name in self._task_names]

    def _launch_app(self) -> None:
        if self._app is not None:
            return
        if not self._root.is_dir():
            raise FileNotFoundError(f"RoboDojo checkout not found: {self._root}")
        # Mirror ``scripts/eval_policy.sh``: RoboDojo imports its policy
        # transport as the top-level ``client_server`` package from XPolicyLab.
        # Insert in reverse priority because each path is prepended: RoboDojo
        # itself must win over XPolicyLab for top-level packages such as
        # ``utils`` (matching PYTHONPATH=RoboDojo:RoboDojo/XPolicyLab).
        for import_root in (self._root / "XPolicyLab", self._root):
            path = str(import_root)
            if path not in sys.path:
                sys.path.insert(0, path)
        os.chdir(self._root)

        # AppLauncher must run before modules that import Isaac/Omniverse.
        from isaaclab.app import AppLauncher

        # Match RoboDojo's supported ``scripts/eval_policy.sh`` launcher.
        # These extensions are part of the image; naming them explicitly
        # avoids Kit consulting remote registries during environment startup.
        kit_args = "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
        # This adapter runs Isaac work on a worker thread so SimulationApp
        # cannot replace the orchestrator's asyncio loop. Signal handlers may
        # only be registered on Python's main thread, so suppress Isaac's
        # redundant registration there (the process already owns SIGINT).
        import signal
        import threading

        original_signal = signal.signal
        if threading.current_thread() is not threading.main_thread():
            signal.signal = lambda *args, **kwargs: None
        try:
            launcher = AppLauncher(headless=True, enable_cameras=True, kit_args=kit_args)
        finally:
            signal.signal = original_signal
        self._app = launcher.app

    def _build_env(self, task_name: str) -> Any:
        self._launch_app()
        from omegaconf import OmegaConf

        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        import src.eval_client.eval_env as eval_env_module
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        config_path = Path(ENV_CONFIG_PATH)
        eval_cfg = load_yaml(str(config_path / f"{self._env_config}.yml"))
        eval_cfg.update(
            {
                "task_name": task_name,
                "num_envs": 1,
                "device_id": 0,
                "eval_batch": False,
                "policy_name": "vla_eval",
                "additional_info": "vla_eval",
                "seed": self._seed,
                "eval_num": 1,
            }
        )
        eval_cfg["config_name"] = self._env_config
        sim_cfg = load_yaml(str(config_path / "sim" / f"{eval_cfg['config']['sim']}.yml"))
        sim_cfg["scene"]["num_envs"] = 1
        benchmark_path = Path(ROOT_DIR) / "task" / BENCHMARK
        task_registry = __import__(f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"])
        cfg = OmegaConf.create(
            {
                "sim": sim_cfg,
                "scene": load_yaml(str(config_path / "scene" / f"{eval_cfg['config']['scene']}.yml")),
                "camera": load_yaml(str(config_path / "camera" / f"{eval_cfg['config']['camera']}.yml")),
                "robot": load_yaml(str(config_path / "robot" / f"{eval_cfg['config']['robot']}.yml")),
                "task_env": load_yaml(task_registry.task_config_path(str(benchmark_path / "config"), task_name)),
                # EvalEnv requires this legacy section during construction,
                # but its client is replaced immediately below.
                "deploy_cfg": {"port": 1, "host": "127.0.0.1", "policy_name": "vla_eval"},
                "eval_cfg": eval_cfg,
            }
        )
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=task_name)
        if not self._enable_planner:
            for robot_cfg in cfg.robot.robots:
                robot_cfg["need_planner"] = False
        OmegaConf.update(
            cfg,
            "camera.default_frequency",
            eval_cfg.get("observation", {}).get("collect_freq", 0),
            force_add=True,
        )
        cfg.sim.seed = [0]

        # Distributed asset bundles may contain a generated ``curobo.yml``
        # whose URDF paths still point at the machine that produced it. Keep
        # the bundle read-only and resolve those two fields from the planner
        # file's location at runtime.
        from env.planner_manager.curobo_planner import CuroboPlanner

        if not getattr(CuroboPlanner, "_vla_eval_paths_patched", False):
            original_build_cfg = CuroboPlanner._build_robot_and_scene_cfg

            def _build_portable_cfg(planner: Any, yml_data: dict[str, Any], table_height: float) -> Any:
                data = dict(yml_data)
                robot_cfg = dict(data.get("robot_cfg", {}))
                kinematics = dict(robot_cfg.get("kinematics", {}))
                robot_dir = Path(planner.yml_path).resolve().parent
                urdf_path = kinematics.get("urdf_path")
                if urdf_path:
                    kinematics["urdf_path"] = str(robot_dir / Path(urdf_path).name)
                kinematics["asset_root_path"] = str(robot_dir)
                robot_cfg["kinematics"] = kinematics
                data["robot_cfg"] = robot_cfg
                return original_build_cfg(planner, data, table_height)

            CuroboPlanner._build_robot_and_scene_cfg = _build_portable_cfg
            CuroboPlanner._vla_eval_paths_patched = True

        # Prevent EvalEnv from opening a second, legacy policy connection while
        # it is being constructed; the harness connection is authoritative.
        eval_env_module.WsModelClient = _NoopModelClient
        # RoboDojo's public release uses a wildcard import in layout_manager,
        # but these helpers are absent from its module globals at runtime.
        # Bind them explicitly without modifying the upstream checkout.
        import importlib.util

        import env.scene_manager.layout_manager as layout_manager_module

        load_file_spec = importlib.util.spec_from_file_location(
            "_vla_eval_robodojo_load_file", self._root / "utils" / "load_file.py"
        )
        if load_file_spec is None or load_file_spec.loader is None:
            raise ImportError(f"Cannot load RoboDojo utils/load_file.py from {self._root}")
        load_file_module = importlib.util.module_from_spec(load_file_spec)
        load_file_spec.loader.exec_module(load_file_module)
        layout_manager_module.load_object_metadata = load_file_module.load_object_metadata
        layout_manager_module.load_desc_info = load_file_module.load_desc_info
        env = eval_env_module.create_eval_env(cfg, self._app)
        env.model_client = _NoopModelClient()
        actual_dims = [
            dim
            for pair in zip(env.robot_action_dim_info["arm_dim"], env.robot_action_dim_info["ee_dim"])
            for dim in pair
        ]
        if actual_dims != self._action_dims:
            env.close()
            raise ValueError(
                f"action_dims {self._action_dims} do not match RoboDojo embodiment {eval_cfg['config']['robot']}: {actual_dims}"
            )
        return env

    def reset(self, task: Task) -> Any:
        task_name = str(task["name"])
        if self._env is None or task_name != self._current_task_name:
            if self._env is not None:
                self._env.close()
            self._env = self._build_env(task_name)
            self._current_task_name = task_name
        # ``seed`` selects an Eval_Layout group (e.g. arx_x5/0). Inside that
        # group, RoboDojo addresses deterministic layouts by zero-based index.
        layout_idx = int(task.get("episode_idx", 0))
        self._env.env_seeds = [layout_idx]
        self._env.reset(seed=[layout_idx])
        self._env.run_reward()
        self._step_count = 0
        return self._env.get_obs()

    def _unflatten_action(self, values: np.ndarray) -> dict[str, np.ndarray]:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size != self._action_dim:
            raise ValueError(f"RoboDojo expects {self._action_dim} action values, got {flat.size}")
        cuts = np.cumsum(self._action_dims)[:-1]
        parts = np.split(flat, cuts)
        if len(parts) == 2:
            return {"arm_joint_state": parts[0], "ee_joint_state": parts[1]}
        return {
            "left_arm_joint_state": parts[0],
            "left_ee_joint_state": parts[1],
            "right_arm_joint_state": parts[2],
            "right_ee_joint_state": parts[3],
        }

    def step(self, action: Action) -> StepResult:
        assert self._env is not None
        self._env.take_action(self._unflatten_action(action["actions"]))
        self._step_count += 1
        native_done = bool(self._env.is_episode_end())
        done = native_done or self._step_count >= self._max_steps
        # EvalEnv initializes success=True, so it is authoritative only after
        # its native reward manager terminates the episode.
        success = bool(native_done and self._env.success[0])
        return StepResult(obs=self._env.get_obs(), reward=float(success), done=done, info={"success": success})

    @staticmethod
    def _to_rgb(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[..., :3])

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        vision = raw_obs.get("vision", {})
        camera_names = self._camera_names or list(vision)
        images: dict[str, np.ndarray] = {}
        for name in camera_names:
            camera = vision.get(name)
            if not camera or "color" not in camera:
                continue
            images[name] = self._to_rgb(camera["color"])
        if not images:
            raise RuntimeError(f"RoboDojo observation contains no RGB cameras; available keys: {list(vision)}")
        obs: Observation = {
            "images": images,
            "task_description": str(raw_obs.get("lang") or str(task["name"]).replace("_", " ")),
        }
        if self._send_state and isinstance(raw_obs.get("state"), dict):
            obs["state"] = np.concatenate(
                [np.asarray(value, dtype=np.float32).reshape(-1) for value in raw_obs["state"].values()]
            )
        return obs

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": bool(step_result.info.get("success", False))}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps, "suite": "robodojo", "embodiment": self._env_config}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "actions": DimSpec(
                "actions",
                self._action_dim,
                "absolute_joint_positions_with_grippers",
                description="arm joints followed by continuous [0,1] gripper command for each arm",
            )
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec = {"images": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = STATE_JOINT
        return spec

    def cleanup(self) -> None:
        global _ISAAC_LIFETIME_GUARD

        # Keep the C++ backed objects alive: destroying them explicitly or at
        # normal interpreter teardown can hang or segfault Isaac Sim 5.0.
        _ISAAC_LIFETIME_GUARD.extend(obj for obj in (self._env, self._app) if obj is not None)
        if self._env is not None:
            # RoboDojo/Isaac shutdown can hang in both env.close() and
            # SimulationApp.close(), preventing result serialization. Runs are
            # isolated in a dedicated benchmark container, so process exit is
            # the reliable cleanup boundary.
            self._env = None
        if self._app is not None:
            self._app = None

        # The harness writes its aggregate immediately after cleanup returns.
        # Once that durable result exists, bypass unsafe Isaac C++ teardown.
        # This benchmark runs in its own Docker container, so os._exit is the
        # intended process boundary and Docker reclaims all GPU resources.
        def exit_after_result() -> None:
            results_dir = Path("/workspace/results")
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if any(results_dir.glob("*.json")):
                    os._exit(0)
                time.sleep(0.1)

        threading.Thread(target=exit_after_result, name="robodojo-result-exit", daemon=True).start()

    def render(self) -> np.ndarray | None:
        if self._env is None:
            return None
        raw = self._env.get_obs()
        vision = raw.get("vision", {})
        for camera in vision.values():
            if isinstance(camera, dict) and "color" in camera:
                return self._to_rgb(camera["color"])
        return None

    # Isaac Sim owns an internal asyncio loop during synchronous reset/step.
    # Keep it on a worker thread so it cannot clear the harness loop that
    # transports observations and actions.
    async def start_episode(self, task: Task, recorder: EpisodeRecorder | None = None) -> None:
        self._t0 = time.monotonic()
        self._task = task
        self._recorder = recorder or NullEpisodeRecorder()
        raw_obs = await _run_in_thread(self.reset, task)
        self._last_result = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        self._last_result = await _run_in_thread(self.step, action)
