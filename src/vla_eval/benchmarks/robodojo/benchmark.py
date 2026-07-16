"""RoboDojo benchmark adapter (Isaac Lab, dual ARX-X5).

RoboDojo (https://github.com/RoboDojo-Benchmark/RoboDojo, arXiv:2607.04434) evaluates
bimanual manipulation policies on 42 simulation tasks across five capability dimensions.
This adapter drives RoboDojo's native ``EvalEnv`` (scene/reward/reset/observation managers)
directly and leaves model communication to vla-eval, mirroring the episode semantics of
upstream ``EvalEnv.run_eval()``:

* ``reset(seed=[layout_id])`` loads a deterministic published layout from
  ``Assets/Eval_Layout/RoboDojo/<config_name>/<seed>/<task>_<layout_id>.json``,
  then ``run_reward()`` initialises the task's reward state. Layouts are consumed
  in order and broken ones are skipped, exactly like upstream's seed queue — the
  published groups ship 55–65 layouts so 50 counted episodes survive skips; each
  episode result records its ``layout_id``.
* One ``take_action(dict)`` = one policy step; ``is_episode_end()`` computes the
  reward, flips ``success``/``end_flag``, and enforces the task's own ``step_lim``.
  The harness-side ``max_steps`` is only a safety net and must stay above every
  task's ``step_lim`` (largest published value: fasten_screws, 1900).
* Score matches the paper's metric: ``1.0`` on success, else the task's partial
  progress ``reward_manager.get_score()/100`` when the task defines ``get_score``.

Scene-instability and other reset-time failures skip to the next layout (the
official protocol neither counts nor retries them); an episode only errors out
once a task's layout group is exhausted.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

DEFAULT_TASKS = ["stack_blocks"]
# Dual ARX-X5 for all 42 tasks: left arm (6), left gripper (1), right arm (6), right
# gripper (1). The Franka in the three competition tasks is scripted, not policy-driven.
ACTION_DIMS = (6, 1, 6, 1)
ACTION_DIM = sum(ACTION_DIMS)
# Safety cap above the largest per-task ``step_lim`` (fasten_screws: 1900) so
# the harness never truncates an episode before RoboDojo's own limit ends it.
DEFAULT_MAX_STEPS = 2000


class _NoopModelClient:
    """Stand-in for RoboDojo's legacy WebSocket policy client.

    ``EvalEnv`` opens a policy connection during construction and pings it on
    every ``reset()``; vla-eval owns the model connection, so both become no-ops.
    """

    def __init__(self, **_: Any) -> None:
        pass

    def call(self, **_: Any) -> None:
        return None


class RoboDojoBenchmark(StepBenchmark):
    """Run RoboDojo tasks through the vla-eval step interface.

    Args:
        root: RoboDojo checkout inside the container.
        tasks: Task module names under ``task/RoboDojo/tasks``.
        env_config: Eval config basename under ``env_cfg`` (default ``arx_x5``).
        camera_names: Optional subset of RoboDojo camera keys to expose.
        send_state: Include concatenated proprioceptive state in observations.
        seed: Published ``Eval_Layout`` group. Episode ``episode_idx`` selects the
            numbered layout inside that group, exactly like upstream's layout ids.
        enable_planner: Build RoboDojo's cuRobo planners. Joint-space VLA
            evaluation never uses them (even the Franka competition arm replays
            pkl trajectories), and the public assets ship only ``curobo_tmp.yml``
            templates, so enabling this requires generating ``curobo.yml`` first.
        stream_upstream_videos: Keep RoboDojo's own per-camera ffmpeg episode
            streams (written under ``<root>/eval_result``). Off by default —
            vla-eval's recorder owns video capture.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success", "score"})

    def __init__(
        self,
        root: str = "/workspace/RoboDojo",
        tasks: list[str] | None = None,
        env_config: str = "arx_x5",
        camera_names: list[str] | None = None,
        send_state: bool = True,
        seed: int = 0,
        enable_planner: bool = False,
        stream_upstream_videos: bool = False,
    ) -> None:
        super().__init__()
        self._root = Path(root).resolve()
        self._task_names = list(tasks or DEFAULT_TASKS)
        self._env_config = env_config
        self._camera_names = camera_names
        self._send_state = send_state
        self._seed = int(seed)
        self._enable_planner = bool(enable_planner)
        self._stream_upstream_videos = bool(stream_upstream_videos)
        self._env: Any = None
        self._app: Any = None
        self._current_task_name = ""
        self._layout_cursor = 0
        self._current_layout_id = -1
        self._recorder: EpisodeRecorder = NullEpisodeRecorder()

    # -----------------------------------------------------------------
    # Isaac app + env construction
    # -----------------------------------------------------------------

    def _launch_app(self) -> None:
        if self._app is not None:
            return
        if not self._root.is_dir():
            raise FileNotFoundError(f"RoboDojo checkout not found: {self._root}")
        # Both trees ship a top-level ``utils``; RoboDojo's must precede XPolicyLab's (subset
        # load_file). The image bakes PYTHONPATH=<root>, so force-reposition, don't skip-if-present.
        for import_root in (self._root / "XPolicyLab", self._root):
            path = str(import_root)
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        # Upstream writes eval_result/ relative to the checkout; the harness output_dir
        # is already pinned absolute before benchmarks run (cli/main.py), so chdir is safe.
        os.chdir(self._root)

        # AppLauncher must run before any module that imports Isaac/Omniverse.
        from isaaclab.app import AppLauncher

        # Same kit extensions as upstream eval_policy.sh; naming them explicitly
        # keeps Kit from consulting remote registries at startup.
        kit_args = "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"

        # Kit boots on a worker thread (the main thread would lose the harness asyncio loop),
        # and signal handlers register only on the main thread — mute Isaac's SIGINT hook.
        import signal
        import threading

        original_signal = signal.signal
        if threading.current_thread() is not threading.main_thread():
            setattr(signal, "signal", lambda *args, **kwargs: None)
        try:
            launcher = AppLauncher(headless=True, enable_cameras=True, kit_args=kit_args)
        finally:
            setattr(signal, "signal", original_signal)
        self._app = launcher.app

    def _build_env(self, task_name: str) -> Any:
        self._launch_app()
        from omegaconf import OmegaConf

        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        import src.eval_client.eval_env as eval_env_module
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        # Mirror upstream src/eval_client/main.py::main() config assembly.
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
                "physx_monitor_enabled": False,
            }
        )
        task_registry = __import__(f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"])
        benchmark_path = Path(ROOT_DIR) / "task" / BENCHMARK
        cfg = OmegaConf.create(
            {
                "sim": load_yaml(str(config_path / "sim" / f"{eval_cfg['config']['sim']}.yml")),
                "scene": load_yaml(str(config_path / "scene" / f"{eval_cfg['config']['scene']}.yml")),
                "camera": load_yaml(str(config_path / "camera" / f"{eval_cfg['config']['camera']}.yml")),
                "robot": load_yaml(str(config_path / "robot" / f"{eval_cfg['config']['robot']}.yml")),
                "task_env": load_yaml(task_registry.task_config_path(str(benchmark_path / "config"), task_name)),
                # EvalEnv requires a deploy section during construction; the
                # legacy client it configures is replaced with a no-op below.
                "deploy_cfg": {"port": 1, "host": "127.0.0.1", "policy_name": "vla_eval"},
                "eval_cfg": eval_cfg,
            }
        )
        OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
        cfg = process_randomization(cfg)
        cfg, eval_num = process_config(cfg, task_name=task_name)
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

        # Keep EvalEnv from opening its legacy policy connection while it is
        # being constructed; the harness connection is authoritative.
        eval_env_module.WsModelClient = _NoopModelClient
        self._check_utils_resolution()
        env = eval_env_module.create_eval_env(cfg, self._app)
        env.model_client = _NoopModelClient()
        if not self._stream_upstream_videos:
            # vla-eval's recorder owns capture; skip EvalEnv's own per-camera ffmpeg
            # streams (instance attributes shadow the methods).
            env._stream_vision = lambda *args, **kwargs: None
            env.save_video = lambda *args, **kwargs: None

        info = env.robot_action_dim_info
        actual_dims = tuple(dim for pair in zip(info["arm_dim"], info["ee_dim"]) for dim in pair)
        if actual_dims != ACTION_DIMS:
            self._bounded_close(env)
            raise RuntimeError(f"RoboDojo embodiment reports action dims {actual_dims}, expected {ACTION_DIMS}")
        logger.info(
            "RoboDojo env built: task=%s step_lim=%s native_eval_num=%s layout_group=%s",
            task_name,
            getattr(env, "step_lim", "?"),
            eval_num,
            self._seed,
        )
        return env

    def _check_utils_resolution(self) -> None:
        """Fail fast if ``utils.load_file`` resolved to the wrong tree.

        Both RoboDojo and XPolicyLab ship a ``utils`` namespace package;
        XPolicyLab's ``load_file`` is a subset (no load_object_metadata /
        load_desc_info / load_pkl) that produces confusing NameErrors deep in
        layout loading if it wins the sys.path race (see ``_launch_app``).
        """
        import utils.load_file as load_file_module

        expected = (self._root / "utils" / "load_file.py").resolve()
        actual = Path(getattr(load_file_module, "__file__", "")).resolve()
        if actual != expected:
            raise RuntimeError(
                f"utils.load_file resolved to {actual}, expected {expected}; "
                "sys.path ordering is broken (RoboDojo root must precede XPolicyLab)"
            )

    # -----------------------------------------------------------------
    # StepBenchmark interface
    # -----------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        missing = [
            name
            for name in self._task_names
            if not (self._root / "task" / "RoboDojo" / "tasks" / f"{name}.py").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"RoboDojo task modules not found under {self._root}: {missing}")
        return [{"name": name, "suite": "robodojo"} for name in self._task_names]

    def reset(self, task: Task) -> Any:
        task_name = str(task["name"])
        if self._env is not None and task_name != self._current_task_name:
            # Measured: env.close() on a clean stack_blocks scene never returns (>180s), so
            # clear_instance() is never reached and a second env raises "context already exists".
            raise RuntimeError(
                "RoboDojo runs one task per process; split multi-task configs into "
                "one `vla-eval run` per task (see scripts/run_robodojo_protocol.sh)"
            )
        if self._env is None:
            self._env = self._build_env(task_name)
            self._current_task_name = task_name
            self._layout_cursor = 0
        env = self._env
        # Upstream semantics: consume the group's numbered layouts in order and skip
        # ones whose scene fails to build/settle, so every counted episode ran.
        total = len(env.seed_manager.seed_list)
        while self._layout_cursor < total:
            layout_id = self._layout_cursor
            self._layout_cursor += 1
            try:
                env.reset(seed=[layout_id])
                env.run_reward()
                # Upstream registers the task's process-score checks BEFORE stepping; each
                # get_reward() then accumulates progress. Registering late scores 0.
                if hasattr(env, "get_score"):
                    env.get_score()
                raw_obs = env.get_obs()
            except Exception:
                # Never close() here: teardown destroys the cameras and any later
                # reset dies in init_cameras. Just move on to the next layout.
                logger.warning(
                    "RoboDojo layout %d failed for %s; skipping to the next layout",
                    layout_id,
                    task_name,
                    exc_info=True,
                )
                continue
            self._current_layout_id = layout_id
            self._recorder.record_video(self._extract_frame(raw_obs))
            return raw_obs
        raise RuntimeError(f"RoboDojo: exhausted all {total} layouts for {task_name}")

    @staticmethod
    def _unflatten_action(values: Any) -> dict[str, np.ndarray]:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size != ACTION_DIM:
            raise ValueError(f"RoboDojo expects {ACTION_DIM} action values, got {flat.size}")
        parts = np.split(flat, np.cumsum(ACTION_DIMS)[:-1])
        return {
            "left_arm_joint_state": parts[0],
            "left_ee_joint_state": parts[1],
            "right_arm_joint_state": parts[2],
            "right_ee_joint_state": parts[3],
        }

    def step(self, action: Action) -> StepResult:
        assert self._env is not None
        env = self._env
        env.take_action(self._unflatten_action(action.get("actions", action.get("action"))))
        # Upstream's authoritative terminator: computes reward, flips success/end_flag,
        # enforces step_lim, and grabs the final frame of just-ended envs.
        done = bool(env.is_episode_end())
        success = bool(done and env.success[0])
        score = 0.0
        if done:
            score = 1.0 if success else self._partial_score(env)
        raw_obs = env.get_obs()
        self._recorder.record_video(self._extract_frame(raw_obs))
        self._recorder.record_step(reward=1.0 if success else 0.0, done=done, success=success, score=score)
        return StepResult(
            obs=raw_obs,
            reward=1.0 if success else 0.0,
            done=done,
            info={"success": success, "score": score},
        )

    @staticmethod
    def _partial_score(env: Any) -> float:
        """Paper score for failed episodes: task progress in [0, 1].

        Reads the gated score the reward manager accumulated over the episode
        (the checks were registered at reset), exactly like upstream ``run_eval()``.
        Tasks without ``get_score`` score 0 on failure.
        """
        if not hasattr(env, "get_score"):
            return 0.0
        try:
            return float(env.reward_manager.get_score()[0]) / 100.0
        except Exception:
            logger.exception("RoboDojo get_score failed; recording 0.0")
            return 0.0

    @staticmethod
    def _to_rgb(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[..., :3])

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        vision = (raw_obs.get("vision") or {}) if isinstance(raw_obs, dict) else {}
        for camera in vision.values():
            if isinstance(camera, dict) and "color" in camera:
                return self._to_rgb(camera["color"])
        return None

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        vision = raw_obs.get("vision") or {}
        camera_names = self._camera_names or list(vision)
        images: dict[str, np.ndarray] = {}
        for name in camera_names:
            camera = vision.get(name)
            if isinstance(camera, dict) and "color" in camera:
                images[name] = self._to_rgb(camera["color"])
        if not images:
            raise RuntimeError(f"RoboDojo observation has no RGB cameras; available: {list(vision)}")
        obs: Observation = {
            "images": images,
            "task_description": str(raw_obs.get("instruction") or str(task["name"]).replace("_", " ")),
        }
        if self._send_state and isinstance(raw_obs.get("state"), dict):
            obs["state"] = self._pack_state(raw_obs["state"])
        return obs

    @staticmethod
    def _pack_state(state: dict[str, Any]) -> np.ndarray:
        """14-D qpos in XPolicyLab's canonical packing:
        [left arm (6), left gripper (1), right arm (6), right gripper (1)]."""
        canonical = (
            "left_arm_joint_state",
            "left_ee_joint_state",
            "right_arm_joint_state",
            "right_ee_joint_state",
        )
        keys = canonical if all(k in state for k in canonical) else tuple(state)
        return np.concatenate([np.asarray(state[k], dtype=np.float32).reshape(-1) for k in keys])

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {
            "success": bool(step_result.info.get("success", False)),
            "score": float(step_result.info.get("score", 0.0)),
            "layout_id": self._current_layout_id,
        }

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "score": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": DEFAULT_MAX_STEPS,
            "suite": "robodojo",
            "embodiment": self._env_config,
            "action_dim": ACTION_DIM,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "actions": DimSpec(
                "actions",
                ACTION_DIM,
                "absolute_joint_positions_with_grippers",
                description="left arm (6), left gripper (1), right arm (6), right gripper (1)",
            )
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec = {"images": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = STATE_JOINT
        return spec

    @staticmethod
    def _bounded_close(env: Any) -> None:
        import threading

        def _close() -> None:
            try:
                env.close()
            except Exception:
                logger.exception("RoboDojo env close failed")

        # EvalEnv.close() has been measured to hang past 180s even on a clean scene;
        # bound it so callers make progress and the harness can still write results.
        closer = threading.Thread(target=_close, name="robodojo-env-close", daemon=True)
        closer.start()
        closer.join(timeout=60)
        if closer.is_alive():
            logger.warning("RoboDojo env close still hanging after 60s; abandoning it")

    def _close_env(self) -> None:
        if self._env is None:
            return
        env = self._env
        self._env = None
        self._bounded_close(env)

    def cleanup(self) -> None:
        self._close_env()
        # Like behavior1k, never call SimulationApp.close() — Kit teardown can hang;
        # the benchmark container's process exit reclaims the GPU.
        self._app = None

    def render(self) -> np.ndarray | None:
        if self._env is None:
            return None
        return self._extract_frame(self._env.get_obs())

    # Async bridge override (as in behavior1k): keep Isaac on a worker thread so its
    # own asyncio loop cannot tear down the orchestrator's.

    async def start_episode(self, task: Task, recorder: EpisodeRecorder | None = None) -> None:
        self._t0 = time.monotonic()
        self._task = task
        self._recorder = recorder or NullEpisodeRecorder()
        raw_obs = await _run_in_thread(self.reset, task)
        self._last_result = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        self._last_result = await _run_in_thread(self.step, action)
