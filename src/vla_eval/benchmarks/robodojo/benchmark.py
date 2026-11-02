"""RoboDojo benchmark adapter (Isaac Lab, dual ARX-X5).

RoboDojo (https://github.com/RoboDojo-Benchmark/RoboDojo, arXiv:2607.04434) evaluates
bimanual manipulation policies on 42 simulation tasks across five capability dimensions.
This adapter drives RoboDojo's native ``EvalEnv`` (scene/reward/reset/observation managers)
directly and leaves model communication to vla-eval, following upstream's eval loop
(``src/eval_client/main.py`` at the pinned commit; line numbers below refer to it):

* A task's layouts ``Assets/Eval_Layout/RoboDojo/<config>/<seed>/<task>_<n>.json`` form a queue in id
  order. Each round (a wave of ``num_envs`` slots) takes the next ``min(num_envs, episodes still to
  count)`` ids, pads the rest with ``None``, closes the used scene and resets onto them (main.py:347-356,
  434-439). The harness hands a wave as many episodes as are still to count, so rounds match.
* Not counted, not retried (``EpisodeSkipped``; the episode runs again on the next ids): layouts whose
  scene does not settle (eval_env.py:247-255), envs a task flags unstable mid-episode (eval_env.py:784-794),
  and every layout of a round that raised (main.py:371-394). PhysX-broken envs (tasks with articulations,
  as upstream's monitor) abandon their layout and the round reruns with refills (main.py:396-424).
* A round's episodes end together and score like ``run_eval()`` (eval_env.py:771-826): ``score`` is 1.0 on
  success, else the task's process score / 100. The queue running dry ends the remaining episodes uncounted.
* Every counted or skipped layout is recorded, so a rerun with the same eval id resumes the queue past them
  (upstream's resume manifest, eval_env.py:674-717, also retries the unstable ones).

A task's episodes stay in one process (``task_exclusive``). Kit holds one EvalEnv, so the next task closes the last
one first; ``close()`` only works on the thread that built the env, so all Kit calls run on one process-wide thread.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np

from vla_eval import columns
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.base import EnvUnusableError, EpisodeSkipped
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Batch, EpisodeResult, Task

logger = logging.getLogger(__name__)

DEFAULT_TASKS = ["stack_blocks"]
# Dual ARX-X5 for all 42 tasks: left arm (6), left gripper (1), right arm (6), right
# gripper (1). The Franka in the three competition tasks is scripted, not policy-driven.
ACTION_DIMS = (6, 1, 6, 1)
ACTION_DIM = sum(ACTION_DIMS)
# Safety cap above the largest per-task ``step_lim`` (fasten_screws: 1900) so
# the harness never truncates an episode before RoboDojo's own limit ends it.
DEFAULT_MAX_STEPS = 2000
# Rounds in a row that raise (unstable scenes aside) after which the env counts as unusable.
MAX_FAILED_RESETS = 5
# Upstream caps ``*_random`` (clutter) tasks at ``sim.scene.clutter_env_limit`` envs (pipeline_utils.py:57-64).
RANDOM_TASK_MAX_ENVS = 5

_T = TypeVar("_T")


class _Kit:
    """Process-wide: one SimulationApp, its PhysX monitor, the thread every Kit call runs on (``close()`` deadlocks
    off the env's creating thread), and the open EvalEnv of whichever instance built it last."""

    app: Any = None
    monitor: Any = None
    thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robodojo-env")
    env: Any = None


class _NoopModelClient:
    """Stand-in for RoboDojo's legacy WebSocket policy client.

    ``EvalEnv`` opens a policy connection during construction and pings it on
    every ``reset()``; vla-eval owns the model connection, so both become no-ops.
    """

    def __init__(self, **_: Any) -> None:
        pass

    def call(self, **_: Any) -> None:
        return None


class RoboDojoBenchmark(VectorStepBenchmark):
    """Run RoboDojo tasks, ``num_envs`` envs per ``EvalEnv``.

    Args:
        num_envs: Envs per round. Upstream runs 1 for policies whose ``deploy.yml`` sets
            ``eval_batch: false``, else ``sim_config.yml``'s 10; ``*_random`` tasks take at most 5.
        root: RoboDojo checkout inside the container.
        tasks: Task module names under ``task/RoboDojo/tasks``.
        env_config: Eval config basename under ``env_cfg`` (default ``arx_x5``).
        camera_names: Optional subset of RoboDojo camera keys to expose.
        send_state: Include concatenated proprioceptive state in observations.
        seed: Published ``Eval_Layout`` group.
        enable_planner: Build RoboDojo's cuRobo planners. Joint-space VLA
            evaluation never uses them (even the Franka competition arm replays
            pkl trajectories), and the public assets ship only ``curobo_tmp.yml``
            templates, so enabling this requires generating ``curobo.yml`` first.
        stream_upstream_videos: Keep RoboDojo's own per-camera ffmpeg episode
            streams (written under ``<root>/eval_result``). Off by default —
            vla-eval's recorder owns video capture.
    """

    partial_reset: ClassVar[bool] = False
    hard_exit: ClassVar[bool] = True  # Kit segfaults in interpreter teardown (omni.graph.core)
    task_exclusive: ClassVar[bool] = True
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        num_envs: int = 1,
        root: str = "/workspace/RoboDojo",
        tasks: list[str] | None = None,
        env_config: str = "arx_x5",
        camera_names: list[str] | None = None,
        send_state: bool = True,
        seed: int = 0,
        enable_planner: bool = False,
        stream_upstream_videos: bool = False,
    ) -> None:
        task_names = list(tasks or DEFAULT_TASKS)
        if num_envs > RANDOM_TASK_MAX_ENVS and any(name.endswith("_random") for name in task_names):
            if not all(name.endswith("_random") for name in task_names):
                raise ValueError("RoboDojo: list *_random tasks apart, they run at most 5 envs")
            logger.warning("RoboDojo: *_random tasks run %d envs, not %d (upstream)", RANDOM_TASK_MAX_ENVS, num_envs)
            num_envs = RANDOM_TASK_MAX_ENVS
        super().__init__(num_envs)
        self._root = Path(root).resolve()
        self._task_names = task_names
        self._env_config = env_config
        self._camera_names = camera_names
        self._send_state = send_state
        self._seed = int(seed)
        self._enable_planner = bool(enable_planner)
        self._stream_upstream_videos = bool(stream_upstream_videos)
        self._env: Any = None
        self._scene_used = False  # reset since the env (re)launched its scene; see _fresh_scene
        self._task_name = ""
        self._eval_num = 0
        self._used: dict[str, set[int]] = {}  # task -> layouts counted or skipped before this process (resume)
        self._queue: list[int] = []  # the task's layouts not taken yet, in id order
        self._rerun: list[int | None] | None = None  # the round to rerun after a PhysX break
        self._failed_resets = 0  # rounds in a row that raised
        # The running round: layout per env, slots, recorders, why a slot does not count, results at its end.
        self._seeds: list[int | None] = []
        self._slots: list[int] = []
        self._recorders: dict[int, EpisodeRecorder] = {}
        self._skips: dict[int, EpisodeSkipped] = {}
        self._results: dict[int, EpisodeResult] = {}
        self._obs: dict[int, Any] = {}  # latest raw observation per slot

    # -----------------------------------------------------------------
    # Isaac app + env construction
    # -----------------------------------------------------------------

    def _launch_app(self) -> None:
        if _Kit.app is not None:
            return
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
        # Isaac Lab swaps the root log handlers for a WARNING-level one at every sim launch; keep the
        # harness's INFO lines (episode progress, reset timing) on the handlers the CLI set up.
        harness_logger, root_handlers = logging.getLogger("vla_eval"), logging.getLogger().handlers
        if root_handlers and not harness_logger.handlers:
            for handler in root_handlers:
                harness_logger.addHandler(handler)
            harness_logger.propagate = False

        # Upstream's PhysX monitor tees stdout/stderr, so it starts before Kit (main.py:111-121); started for every
        # process, so any later task may use it, and only tasks with articulations read it, as upstream.
        from src.eval_client.physx_warning_monitor import get_monitor

        _Kit.monitor = get_monitor()
        _Kit.monitor.start(enabled=True)

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
        _Kit.app = launcher.app

    def _task_config_path(self, task_name: str) -> str:
        if not self._root.is_dir():
            raise FileNotFoundError(f"RoboDojo checkout not found: {self._root}")
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))  # _launch_app orders it for good
        from env.global_configs import BENCHMARK, ROOT_DIR

        task_registry = __import__(f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"])
        return task_registry.task_config_path(str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), task_name)

    def _build_env(self, task_name: str) -> Any:
        import yaml

        task_cfg_path = self._task_config_path(task_name)
        # Upstream monitors PhysX for tasks whose config has articulations (main.py:85-99).
        monitor = bool((yaml.safe_load(Path(task_cfg_path).read_text()) or {}).get("Articulation"))
        self._launch_app()
        from omegaconf import OmegaConf

        from env.global_configs import ENV_CONFIG_PATH

        import src.eval_client.eval_env as eval_env_module
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs

        # Mirror upstream src/eval_client/main.py::main() config assembly.
        config_path = Path(ENV_CONFIG_PATH)
        eval_cfg = load_yaml(str(config_path / f"{self._env_config}.yml"))
        eval_cfg.update(
            {
                "task_name": task_name,
                "num_envs": self.num_envs,
                "device_id": 0,
                "eval_batch": self.num_envs > 1,  # only picks run_eval's policy loop, which the harness replaces
                "policy_name": "vla_eval",
                "additional_info": "vla_eval",
                "seed": self._seed,
                "physx_monitor_enabled": monitor,
            }
        )
        cfg = OmegaConf.create(
            {
                "sim": load_yaml(str(config_path / "sim" / f"{eval_cfg['config']['sim']}.yml")),
                "scene": load_yaml(str(config_path / "scene" / f"{eval_cfg['config']['scene']}.yml")),
                "camera": load_yaml(str(config_path / "camera" / f"{eval_cfg['config']['camera']}.yml")),
                "robot": load_yaml(str(config_path / "robot" / f"{eval_cfg['config']['robot']}.yml")),
                "task_env": load_yaml(task_cfg_path),
                # EvalEnv requires a deploy section during construction; the
                # legacy client it configures is replaced with a no-op below.
                "deploy_cfg": {"port": 1, "host": "127.0.0.1", "policy_name": "vla_eval"},
                "eval_cfg": eval_cfg,
            }
        )
        if resolve_random_task_num_envs(task_name, self.num_envs, cfg.sim) != self.num_envs:
            raise ValueError(f"RoboDojo: {task_name} takes fewer than {self.num_envs} envs upstream")
        OmegaConf.update(cfg, "sim.scene.num_envs", self.num_envs, force_add=True)
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
        cfg.sim.seed = [0] * self.num_envs

        # Keep EvalEnv from opening its legacy policy connection while it is
        # being constructed; the harness connection is authoritative.
        eval_env_module.WsModelClient = _NoopModelClient
        self._check_utils_resolution()
        env = eval_env_module.create_eval_env(cfg, _Kit.app)
        env.model_client = _NoopModelClient()
        # Instance attributes shadow the methods. vla-eval's recorder owns capture: skip EvalEnv's own per-camera
        # ffmpeg streams, but keep the final frame is_episode_end() grabs for an env that just ended.
        stream = env._stream_vision if self._stream_upstream_videos else None

        def stream_vision(env_idx: int, frame: Any) -> None:
            if stream is not None:
                stream(env_idx, frame)
            if env.end_flag[env_idx]:
                self._obs[env_idx] = frame

        env._stream_vision = stream_vision
        if not self._stream_upstream_videos:
            env.save_video = lambda *args, **kwargs: None

        info = env.robot_action_dim_info
        actual_dims = tuple(dim for pair in zip(info["arm_dim"], info["ee_dim"]) for dim in pair)
        if actual_dims != ACTION_DIMS:
            raise RuntimeError(f"RoboDojo embodiment reports action dims {actual_dims}, expected {ACTION_DIMS}")
        used = self._used.get(task_name, set())
        self._queue = [i for i in env.seed_manager.seed_list if i not in used]
        self._eval_num = int(eval_num)
        logger.info(
            "RoboDojo env built: task=%s num_envs=%d step_lim=%s eval_num=%s layout_group=%s layouts left=%d/%d",
            task_name,
            self.num_envs,
            getattr(env, "step_lim", "?"),
            eval_num,
            self._seed,
            len(self._queue),
            len(env.seed_manager.seed_list),
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
    # VectorStepBenchmark interface
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

    def resume(self, results: list[dict[str, Any]]) -> None:
        for r in results:
            metrics = r["metrics"]
            ids = metrics.get("layout_ids", [metrics["layout_id"]] if "layout_id" in metrics else [])
            self._used.setdefault(r["task_name"], set()).update(ids)

    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch:
        return self._on_env_thread_blocking(self._reset, slots, tasks, recorders)

    def step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]:
        return self._on_env_thread_blocking(self._step, slots, actions)

    def result(self, slot: int) -> EpisodeResult:
        if slot not in self._skips and slot not in self._results:  # max_steps cut the round: a failure
            self._on_env_thread_blocking(self._end_round, self._env)
        if slot in self._skips:
            raise self._skips[slot]
        return self._results[slot]

    # -- on the env thread ------------------------------------------------

    def _ensure_env(self, task_name: str) -> Any:
        if self._env is not None and task_name == self._task_name:
            return self._env
        if _Kit.env is not None:  # Kit holds one EvalEnv: close the last task's, as upstream closes between rounds
            started = time.monotonic()
            _Kit.env.close()
            _Kit.env = None
            logger.info(
                "RoboDojo closed %s's env in %.1fs", self._task_name or "another task", time.monotonic() - started
            )
        self._env, self._task_name = None, task_name
        self._scene_used, self._rerun, self._failed_resets = False, None, 0
        self._env = _Kit.env = self._build_env(task_name)
        return self._env

    def _reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch:
        names = {str(task["name"]) for task in tasks}
        if len(names) != 1:
            raise RuntimeError(f"RoboDojo: one round runs one task, got {sorted(names)}")
        env = self._ensure_env(names.pop())
        if max(int(task.get("episode_idx", 0)) for task in tasks) >= self._eval_num:
            raise ValueError(f"RoboDojo: {self._task_name} counts {self._eval_num} episodes; lower episodes_per_task")
        if self._failed_resets >= MAX_FAILED_RESETS:  # their layouts are spent, as upstream's; a fresh process goes on
            raise EnvUnusableError(f"RoboDojo: {self._failed_resets} rounds in a row failed for {self._task_name}")
        self._slots, self._recorders = list(slots), dict(zip(slots, recorders))
        self._skips, self._results, self._obs = {}, {}, {}
        if self._rerun is not None:
            seeds, self._rerun = self._rerun, None
        else:  # upstream's seed_manager.get_seeds (seed_manager.py:62-77)
            taken, self._queue = self._queue[: len(slots)], self._queue[len(slots) :]
            seeds = [None] * self.num_envs
            for slot, layout in zip(sorted(slots), taken):
                seeds[slot] = layout
        real = [s for s in seeds if s is not None]
        if not real:
            raise EpisodeSkipped(f"{self._task_name} has no layout left", retry=False)
        self._seeds = seeds
        started = time.monotonic()
        try:
            self._fresh_scene(env)
            if env.physx_monitor_enabled:
                _Kit.monitor.reset()
            env.reset(seed=seeds)
            # As run_eval() (eval_env.py:771-783): rewards, then the process-score checks (registered before
            # stepping, or they score 0), then the support arm's first move, then the policy loop's first check.
            env.run_reward()
            if hasattr(env, "get_score"):
                env.get_score()
            running = env.get_running_env_idx_list()
            if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
                for env_idx in running:
                    env.query_support_arm_traj(env_idx=env_idx)
            env.is_episode_end()
            raw = env.get_obs_batch(env.get_running_env_idx_list())
        except Exception as exc:
            raise self._round_failed(exc) from exc
        logger.info("RoboDojo round %s reset in %.1fs; running envs %s", seeds, time.monotonic() - started, running)
        # A slot not running (unsettled, or every env done already) still needs an observation.
        self._obs.update({o["env_idx"]: o for o in raw or env.get_obs_batch(slots)})
        for slot in slots:
            if seeds[slot] is None:
                self._skips[slot] = EpisodeSkipped(f"{self._task_name} has no layout left", retry=False)
            elif slot not in running:  # not settled, or invalid (eval_env.py:644-652)
                self._skips[slot] = EpisodeSkipped(
                    f"layout {seeds[slot]} did not settle", metrics={"layout_ids": [seeds[slot]]}
                )
            else:
                self._recorders[slot].record_video(self._extract_frame(self._obs.get(slot)))
        return self._obs_batch(slots)

    def _step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]:
        env = self._env
        for slot in set(self._slots) - set(slots):  # the harness ended its episode (a model error): it stops here
            env.end_flag[slot], env.success[slot] = True, False
        self._slots = list(slots)
        rows = dict(zip(slots, columns.split(actions, len(slots))))
        live = [s for s in slots if not env.end_flag[s]]
        try:
            if live:
                env.take_action_batch([self._unflatten_action(rows[s].get("actions")) for s in live], live)
            env.is_episode_end()
            raw = env.get_obs_batch(env.get_running_env_idx_list())
        except Exception as exc:
            raise self._round_failed(exc) from exc
        self._obs.update({o["env_idx"]: o for o in raw})
        for slot in live:
            success = bool(env.end_flag[slot] and env.success[slot])
            self._recorders[slot].record_video(self._extract_frame(self._obs[slot]))
            self._recorders[slot].record_step(reward=float(success), done=bool(env.end_flag[slot]), success=success)
        done = all(env.end_flag)
        if done:
            self._end_round(env)
        return self._obs_batch(slots), np.full(len(slots), done)

    def _obs_batch(self, slots: list[int]) -> Batch:
        """Each slot's latest observation; one without any shows another slot's (it is skipped anyway)."""
        fallback = next(iter(self._obs.values()))
        return columns.stack([self.make_obs(self._obs.get(s, fallback), {"name": self._task_name}) for s in slots])

    def _end_round(self, env: Any) -> None:
        """Count the round as ``run_eval()`` does (eval_env.py:784-826)."""
        self._failed_resets = 0
        scores = env.reward_manager.get_score() if hasattr(env, "get_score") else None
        for slot in self._slots:
            if slot in self._skips:
                continue
            layout = int(self._seeds[slot])  # type: ignore[arg-type]
            if slot in env.unstable_envs:
                self._skips[slot] = EpisodeSkipped(
                    f"layout {layout} flagged unstable mid-episode", metrics={"layout_ids": [layout]}
                )
                continue
            success = bool(env.end_flag[slot] and env.success[slot])
            score = 1.0 if success else (float(scores[slot]) / 100.0 if scores is not None else 0.0)
            self._results[slot] = {"success": success, "score": score, "layout_id": layout}

    def _round_failed(self, exc: Exception) -> Exception:
        """What a round that raised becomes (main.py:358-424); the caller raises it for every slot."""
        real = [s for s in self._seeds if s is not None]
        name, monitor = type(exc).__name__, _Kit.monitor if self._env.physx_monitor_enabled else None
        if name == "PhysXFatalError" or (monitor is not None and monitor.is_fatal()):
            return EnvUnusableError(f"RoboDojo: PhysX died ({exc}); a fresh process redoes the round")
        if name == "UnStableError":
            self._failed_resets = 0
            return EpisodeSkipped(f"no layout of {real} settled", metrics={"layout_ids": real})
        logger.warning("RoboDojo round %s failed for %s", self._seeds, self._task_name, exc_info=exc)
        broken = getattr(exc, "broken_envs", None) if name == "PhysXBrokenError" else None
        if broken is None:
            broken = monitor.get_broken_envs() if monitor is not None else set()
        broken = {i for i in broken if i < self.num_envs and self._seeds[i] is not None}
        if broken:
            abandoned = [self._seeds[i] for i in sorted(broken)]
            rerun = [None if i in broken else s for i, s in enumerate(self._seeds)]
            refill, self._queue = self._queue[: len(broken)], self._queue[len(broken) :]
            for i, layout in zip(sorted(broken), refill):
                rerun[i] = layout
            self._rerun = rerun if any(s is not None for s in rerun) else None
            return EpisodeSkipped(
                f"PhysX broke envs {sorted(broken)}: layouts {abandoned} abandoned, the round reruns as {rerun}",
                metrics={"layout_ids": abandoned},
            )
        self._failed_resets += 1
        return EpisodeSkipped(f"round {real} failed: {exc!r}", metrics={"layout_ids": real})

    def _fresh_scene(self, env: Any) -> None:
        """Close a used scene so the next ``env.reset()`` relaunches it, as upstream does per round."""
        if not self._scene_used:
            self._scene_used = True
            return
        started = time.monotonic()
        env.close()
        logger.info("RoboDojo closed the used scene in %.1fs", time.monotonic() - started)

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

    def make_obs(self, raw_obs: Any, task: Task) -> dict[str, Any]:
        vision = raw_obs.get("vision") or {}
        camera_names = self._camera_names or list(vision)
        images: dict[str, np.ndarray] = {}
        for name in camera_names:
            camera = vision.get(name)
            if isinstance(camera, dict) and "color" in camera:
                images[name] = self._to_rgb(camera["color"])
        if not images:
            raise RuntimeError(f"RoboDojo observation has no RGB cameras; available: {list(vision)}")
        obs: dict[str, Any] = {
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

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "score": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": DEFAULT_MAX_STEPS,
            "suite": "robodojo",
            "embodiment": self._env_config,
            "action_dim": ACTION_DIM,
            "num_envs": self.num_envs,
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

    def cleanup(self) -> None:
        # Never close the SimulationApp (Kit teardown can hang; the process exits without it); a later task's
        # instance closes the env.
        self._env = None

    def render(self) -> np.ndarray | None:
        if not self._obs:
            return None
        return self._extract_frame(next(iter(self._obs.values())))

    def _on_env_thread_blocking(self, fn: Callable[..., _T], *args: Any) -> _T:
        return _Kit.thread.submit(fn, *args).result()
