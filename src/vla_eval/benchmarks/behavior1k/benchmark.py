"""BEHAVIOR-1K benchmark implementation (BEHAVIOR Challenge 2026 protocol).

BEHAVIOR-1K is a long-horizon household-activity benchmark built on OmniGibson (NVIDIA Isaac Sim).
The 2026 BEHAVIOR Challenge evaluates 100 tasks (B100) on the R1Pro mobile manipulator.

Protocol version: BEHAVIOR-1K v3.9.4, Isaac Sim 5.1.0, Python 3.11.  Episodes run through upstream's own
``omnigibson.eval.evaluator.BatchedEvaluator`` (one logical env), so scene setup, instance loading, metrics
and timeouts match ``python -m omnigibson.eval.eval``; only the policy transport is the harness WebSocket
protocol instead of upstream's.  The 2025 stack (v3.7.2, B50) is no longer supported.

References:
    - https://behavior.stanford.edu/challenge/evaluation.html
    - https://github.com/StanfordVL/BEHAVIOR-1K (OmniGibson/omnigibson/eval/{eval,evaluator}.py)

Key facts:
    - Robot: R1Pro (default ``omnigibson/eval/r1pro.yaml``), 23-D action:
        base[0:3] (holonomic velocity), torso[3:7], left_arm[7:14], left_gripper[14:15],
        right_arm[15:22], right_gripper[22:23] (absolute joint positions; grippers in [-1, 1]).
    - Cameras: ``env_wrapper`` decides resolution/modalities.  ``DefaultWrapper`` (upstream CLI default) gives
      224x224 RGB; ``RGBDFullResWrapper`` (official RGB-D track) gives 720/480 RGB + ``depth_linear``.
    - Instances: test instance ids 301-340; ``public_test`` indices 0-19 map to 301-320, and the challenge
      reports indices 0-9 with one rollout each.  ``train`` takes raw instance ids (the demo instances).
    - Timeout: 1.5x the task's mean human-demo length unless ``max_steps`` is set.
    - Time budget (v3.9.4): ``max_steps`` seconds of wall clock per rollout, starting once the first policy
      query returns (that query is free; ``server.timeout`` caps it); later queries are capped at 600 s.  A rollout that
      runs over scores zero with a ``failure_reason``, as upstream's ``evaluate_instances_batched``.
    - Metrics: binary ``success`` and the ranking metric ``q_score`` (fraction of goal predicates newly
      satisfied, max over goal options; 1.0 on success), plus upstream's efficiency metrics.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.dirs import ensure_license
from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

# B100 task names and instructions, copied from the 2026-challenge-demos dataset's ``meta/tasks.jsonl``
# (``task_index`` equals the ``Task ID`` column of upstream's ``B100_task_misc.csv``).
_TASKS_FILE = Path(__file__).with_name("tasks.jsonl")
TASK_INSTRUCTIONS: dict[str, str] = {
    row["task_name"]: row["task"] for row in map(json.loads, _TASKS_FILE.read_text().splitlines())
}
B100_TASKS: list[str] = list(TASK_INSTRUCTIONS)

R1PRO_ACTION_DIM = 23
CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
EVAL_MODES = ("train", "public_test", "hidden_test")


class Behavior1KBenchmark(StepBenchmark):
    """BEHAVIOR-1K (OmniGibson) household-activity benchmark.

    Non-obvious behaviors:
        - **Upstream evaluator reuse**: one ``BatchedEvaluator(num_envs=1)`` per task, rebuilt on task change.
          ``reset`` calls its ``load_batch`` (robot reset, TRO instance load, settle, env reset) and ``step`` its
          ``_apply_actions`` (env step, light sync, metric update), so the protocol stays upstream's.
        - **Heavy lazy imports**: ``omnigibson`` boots Isaac Sim, and ``omnigibson.eval.utils.eval_utils`` reads
          the task metadata from ``gm.DATA_PATH`` at import, so imports wait until the first episode.
        - **Time budget**: with one env, upstream's accounted time (policy queries plus simulator steps) is
          the wall clock between the first action and now, so ``step`` measures it there.  A query that never
          returns hits ``server.timeout`` (600 s in the configs) and becomes a harness error, not a zero score.
        - **Threads**: Isaac Sim runs on a worker thread (see ``start_episode``) so the orchestrator's event
          loop survives ``SimulationApp`` start-up.

    Args:
        tasks: Subset of B100 task names.  ``None`` runs all 100.
        mode: Instance split, as upstream ``--mode``: ``public_test``, ``hidden_test`` or ``train``.
        instance_indices: Upstream ``--instance-indices``.  For test splits they index the split (public 0-19);
            for ``train`` they are raw instance ids.  Episode ``i`` uses ``instance_indices[i % len]``.
        max_steps: Episode timeout.  ``None`` keeps upstream's 1.5x mean human-demo length per task.
        env_wrapper: Upstream eval wrapper (``--env-wrapper``); it fixes camera resolution and modalities.
        robot_config: Path (inside the container) to a custom robot YAML (``--robot-config``).
        partial_scene_load: Load only the task's rooms from ``B100_task_misc.csv`` (upstream always does).
        send_proprio: Include the robot proprio vector (``<robot name>::proprio``) as ``states``.
        camera_names: Camera roles to forward.  Defaults to ``head``, ``left_wrist``, ``right_wrist``.
        time_budget: Enforce the challenge's wall-clock budget and per-query timeout (upstream default).
    """

    _ALL_RECORD_FIELDS = frozenset({"done", "terminated", "truncated", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        mode: str = "public_test",
        instance_indices: list[int] | None = None,
        max_steps: int | None = None,
        env_wrapper: str = "omnigibson.eval.wrappers.DefaultWrapper",
        robot_config: str | None = None,
        partial_scene_load: bool = True,
        send_proprio: bool = False,
        camera_names: list[str] | None = None,
        time_budget: bool = True,
    ) -> None:
        super().__init__()
        unknown = [t for t in tasks or [] if t not in TASK_INSTRUCTIONS]
        if unknown:
            raise ValueError(f"Unknown BEHAVIOR-1K tasks: {unknown}")
        if mode not in EVAL_MODES:
            raise ValueError(f"mode must be one of {EVAL_MODES}, got {mode!r}")
        self._task_names: list[str] = list(tasks) if tasks else list(B100_TASKS)
        self._mode = mode
        self._instance_indices = [int(i) for i in instance_indices] if instance_indices else list(range(10))
        self._max_steps = max_steps
        self._env_wrapper = env_wrapper
        self._robot_config = robot_config
        self._partial_scene_load = partial_scene_load
        self._send_proprio = send_proprio
        self._camera_names = list(camera_names or CAMERA_ROLES)
        self._time_budget = time_budget
        unknown_cams = [c for c in self._camera_names if c not in CAMERA_ROLES]
        if unknown_cams:
            raise ValueError(f"Unknown camera roles: {unknown_cams}. Valid: {list(CAMERA_ROLES)}")

        self._evaluator: Any = None
        self._current_task_name: str | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _init_og(self) -> None:
        """First-time asset check + OmniGibson import, on the main thread (OmniGibson installs a SIGINT handler)."""
        if self._initialized:
            return
        from omnigibson.macros import gm

        gm.HEADLESS = True
        # As eval.py: skip the unused 1280x720 viewer render product in headless runs.
        gm.RENDER_VIEWER_CAMERA = False
        self._ensure_assets(Path(gm.DATA_PATH))
        import omnigibson.eval.evaluator  # noqa: F401  (reads B100 metadata from gm.DATA_PATH)

        self._initialized = True

    def _ensure_assets(self, data_path: Path) -> None:
        """Make sure the 2026 scene, robot and task-instance data is at ``data_path``.

        A fresh cache prompts for the BEHAVIOR Dataset ToS and runs setup.sh's three download helpers.
        """
        from omnigibson.utils.asset_utils import (
            BEHAVIOR_1K_ASSET_VERSION,
            get_behavior_1k_assets_version,
            get_key_path,
        )

        installed = get_behavior_1k_assets_version()
        if installed is not None and installed != BEHAVIOR_1K_ASSET_VERSION:
            raise RuntimeError(
                f"{data_path} holds behavior-1k-assets {installed}, but this stack needs {BEHAVIOR_1K_ASSET_VERSION}. "
                "Point the behavior1k asset cache at an empty directory to download the 2026 assets."
            )
        if (data_path / "2026-challenge-task-instances").exists() and installed and Path(get_key_path()).exists():
            return
        ensure_license(
            "behavior-dataset-tos",
            url="https://behavior.stanford.edu/dataset",
            description="BEHAVIOR Dataset ToS (one-time, ~32 GiB download).",
        )
        data_path.mkdir(parents=True, exist_ok=True)
        from omnigibson.utils.asset_utils import (
            download_2026_challenge_task_instances,
            download_behavior_1k_assets,
            download_omnigibson_robot_assets,
        )

        logger.info("Fetching BEHAVIOR-1K 2026 assets into %s", data_path)
        download_omnigibson_robot_assets()
        download_behavior_1k_assets(accept_license=True)
        download_2026_challenge_task_instances()

    def _make_evaluator(self, task_name: str) -> Any:
        """Build upstream's evaluator for *task_name* (also boots Isaac Sim on first use).

        A previous task's scene is dropped with ``og.clear()``: ``env.close()`` is a no-op upstream."""
        # SimulationApp.__init__ calls signal.signal(), which raises off the main thread.  OmniGibson's own
        # handler is already installed by the main-thread import, so the extra registration can be skipped.
        import signal as _signal

        orig_signal = _signal.signal
        setattr(_signal, "signal", lambda *a, **kw: None)
        try:
            import omnigibson as og
            from omegaconf import OmegaConf
            from omnigibson.eval.evaluator import BatchedEvaluator
            from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything

            if self._evaluator is not None:
                self._evaluator = None
                og.clear()
            seed_everything(DEFAULT_EVAL_SEED)
            cfg = OmegaConf.create(
                {
                    "env_wrapper": {"_target_": self._env_wrapper},
                    "policy_name": "local",
                    # Never queried: actions come from the harness.  Required by BatchedEvaluator.__init__.
                    "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
                    "headless": True,
                    "partial_scene_load": self._partial_scene_load,
                    "max_steps": self._max_steps,
                    "write_video": False,
                    "mode": self._mode,
                    "seed": DEFAULT_EVAL_SEED,
                    "num_envs": 1,
                    "task": {"name": task_name},
                    "robot": OmegaConf.load(self._robot_config) if self._robot_config else None,
                }
            )
            return BatchedEvaluator(cfg)
        finally:
            setattr(_signal, "signal", orig_signal)

    @property
    def _state(self) -> Any:
        return self._evaluator.instance_eval_states[0]

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        return [{"name": t, "task_name": t, "suite": "behavior_1k"} for t in self._task_names]

    def reset(self, task: Task) -> Any:
        from omnigibson.eval.evaluator import resolve_instance_ids

        task_name = task["task_name"]
        if self._evaluator is None or self._current_task_name != task_name:
            self._evaluator = self._make_evaluator(task_name)
            self._current_task_name = task_name
        index = self._instance_indices[int(task.get("episode_idx", 0)) % len(self._instance_indices)]
        self._instance_id = resolve_instance_ids(task_name, [index], mode=self._mode)[0]
        self._evaluator.load_batch({0: self._instance_id})
        self._steps = 0
        self._failure: str | None = None
        self._deadline: float | None = None
        obs = self._state.obs
        self._recorder.record_video(self._frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        if self._time_budget and (failure := self._check_budget(time.monotonic())):
            return self._fail(failure)
        import torch as th

        raw = action.get("actions", action.get("action"))
        tensor = th.as_tensor(np.asarray(raw), dtype=th.float32).flatten()
        action_dim = self._state.env_accessor.robot.action_dim
        if tensor.shape[0] != action_dim:
            raise ValueError(f"BEHAVIOR-1K robot expects a {action_dim}-D action, got {tensor.shape[0]}D.")

        terminated, truncated, info = self._evaluator._apply_actions(tensor.unsqueeze(0), [0])
        self._steps += 1
        self._recorder.record_video(self._frame(self._state.obs))
        # As upstream, a step that ends past the deadline fails the rollout even if it terminated.
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return self._fail("group_time_budget_exceeded")
        term, trunc = bool(terminated[0]), bool(truncated[0])
        success = self._state.env_accessor.success
        self._recorder.record_step(done=term or trunc, terminated=term, truncated=trunc, success=success)
        self._obs_ready = time.monotonic()
        return StepResult(obs=self._state.obs, reward=0.0, done=term or trunc, info={"success": success})

    def _check_budget(self, now: float) -> str | None:
        """Failure reason for the query that just returned, as ``evaluate_instances_batched`` would record it."""
        from omnigibson.eval.utils.network_utils import POLICY_RESPONSE_TIMEOUT

        if self._deadline is None:
            # The first query is uncharged and, like EPISODE_START before it, bounded by ``server.timeout``.
            self._deadline = now + self._evaluator.max_steps
            return None
        if now >= self._deadline:
            return "group_time_budget_exceeded"
        if now - self._obs_ready > POLICY_RESPONSE_TIMEOUT:
            return "policy_query_timeout"
        return None

    def _fail(self, reason: str) -> StepResult:
        logger.warning("Rollout failed at step %d for instance %d: %s", self._steps, self._instance_id, reason)
        self._failure = reason
        self._recorder.record_step(done=True, terminated=False, truncated=True, success=False)
        return StepResult(obs=self._state.obs, reward=0.0, done=True, info={"success": False})

    def _frame(self, obs: dict) -> np.ndarray | None:
        names = self._evaluator.robot_camera_names
        for cam in self._camera_names:
            value = obs.get(f"{names[cam]}::rgb") if cam in names else None
            if value is not None:
                return _to_rgb(value)
        return None

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        names = self._evaluator.robot_camera_names
        images: dict[str, np.ndarray] = {}
        depth: dict[str, np.ndarray] = {}
        for cam in self._camera_names:
            if cam not in names:
                continue
            if (rgb := raw_obs.get(f"{names[cam]}::rgb")) is not None:
                images[cam] = _to_rgb(rgb)
            if (d := raw_obs.get(f"{names[cam]}::depth_linear")) is not None:
                depth[cam] = np.asarray(_to_numpy(d), dtype=np.float32)

        out: Observation = {"images": images, "task_description": TASK_INSTRUCTIONS[task["task_name"]]}
        if depth:
            out["depth"] = depth
        if self._send_proprio and (proprio := raw_obs.get(f"{self._evaluator.robot_name}::proprio")) is not None:
            out["states"] = np.asarray(_to_numpy(proprio), dtype=np.float32)
        return out

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        """Upstream's per-rollout record (``BatchedEvaluator.run.record_fn``), flattened to scalars."""
        state = self._state
        record: dict[str, Any] = {}
        if self._failure is None or self._steps > 0:
            for metric in state.metrics:
                record.update(metric.aggregate())
        if self._failure is not None:  # record_fn's zero score for a failed rollout
            import omnigibson as og
            from omnigibson.eval.evaluator import EVAL_TIMEOUT_MULTIPLIER

            zeros = {"base": 0.0, "left": 0.0, "right": 0.0}
            record["q_score"] = {"final": 0.0}
            record.setdefault("agent_distance", dict(zeros))
            record["normalized_agent_distance"] = dict(zeros)
            sim_time = self._steps * og.sim.get_rendering_dt()
            record.setdefault("time", {"simulator_steps": self._steps, "simulator_time": sim_time})
            record["time"]["normalized_time"] = 1 / EVAL_TIMEOUT_MULTIPLIER
        q_score = record.get("q_score", {}).get("final", 0.0)
        result: dict[str, Any] = {
            "success": self._failure is None and bool(state.env_accessor.success),
            "q_score": float(q_score),
            "instance_id": int(self._instance_id),
        }
        if self._failure is not None:
            result["failure_reason"] = self._failure
        for group in ("time", "agent_distance", "normalized_agent_distance"):
            for key, value in record.get(group, {}).items():
                name = key if group == "time" else f"{group}_{key}"
                result[name] = float(value) if math.isfinite(value) else None
        return result

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "q_score": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        # max_steps None leaves the runner unbounded; upstream's per-task timeout ends the episode.
        return {
            "action_dim": R1PRO_ACTION_DIM,
            "max_steps": self._max_steps,
            "robot": "R1Pro",
            "n_tasks": len(self._task_names),
            "mode": self._mode,
        }

    def cleanup(self) -> None:
        # Not calling ``og.shutdown()`` (BatchedEvaluator.__exit__ does): Isaac Sim's shutdown can hang for
        # minutes and block the result write.  Process exit reclaims everything.
        self._evaluator = None

    # Isaac Sim's SimulationApp installs its own event loop when booted from the main thread, which breaks
    # the orchestrator's ``await conn.act(...)``.  Run env construction / stepping on a worker thread.

    async def start_episode(self, task: Task, recorder: EpisodeRecorder | None = None) -> None:
        self._t0 = time.monotonic()
        self._task = task
        self._recorder = recorder or NullEpisodeRecorder()
        self._init_og()
        raw_obs = await _run_in_thread(self.reset, task)
        self._last_result = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        self._last_result = await _run_in_thread(self.step, action)

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"joints": DimSpec("joints", R1PRO_ACTION_DIM, "joint_positions_r1pro")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"language": LANGUAGE}
        for cam in self._camera_names:
            spec[cam] = IMAGE_RGB
        if self._send_proprio:
            spec["state"] = RAW
        return spec


def _to_numpy(value: Any) -> np.ndarray:
    # Tiled (multi-env) sensors return CUDA tensors; single-env ones CPU tensors.
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _to_rgb(value: Any) -> np.ndarray:
    """OmniGibson VisionSensor RGB is (H, W, 4) RGBA uint8; drop alpha."""
    arr = np.asarray(_to_numpy(value), dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)
