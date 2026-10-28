"""Batched RoboDojo: one EvalEnv with ``num_envs`` envs, exposed as one benchmark view per env.

The barrier lives in :mod:`vla_eval.benchmarks.batching`; this module holds the RoboDojo
batched reset / step. An unstable env or a failed batch reset swaps the episode onto a
spare layout and raises :class:`RetryEpisode` in that slot, as the single-env path does.
"""

from __future__ import annotations

import logging
from typing import Any

from vla_eval.benchmarks import batching
from vla_eval.benchmarks.base import Benchmark, StepResult
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.base import RetryEpisode
from vla_eval.types import Action, EpisodeResult, Observation, Task

from vla_eval.benchmarks.robodojo.benchmark import MAX_LAYOUT_ATTEMPTS, RoboDojoBenchmark

logger = logging.getLogger(__name__)


class _BatchCore(batching.BatchCore):
    """RoboDojo's batched EvalEnv calls; the env itself stays on the parent benchmark."""

    bench: RoboDojoBenchmark

    def __init__(self, bench: RoboDojoBenchmark, size: int) -> None:
        """Batch of ``size`` slots over the parent benchmark's EvalEnv."""
        super().__init__(bench, size)
        self._recorders: dict[int, EpisodeRecorder] = {}
        self.layout_ids: dict[int, int] = {}

    def make_obs(self, slot: int, raw_obs: Any, task: Task) -> Observation:
        """The slot's observation, built exactly like the single-env one."""
        return self.bench.make_obs(raw_obs, task)

    def get_result(self, slot: int, last: StepResult) -> EpisodeResult:
        """Success, score and layout of the slot's episode."""
        return {
            "success": bool(last.info.get("success", False)),
            "score": float(last.info.get("score", 0.0)),
            "layout_id": self.layout_ids.get(slot),
        }

    # -- batched env calls (worker thread) ----------------------------------

    def _reset_batch(self, payloads: dict[int, tuple[Task, EpisodeRecorder]]) -> tuple[dict[int, Any], set[int]]:
        """Reset the submitted slots on their layouts; an unusable layout retries on a spare."""
        bench = self.bench
        names = {str(task["name"]) for task, _ in payloads.values()}
        if len(names) != 1:
            raise RuntimeError(f"RoboDojo: one batch must share one task, got {sorted(names)}")
        env = bench._ensure_env(names.pop())
        results: dict[int, Any] = {}
        layouts: dict[int, int] = {}
        total = 0
        for slot, (task, recorder) in payloads.items():
            try:
                total, episode_idx = bench._counted_episode(env, task)
            except ValueError as exc:
                results[slot] = exc
                continue
            layouts[slot] = bench._episode_layouts.get(episode_idx, episode_idx)
            self._recorders[slot] = recorder
        valid: list[int] = []
        if layouts:
            # Upstream resets the whole batch; envs without a layout this round get None.
            seeds = [layouts.get(i) for i in range(bench.batch_size())]
            try:
                env.reset(seed=seeds)
                env.run_reward()
                # Register the process-score checks before stepping (see RoboDojoBenchmark._try_layout).
                if hasattr(env, "get_score"):
                    env.get_score()
            except Exception:
                # The failing layout cannot be told apart, so every episode moves to a spare,
                # as the single-env path does for a reset that raises.
                bench._failed_batch_resets += 1
                logger.warning("RoboDojo batch reset failed for layouts %s", seeds, exc_info=True)
            else:
                bench._failed_batch_resets = 0
                unstable = set(getattr(env, "unstable_envs", None) or ())
                valid = [s for s in sorted(layouts) if not env.end_flag[s] and s not in unstable]
            for slot in sorted(layouts.keys() - set(valid)):
                logger.warning("RoboDojo layout %d unusable in env %d", layouts[slot], slot)
                results[slot] = self._layout_failed(payloads[slot][0], total)
            if bench._failed_batch_resets >= MAX_LAYOUT_ATTEMPTS:
                unusable = RuntimeError(
                    f"RoboDojo: {bench._failed_batch_resets} batch resets in a row failed; the environment is unusable"
                )
                results.update(dict.fromkeys(layouts, unusable))
        if valid:
            for slot, raw_obs in zip(valid, env.get_obs_batch(valid)):
                self.layout_ids[slot] = layouts[slot]
                self._recorders[slot].record_video(bench._extract_frame(raw_obs))
                results[slot] = raw_obs
        return results, set(payloads) - set(valid)

    def _layout_failed(self, task: Task, total: int) -> Any:
        """Swap the episode onto its next spare layout; an exception once there is none left."""
        bench = self.bench
        episode_idx = int(task.get("episode_idx", 0))
        failures = bench._episode_failures.get(episode_idx, 0) + 1
        bench._episode_failures[episode_idx] = failures
        if failures >= MAX_LAYOUT_ATTEMPTS:
            return RuntimeError(
                f"RoboDojo: {failures} layouts in a row failed for episode {episode_idx}; the environment is unusable"
            )
        shard_id = int(task.get("shard_id", 0))
        spare = bench._next_spare_layout(total, shard_id, int(task.get("num_shards", 1)))
        if spare is None:
            return RuntimeError(f"RoboDojo: shard {shard_id} exhausted its spare layouts for {task['name']}")
        bench._episode_layouts[episode_idx] = spare
        return RetryEpisode(f"RoboDojo layout for episode {episode_idx} was unusable; retrying")

    def _step_batch(self, payloads: dict[int, Action]) -> tuple[dict[int, StepResult], set[int]]:
        """Step the submitted envs; finished envs drop out of the batch."""
        bench = self.bench
        env = bench._env
        if env is None:
            raise RuntimeError("RoboDojo: step before reset")
        idx = sorted(payloads)
        actions = [bench._unflatten_action(payloads[s].get("actions", payloads[s].get("action"))) for s in idx]
        env.take_action_batch(actions, idx)
        # Same terminator as the single-env path: flips success/end_flag and enforces step_lim per env.
        env.is_episode_end()
        results: dict[int, StepResult] = {}
        finished: set[int] = set()
        for slot, raw_obs in zip(idx, env.get_obs_batch(idx)):
            done = bool(env.end_flag[slot])
            success = bool(done and env.success[slot])
            score = 0.0
            if done:
                score = 1.0 if success else bench._partial_score(env, slot)
                finished.add(slot)
            reward = 1.0 if success else 0.0
            recorder = self._recorders[slot]
            recorder.record_video(bench._extract_frame(raw_obs))
            recorder.record_step(reward=reward, done=done, success=success, score=score)
            results[slot] = StepResult(
                obs=raw_obs, reward=reward, done=done, info={"success": success, "score": score}
            )
        return results, finished


def open_batch(bench: RoboDojoBenchmark, size: int) -> list[Benchmark]:
    """One view per slot over a fresh RoboDojo batch core."""
    return batching.open_views(_BatchCore(bench, size))
