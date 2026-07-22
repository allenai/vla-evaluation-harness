"""SyncEpisodeRunner: waits for inference before stepping."""

from __future__ import annotations

import itertools
from typing import Any

from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.base import EpisodeRunner
from vla_eval.types import EpisodeResult, Task


class SyncEpisodeRunner(EpisodeRunner):
    """Synchronous episode runner: one observation → one action per step.

    Episode flow:
        1. ``benchmark.start_episode(task, recorder=...)``
        2. ``benchmark.get_observation()`` → initial observation.
        3. ``conn.start_episode(task_info)``
        4. Step loop (up to ``max_steps``):
           a. ``conn.act(obs)`` → action from model server
           b. ``benchmark.apply_action(action)``
           c. If ``benchmark.is_done()``: break
           d. ``benchmark.get_observation()`` → next observation
        5. ``conn.end_episode()``
    """

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a synchronous episode."""
        await benchmark.start_episode(task, recorder=recorder)
        obs_dict = await benchmark.get_observation()

        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        ep_payload: dict[str, Any] = {"task": task_info}
        if recorder is not None and recorder.is_active:
            ep_payload["recording"] = {
                "sid": recorder.sid,
                "eid": recorder.eid,
                "eval_id": recorder.eval_id,
                "db_path": recorder.db_path,
            }
        await conn.start_episode(ep_payload)

        steps = range(max_steps) if max_steps is not None else itertools.count()
        step_count = 0
        policy_terminated = False
        for _ in steps:
            action = await conn.act(obs_dict)
            if action.get("terminate_episode") is True:
                policy_terminated = True
                break
            await benchmark.apply_action(action)
            step_count += 1
            if await benchmark.is_done():
                break
            obs_dict = await benchmark.get_observation()

        elapsed = await benchmark.get_time()
        metrics = await benchmark.get_result()
        episode_result: dict = {"metrics": metrics, "steps": step_count, "elapsed_sec": round(elapsed, 3)}
        if policy_terminated:
            episode_result["policy_terminated"] = True

        await conn.end_episode(episode_result)
        return episode_result
