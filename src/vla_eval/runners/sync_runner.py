"""SyncEpisodeRunner: waits for inference before stepping."""

from __future__ import annotations

import itertools
from typing import Any

from vla_eval.benchmarks.base import Benchmark
from vla_eval.runners.base import EpisodeRunner
from vla_eval.types import EpisodeResult, Task


class SyncEpisodeRunner(EpisodeRunner):
    """Synchronous episode runner: one observation → one action per step.

    Episode flow:
        1. ``benchmark.start_episode(task)``
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
    ) -> EpisodeResult:
        """Run a synchronous episode."""
        await benchmark.start_episode(task)
        obs_dict = await benchmark.get_observation()

        # Send only serializable task info to the model server
        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        await conn.start_episode({"task": task_info})

        steps = range(max_steps) if max_steps is not None else itertools.count()
        for step in steps:
            action = await conn.act(obs_dict)
            await benchmark.apply_action(action)
            if await benchmark.is_done():
                break
            obs_dict = await benchmark.get_observation()

        elapsed = await benchmark.get_time()
        episode_result = await benchmark.get_result()
        episode_result["steps"] = step + 1
        episode_result["elapsed_sec"] = elapsed

        await conn.end_episode(episode_result)
        return episode_result
