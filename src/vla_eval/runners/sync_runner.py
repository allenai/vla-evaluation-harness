"""SyncEpisodeRunner: waits for inference before stepping."""

from __future__ import annotations

import itertools
from typing import Any

from vla_eval import watchdog
from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.base import (
    EnvStartError,
    EnvStepError,
    EpisodeRunner,
    ModelActError,
    end_failed_episode,
    episode_end_payload,
    episode_start_payload,
    phase,
)
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
        5. ``conn.end_episode()`` with the status (also when a step raised)
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
        with phase(EnvStartError):
            await benchmark.start_episode(task, recorder=recorder)
            obs_dict = await benchmark.get_observation()

        await conn.start_episode(episode_start_payload(task, recorder))
        try:
            steps = range(max_steps) if max_steps is not None else itertools.count()
            for step in steps:
                with phase(ModelActError):
                    action = await conn.act(obs_dict)
                with phase(EnvStepError):
                    await benchmark.apply_action(action)
                    watchdog.pet()  # a slow simulator's episode can outlast the stall timeout
                    if await benchmark.is_done():
                        break
                    obs_dict = await benchmark.get_observation()

            with phase(EnvStepError):
                elapsed = await benchmark.get_time()
                metrics = await benchmark.get_result()
        except Exception as exc:
            await end_failed_episode(conn, exc)
            raise
        episode_result: dict = {"metrics": metrics, "steps": step + 1, "elapsed_sec": round(elapsed, 3)}

        await conn.end_episode(episode_end_payload(episode_result))
        return episode_result
