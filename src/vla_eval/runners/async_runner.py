"""AsyncEpisodeRunner: real-time wall-clock evaluation.

Ties the environment clock to wall-clock time.  The environment advances at
a fixed Hz whether or not the model has returned an action.

Pacing is controlled by a :class:`~vla_eval.runners.clock.Clock` instance.
Pass ``Clock(pace=math.inf)`` to run at maximum speed (no sleeping) — useful
for fast simulation where the deployment gap should emerge from actual
inference latency rather than wall-clock pacing.
"""

from __future__ import annotations

import itertools
import logging
import time as _time
from typing import Any

import anyio

import numpy as np

from vla_eval.benchmarks.base import Benchmark
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.base import EpisodeRunner
from vla_eval.runners.clock import Clock
from vla_eval.types import EpisodeResult, Task

logger = logging.getLogger(__name__)


class AsyncEpisodeRunner(EpisodeRunner):
    """Real-time episode runner with clock-based pacing.

    Args:
        hz: Environment step frequency (default 10.0).
        hold_policy: ActionBuffer hold policy (default "repeat_last").
        action_dim: Action dimension for zero hold policy.
        clock: Clock instance for pacing. Defaults to real-time (pace=1.0).
    """

    def __init__(
        self,
        hz: float = 10.0,
        hold_policy: str = "repeat_last",
        action_dim: int = 7,
        clock: Clock | None = None,
        wait_first_action: bool = False,
    ) -> None:
        self.hz = hz
        self.hold_policy = hold_policy
        self.action_dim = action_dim
        self.clock = clock or Clock()
        self.wait_first_action = wait_first_action

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
    ) -> EpisodeResult:
        """Run a single real-time episode.

        The clock resets right before the first observation is sent, so setup
        time (env reset, model server episode_start) is NOT counted.  The step
        loop starts immediately — no waiting for the first action.  Until the
        model server responds, the hold policy (zero or repeat_last) supplies
        actions, faithfully reflecting real deployment where physics does not
        pause for the controller to warm up.
        """
        clock = self.clock

        # --- Setup phase (not timed) ---
        await benchmark.start_episode(task)
        obs_dict = await benchmark.get_observation()

        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        await conn.start_episode({"task": task_info, "mode": "realtime"})

        action_buffer = ActionBuffer(
            hold_policy=self.hold_policy,
            action_dim=self.action_dim,
        )
        conn.on_action(lambda a: action_buffer.update(a))
        await conn.start_listener()

        step_period = 1.0 / self.hz
        step_times: list[float] = []
        step_count = 0

        try:
            # --- Episode begins: clock starts, first obs sent ---
            clock.reset()
            await conn.send_observation(obs_dict)

            # By default, no waiting for first action: the step loop starts
            # immediately.  The model server computes concurrently; until it
            # responds, action_buffer.get() returns a zero/held action — just
            # like real deployment where physics does not pause for inference.
            #
            # When wait_first_action=True, we block until the first action
            # arrives.  This is useful for sanity-checking that the async
            # pipeline matches sync results (eliminates step-0 zero action).
            if self.wait_first_action:
                deadline = _time.monotonic() + 30.0
                while not action_buffer.has_action():
                    if _time.monotonic() > deadline:
                        raise TimeoutError("wait_first_action: no action received within 30s")
                    await anyio.sleep(0.0001)

            steps = range(max_steps) if max_steps is not None else itertools.count()
            for step in steps:
                step_start = clock.time()

                action = action_buffer.get()

                _t0 = _time.monotonic()
                await benchmark.apply_action(action)
                step_times.append(_time.monotonic() - _t0)
                step_count += 1
                if await benchmark.is_done():
                    break
                obs_dict = await benchmark.get_observation()

                # Send next observation
                await conn.send_observation(obs_dict)

                # Pacing via clock
                await clock.wait_until(step_start + step_period)

        finally:
            await conn.stop_listener()

        elapsed = clock.time()
        episode_result = await benchmark.get_result()
        episode_result["steps"] = step_count
        episode_result["elapsed_sec"] = elapsed

        # Real-time metrics
        metrics = action_buffer.get_metrics()
        step_mean = float(np.mean(step_times)) if step_times else 0.0
        step_max = float(np.max(step_times)) if step_times else 0.0
        effective_hz = metrics["update_count"] / elapsed if elapsed > 0 else 0.0
        episode_result["rt_metrics"] = {
            **metrics,
            "effective_control_hz": effective_hz,
            "step_time_mean": step_mean,
            "step_time_max": step_max,
        }

        logger.info(
            "Episode done: %d steps %.1fs | stale=%.0f%% | control=%.1f/%.1fHz | env.step mean=%.1fms max=%.1fms",
            step_count,
            elapsed,
            metrics["stale_action_ratio"] * 100,
            effective_hz,
            self.hz,
            step_mean * 1000,
            step_max * 1000,
        )

        if step_mean > step_period:
            logger.warning(
                "env.step (%.1fms) exceeds step period (%.1fms at %.0fHz) — "
                "simulation cannot keep up with target frequency.",
                step_mean * 1000,
                step_period * 1000,
                self.hz,
            )

        await conn.end_episode(episode_result)
        return episode_result
