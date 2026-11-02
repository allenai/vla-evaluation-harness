"""LiveEpisodeRunner: real-time wall-clock evaluation.

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

import numpy as np

from vla_eval import watchdog
from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.base import (
    EnvStartError,
    EnvStepError,
    EpisodeRunner,
    end_failed_episode,
    episode_end_payload,
    episode_start_payload,
    phase,
    start_model_episode,
)
from vla_eval.runners.clock import Clock
from vla_eval.types import EpisodeResult, Task

logger = logging.getLogger(__name__)


class LiveEpisodeRunner(EpisodeRunner):
    """Real-time episode runner with clock-based pacing.

    Args:
        hz: Environment step frequency (default 10.0).
        clock: Clock instance for pacing. Defaults to real-time (pace=1.0).

    The stale-tick hold action comes from ``benchmark.get_hold_action`` — the
    embodiment owns it (see :class:`~vla_eval.runners.action_buffer.ActionBuffer`).
    """

    def __init__(self, hz: float = 10.0, clock: Clock | None = None) -> None:
        self.hz = hz
        self.clock = clock or Clock()

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a single real-time episode.

        The clock starts once the server's ``EPISODE_READY`` arrived, right before
        the first observation is sent, so setup (env reset, the server's episode
        start) is NOT counted.  The step loop does not wait for the first action:
        until it arrives the benchmark's hold action (``get_hold_action``) is
        applied, as a robot holds its pose while the controller computes.
        """
        clock = self.clock

        # --- Setup phase (not timed) ---
        with phase(EnvStartError):
            await benchmark.start_episode(task, recorder=recorder)
            obs_dict = await benchmark.get_observation()

        await start_model_episode(conn, episode_start_payload(task, recorder, mode="live", control_hz=self.hz))
        try:
            # Stale-tick hold is embodiment-owned; get_hold_action(None) also covers
            # the pre-first-action fallback. Raises if the benchmark hasn't declared it.
            action_buffer = ActionBuffer(hold_fn=benchmark.get_hold_action)
            conn.on_action(lambda a: action_buffer.update(a))
            conn.on_publish(lambda a: action_buffer.update(a))  # latest-wins with replies

            step_period = 1.0 / self.hz
            step_times: list[float] = []
            step_count = 0

            async with conn.listening():
                # --- Episode begins (the server is ready): clock starts, first obs sent ---
                clock.reset()
                await conn.send_observation(obs_dict)

                steps = range(max_steps) if max_steps is not None else itertools.count()
                for step in steps:
                    step_start = clock.time()

                    action = action_buffer.get()
                    watchdog.pet()

                    _t0 = _time.monotonic()
                    with phase(EnvStepError):
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

            elapsed = clock.time()
            with phase(EnvStepError):
                bench_metrics = await benchmark.get_result()
            episode_result: dict = {"metrics": bench_metrics, "steps": step_count, "elapsed_sec": round(elapsed, 3)}
        except Exception as exc:
            await end_failed_episode(conn, exc)
            raise

        # Real-time metrics
        metrics = action_buffer.get_metrics()
        step_mean = float(np.mean(step_times)) if step_times else 0.0
        step_max = float(np.max(step_times)) if step_times else 0.0
        effective_hz = metrics["fresh_count"] / elapsed if elapsed > 0 else 0.0  # new actions applied, not received
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

        await conn.end_episode(episode_end_payload(episode_result))
        return episode_result
