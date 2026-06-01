"""EpisodeRunner ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.types import EpisodeResult, Task


class EpisodeRunner(ABC):
    """Abstract base class for episode execution strategies."""

    @abstractmethod
    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a single episode and return the result.

        ``recorder`` (when active) is forwarded to ``benchmark.start_episode``
        so video frames and step rows are captured. The runner also bundles
        ``{sid, eid, eval_id, db_path}`` into the ``EPISODE_START`` WS payload
        so model-server code (e.g. reflex-train) can open its own
        :class:`vla_eval.recording.StepRecorder` against the same SQLite file
        and field-union its inference traces with the benchmark's step rows.
        """
