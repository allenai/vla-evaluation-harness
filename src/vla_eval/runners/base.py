"""EpisodeRunner ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vla_eval.benchmarks.base import Benchmark
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
    ) -> EpisodeResult:
        """Run a single episode and return the result."""
