"""Result collector: aggregates episode results into task and benchmark summaries."""

from __future__ import annotations

import json
import sys
from typing import Any, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


class EpisodeResult(TypedDict):
    task: str
    episode_id: str
    success: bool
    steps: NotRequired[int]
    elapsed_sec: NotRequired[float]
    failure_reason: NotRequired[str | None]
    extra: NotRequired[dict[str, Any]]


class TaskResult(TypedDict):
    task: str
    episodes: list[EpisodeResult]
    success_rate: float
    avg_steps: float


class BenchmarkResult(TypedDict):
    benchmark: str
    mode: str
    harness_version: str
    tasks: list[TaskResult]
    overall_success_rate: float
    config: dict[str, Any]


class ResultCollector:
    """Aggregates episode results into task-level and benchmark-level metrics.

    Records are organized hierarchically: episode → task → benchmark.

    Result file naming (handled by Orchestrator):
        - Non-sharded: ``{name}_{partial|sync}_{unix_timestamp}.json``
        - Sharded: ``{name}_shard{id}of{total}.json``

    The final JSON includes a ``config`` snapshot for reproducibility.
    """

    def __init__(self, benchmark_name: str, mode: str = "sync") -> None:
        self.benchmark_name = benchmark_name
        self.mode = mode
        self._episodes: dict[str, list[EpisodeResult]] = {}  # task -> episodes

    def record(self, task_name: str, episode_result: EpisodeResult) -> None:
        """Record a single episode result."""
        # Normalize numpy booleans to Python bool
        if "success" in episode_result:
            episode_result["success"] = bool(episode_result["success"])
        if task_name not in self._episodes:
            self._episodes[task_name] = []
        self._episodes[task_name].append(episode_result)

    def get_task_result(self, task_name: str) -> TaskResult:
        """Aggregate results for a single task."""
        episodes = self._episodes.get(task_name, [])
        successes = sum(1 for e in episodes if e.get("success") is True)
        total_steps = sum(e.get("steps", 0) for e in episodes)
        n = len(episodes) or 1
        return TaskResult(
            task=task_name,
            episodes=episodes,
            success_rate=successes / n,
            avg_steps=total_steps / n,
        )

    def get_benchmark_result(self, config: dict[str, Any] | None = None) -> BenchmarkResult:
        """Aggregate results for the entire benchmark."""
        tasks = [self.get_task_result(t) for t in self._episodes]
        all_episodes = [e for eps in self._episodes.values() for e in eps]
        total = len(all_episodes) or 1
        successes = sum(1 for e in all_episodes if e.get("success", False))
        from vla_eval import __version__

        return BenchmarkResult(
            benchmark=self.benchmark_name,
            mode=self.mode,
            harness_version=__version__,
            tasks=tasks,
            overall_success_rate=successes / total,
            config=config or {},
        )

    def print_summary(self) -> None:
        """Print a human-readable summary table."""
        result = self.get_benchmark_result()
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {result['benchmark']} (mode: {result['mode']})")
        print(f"{'=' * 60}")
        for task in result["tasks"]:
            n = len(task["episodes"])
            print(f"  {task['task']:40s} {task['success_rate']:6.1%} ({int(task['success_rate'] * n)}/{n})")
        print(f"{'─' * 60}")
        print(f"  {'Overall':40s} {result['overall_success_rate']:6.1%}")
        print(f"{'=' * 60}\n")

    def to_json(self, config: dict[str, Any] | None = None) -> str:
        """Serialize benchmark result to JSON."""
        return json.dumps(self.get_benchmark_result(config), indent=2, default=str)
