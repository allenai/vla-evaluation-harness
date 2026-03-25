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
    episode_id: int
    success: bool
    steps: NotRequired[int]
    elapsed_sec: NotRequired[float]
    failure_reason: NotRequired[str | None]


class TaskResult(TypedDict):
    task: str
    episodes: list[EpisodeResult]
    num_episodes: int
    success_rate: float
    avg_steps: float


class BenchmarkResult(TypedDict):
    benchmark: str
    mode: str
    harness_version: str
    created_at: str
    tasks: list[TaskResult]
    overall_success_rate: float
    config: dict[str, Any]
    seed: NotRequired[int | None]


_AGG_FNS: dict[str, Any] = {
    "mean": lambda vals: sum(vals) / len(vals) if vals else 0.0,
    "sum": sum,
    "max": lambda vals: max(vals) if vals else 0.0,
    "min": lambda vals: min(vals) if vals else 0.0,
}


def _aggregate_metrics(result: Any, episodes: Any, metric_keys: dict[str, str]) -> None:
    """Compute custom metric aggregates and insert into *result* dict in-place."""
    for key, agg_type in metric_keys.items():
        values = [e[key] for e in episodes if key in e and isinstance(e[key], (int, float))]
        fn = _AGG_FNS.get(agg_type)
        if fn is not None and values:
            result[f"{agg_type}_{key}"] = round(fn(values), 4)


class ResultCollector:
    """Aggregates episode results into task-level and benchmark-level metrics.

    Records are organized hierarchically: episode → task → benchmark.

    Result file naming (handled by Orchestrator):
        - Non-sharded: ``{name}_{partial|sync}_{unix_timestamp}.json``
        - Sharded: ``{name}_shard{id}of{total}.json``

    The final JSON includes a ``config`` snapshot for reproducibility.
    """

    def __init__(self, benchmark_name: str, mode: str = "sync", metric_keys: dict[str, str] | None = None) -> None:
        self.benchmark_name = benchmark_name
        self.mode = mode
        self.metric_keys = metric_keys or {}
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
        result = TaskResult(
            task=task_name,
            episodes=episodes,
            num_episodes=len(episodes),
            success_rate=successes / n,
            avg_steps=total_steps / n,
        )
        _aggregate_metrics(result, episodes, self.metric_keys)
        return result

    def get_benchmark_result(self, config: dict[str, Any] | None = None) -> BenchmarkResult:
        """Aggregate results for the entire benchmark."""
        from datetime import datetime, timezone

        from vla_eval import __version__

        tasks = [self.get_task_result(t) for t in self._episodes]
        all_episodes = [e for eps in self._episodes.values() for e in eps]
        total = len(all_episodes) or 1
        successes = sum(1 for e in all_episodes if e.get("success", False))

        config = config or {}
        result = BenchmarkResult(
            benchmark=self.benchmark_name,
            mode=self.mode,
            harness_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            tasks=tasks,
            overall_success_rate=successes / total,
            config=config,
        )

        # Promote seed to top level for reproducibility
        seed = config.get("params", {}).get("seed")
        if seed is not None:
            result["seed"] = seed

        # Store metric_keys for merge and add benchmark-level aggregates
        if self.metric_keys:
            result["metric_keys"] = self.metric_keys  # type: ignore[typeddict-unknown-key]
            _aggregate_metrics(result, all_episodes, self.metric_keys)

        return result

    def print_summary(self) -> None:
        """Print a human-readable summary table."""
        from rich.console import Console

        console = Console(highlight=False)
        result = self.get_benchmark_result()
        rate = result["overall_success_rate"]
        rate_color = "green" if rate >= 0.5 else "red"

        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Benchmark: {result['benchmark']}[/bold] (mode: {result['mode']})")
        console.print(f"{'=' * 60}")
        for task in result["tasks"]:
            n = len(task["episodes"])
            tr = task["success_rate"]
            tc = "green" if tr >= 0.5 else "red"
            console.print(f"  {task['task']:40s} [{tc}]{tr:6.1%}[/{tc}] ({int(tr * n)}/{n})")
        console.print(f"{'─' * 60}")
        console.print(f"  {'Overall':40s} [{rate_color}]{rate:6.1%}[/{rate_color}]")
        console.print(f"{'=' * 60}\n")

    def to_json(self, config: dict[str, Any] | None = None) -> str:
        """Serialize benchmark result to JSON."""
        return json.dumps(self.get_benchmark_result(config), indent=2, default=str)
