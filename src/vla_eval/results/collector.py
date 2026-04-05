"""Result collector: aggregates episode results into task and benchmark summaries."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


class EpisodeResult(TypedDict):
    """Single episode result.

    ``metrics`` holds benchmark-defined values (e.g. ``success``,
    ``completed_subtasks``) declared via :meth:`Benchmark.get_metric_keys`.
    """

    episode_id: int
    metrics: dict[str, Any]
    steps: NotRequired[int]
    elapsed_sec: NotRequired[float]
    failure_reason: NotRequired[str | None]
    failure_detail: NotRequired[str | None]


class TaskResult(TypedDict):
    """Per-task aggregate.  ``{agg}_{key}`` fields added by :func:`_aggregate_metrics`."""

    task: str
    episodes: list[EpisodeResult]
    num_episodes: int
    num_errors: NotRequired[int]
    avg_steps: float


class BenchmarkResult(TypedDict):
    """Top-level result.  ``{agg}_{key}`` fields added by :func:`_aggregate_metrics`."""

    benchmark: str
    mode: str
    harness_version: str
    created_at: str
    tasks: list[TaskResult]
    config: dict[str, Any]
    seed: NotRequired[int | None]
    metric_keys: NotRequired[dict[str, str]]


_AGG_FNS: dict[str, Any] = {
    "mean": lambda vals: sum(vals) / len(vals) if vals else 0.0,
    "sum": sum,
    "max": lambda vals: max(vals) if vals else 0.0,
    "min": lambda vals: min(vals) if vals else 0.0,
}


def _extract_seed(config: dict[str, Any]) -> int | None:
    """Extract seed from config params, or None."""
    return config.get("params", {}).get("seed")


def _build_task_result(task_name: str, episodes: list, metric_keys: dict[str, str]) -> TaskResult:
    """Build a TaskResult with aggregated metrics from episodes.

    Episodes with a ``failure_reason`` (infra errors like connection loss,
    timeout, etc.) are excluded from metric aggregation so they don't
    inflate the failure rate.  They are still counted in ``num_episodes``
    and reported separately via ``num_errors``.
    """
    completed = [e for e in episodes if not e.get("failure_reason")]
    num_errors = len(episodes) - len(completed)
    total_steps = sum(e.get("steps", 0) for e in completed)
    n = len(completed) or 1
    result = TaskResult(
        task=task_name,
        episodes=episodes,
        num_episodes=len(episodes),
        avg_steps=total_steps / n,
    )
    if num_errors:
        result["num_errors"] = num_errors
    _aggregate_metrics(result, completed, metric_keys)
    return result


def _aggregate_metrics(result: Any, episodes: Any, metric_keys: dict[str, str]) -> None:
    """Compute metric aggregates from ``episode["metrics"]`` and insert into *result*."""
    for key, agg_type in metric_keys.items():
        values = [
            m[key]
            for e in episodes
            if isinstance((m := e.get("metrics", {})), dict) and key in m and isinstance(m[key], (int, float, bool))
        ]
        fn = _AGG_FNS.get(agg_type)
        if fn is not None and values:
            result[f"{agg_type}_{key}"] = round(fn(values), 4)


def print_task_table(console: Any, tasks: list, rate: float, rate_color: str) -> None:
    """Print per-task summary table with error annotations. Shared by collector and merge."""
    total_errors = 0
    for task in tasks:
        n = task["num_episodes"]
        errs = task.get("num_errors", 0)
        total_errors += errs
        evaluated = n - errs
        tr = task.get("mean_success", 0.0)
        tc = "green" if tr >= 0.5 else "red"
        err_tag = f" [yellow]({errs} err)[/yellow]" if errs else ""
        console.print(f"  {task['task']:40s} [{tc}]{tr:6.1%}[/{tc}] ({int(tr * evaluated)}/{evaluated}){err_tag}")
    console.print(f"{'─' * 60}")
    console.print(f"  {'Overall':40s} [{rate_color}]{rate:6.1%}[/{rate_color}]")
    if total_errors:
        console.print(f"  [yellow]{total_errors} episode(s) excluded from metrics (infra errors)[/yellow]")


class ResultCollector:
    """Aggregates episode results into task-level and benchmark-level metrics.

    Records are organized hierarchically: episode → task → benchmark.

    Benchmark-defined metrics live under ``episode["metrics"]`` and are
    aggregated according to ``metric_keys`` (e.g. ``{"success": "mean"}``
    produces ``mean_success`` at task and benchmark level).
    """

    def __init__(self, benchmark_name: str, mode: str = "sync", metric_keys: dict[str, str] | None = None) -> None:
        self.benchmark_name = benchmark_name
        self.mode = mode
        self.metric_keys = metric_keys or {}
        self._episodes: dict[str, list[EpisodeResult]] = {}  # task -> episodes

    def record(self, task_name: str, episode_result: EpisodeResult) -> None:
        """Record a single episode result."""
        # Normalize numpy scalars inside metrics to JSON-serializable Python types
        metrics = episode_result.get("metrics", {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if hasattr(v, "item"):
                    metrics[k] = v.item()
        if task_name not in self._episodes:
            self._episodes[task_name] = []
        self._episodes[task_name].append(episode_result)

    def get_task_result(self, task_name: str) -> TaskResult:
        """Aggregate results for a single task."""
        return _build_task_result(task_name, self._episodes.get(task_name, []), self.metric_keys)

    def get_benchmark_result(self, config: dict[str, Any] | None = None) -> BenchmarkResult:
        """Aggregate results for the entire benchmark."""
        from vla_eval import __version__

        tasks = [self.get_task_result(t) for t in self._episodes]
        all_episodes = [e for eps in self._episodes.values() for e in eps]
        completed = [e for e in all_episodes if not e.get("failure_reason")]

        config = config or {}
        result = BenchmarkResult(
            benchmark=self.benchmark_name,
            mode=self.mode,
            harness_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            tasks=tasks,
            config=config,
        )

        # Promote seed to top level for reproducibility
        seed = _extract_seed(config)
        if seed is not None:
            result["seed"] = seed

        # Store metric_keys and add benchmark-level aggregates (excluding infra errors)
        if self.metric_keys:
            result["metric_keys"] = self.metric_keys
            _aggregate_metrics(result, completed, self.metric_keys)

        return result

    def print_summary(self) -> None:
        """Print a human-readable summary table."""
        from rich.console import Console

        console = Console(highlight=False)
        result = self.get_benchmark_result()
        rate = result.get("mean_success", 0.0)
        rate_color = "green" if rate >= 0.5 else "red"

        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Benchmark: {result['benchmark']}[/bold] (mode: {result['mode']})")
        console.print(f"{'=' * 60}")
        print_task_table(console, result["tasks"], rate, rate_color)
        console.print(f"{'=' * 60}\n")

    def to_json(self, config: dict[str, Any] | None = None) -> str:
        """Serialize benchmark result to JSON."""
        return json.dumps(self.get_benchmark_result(config), indent=2, default=str)
