"""Merge shard result files produced by ``--shard-id`` / ``--num-shards`` runs.

Merge behavior:
    - All shards must share the same ``benchmark`` name and ``shard.total``.
    - Missing shards are allowed — the result is marked ``"partial": True``.
    - Duplicate ``episode_id`` across shards: **last file wins** (dict overwrite,
      logged as warning).
    - Metric aggregates are recomputed from the merged episode set.

Expected input format:
    Each shard file is a JSON object with at minimum::

        {
            "benchmark": "...",
            "shard": {"id": 0, "total": 4},
            "tasks": [{"task": "...", "episodes": [...]}]
        }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vla_eval import __version__
from vla_eval.results.collector import _aggregate_metrics, _build_task_result, _extract_seed, print_task_table

logger = logging.getLogger(__name__)


def load_shard_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate shard JSON files."""
    shards = []
    for p in paths:
        data = json.loads(p.read_text())
        if "shard" not in data:
            raise ValueError(f"{p}: not a shard result file (missing 'shard' field)")
        shards.append(data)
    return shards


def merge_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge shard results into a single BenchmarkResult.

    Returns the merged result dict with coverage metadata.
    """
    if not shards:
        raise ValueError("No shard files to merge")

    # Validate consistency
    benchmark_name = shards[0]["benchmark"]
    expected_total = shards[0]["shard"]["total"]
    for s in shards:
        if s["benchmark"] != benchmark_name:
            raise ValueError(f"Benchmark mismatch: {s['benchmark']!r} vs {benchmark_name!r}")
        if s["shard"]["total"] != expected_total:
            raise ValueError(f"Shard total mismatch: {s['shard']['total']} vs {expected_total}")

    # Detect missing/duplicate shards
    found_ids = sorted(s["shard"]["id"] for s in shards)
    expected_ids = list(range(expected_total))
    missing_ids = sorted(set(expected_ids) - set(found_ids))

    from collections import Counter

    id_counts = Counter(found_ids)
    duplicate_ids = [sid for sid, count in id_counts.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate shard IDs found: {sorted(duplicate_ids)}")

    # Merge episodes by task, dedup by episode_id (last-write-wins)
    all_episodes: dict[str, dict[int, dict[str, Any]]] = {}  # task -> {ep_id -> ep}
    for shard in shards:
        shard_id = shard.get("shard", {}).get("id", "?")
        for task_result in shard.get("tasks", []):
            task_name = task_result["task"]
            if task_name not in all_episodes:
                all_episodes[task_name] = {}
            for ep in task_result.get("episodes", []):
                ep_id = ep.get("episode_id", 0)
                if ep_id in all_episodes[task_name]:
                    logger.warning(
                        "Duplicate episode_id %r in task %r (shard %s overwrites previous)", ep_id, task_name, shard_id
                    )
                all_episodes[task_name][ep_id] = ep

    # Build merged task results
    metric_keys: dict[str, str] = shards[0].get("metric_keys", {})
    tasks = []
    all_episodes_flat: list[dict] = []
    for task_name in sorted(all_episodes.keys()):
        episodes = list(all_episodes[task_name].values())
        tasks.append(_build_task_result(task_name, episodes, metric_keys))
        all_episodes_flat.extend(episodes)

    is_partial = bool(missing_ids) or any(s.get("partial") for s in shards)

    config = shards[0].get("config", {})
    merged: dict[str, Any] = {
        "benchmark": benchmark_name,
        "mode": shards[0].get("mode", "sync"),
        "harness_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "config": config,
        "merge_info": {
            "num_shards": expected_total,
            "shards_found": found_ids,
            "shards_missing": missing_ids,
            "total_episodes": len(all_episodes_flat),
        },
    }
    if is_partial:
        merged["partial"] = True

    server_info = shards[0].get("server_info")
    if server_info is not None:
        merged["server_info"] = server_info

    # Preserve original measurement metadata from shards
    merged["shard_harness_version"] = shards[0].get("harness_version")
    shard_dates = sorted(s.get("created_at", "") for s in shards if s.get("created_at"))
    if shard_dates:
        merged["shard_created_at"] = {"first": shard_dates[0], "last": shard_dates[-1]}

    seed = _extract_seed(config)
    if seed is not None:
        merged["seed"] = seed

    if metric_keys:
        merged["metric_keys"] = metric_keys
        _aggregate_metrics(merged, all_episodes_flat, metric_keys)

    return merged


def print_merge_report(merged: dict[str, Any]) -> None:
    """Print a human-readable merge report to stderr."""
    from rich.console import Console

    con = Console(stderr=True, highlight=False)
    info = merged["merge_info"]
    total_shards = info["num_shards"]
    found = info["shards_found"]
    missing = info["shards_missing"]
    total_eps = info["total_episodes"]
    rate = merged.get("mean_success", 0.0)
    rate_color = "green" if rate >= 0.5 else "red"

    if missing:
        con.print(f"\n[yellow]⚠  Missing shards: {missing} (expected 0..{total_shards - 1})[/yellow]")
        con.print(f"Coverage: {total_eps} episodes (shards {len(found)}/{total_shards})")
        con.print(f"Merged result ([yellow]PARTIAL[/yellow]): [{rate_color}]{rate:.1%}[/{rate_color}]")
        for sid in missing:
            con.print(
                f"  To complete: [dim]vla-eval run -c <config> --shard-id {sid} --num-shards {total_shards}[/dim]"
            )
    else:
        con.print(f"\n[green]All {total_shards} shards complete.[/green] {total_eps} episodes.")
        con.print(f"Overall: [{rate_color}]{rate:.1%}[/{rate_color}]")

    # Per-task summary
    con.print(f"\n{'=' * 60}")
    con.print(f"[bold]Benchmark: {merged['benchmark']}[/bold]")
    con.print(f"{'=' * 60}")
    print_task_table(con, merged["tasks"], rate, rate_color)
    con.print(f"{'=' * 60}\n")
