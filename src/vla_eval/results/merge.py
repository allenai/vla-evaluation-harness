"""Merge shard result files produced by ``--shard-id`` / ``--num-shards`` runs.

Merge behavior:
    - All shards must share the same ``benchmark`` name and ``shard.total``.
    - Missing shards are allowed — the result is marked ``"partial": True``.
    - Duplicate ``episode_id`` across shards: **last file wins** (dict overwrite,
      logged as warning).
    - Success rates are recomputed from the merged episode set.

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
import sys
from pathlib import Path
from typing import Any

from vla_eval import __version__

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
    all_episodes: dict[str, dict[str, dict[str, Any]]] = {}  # task -> {ep_id -> ep}
    for shard in shards:
        shard_id = shard.get("shard", {}).get("id", "?")
        for task_result in shard.get("tasks", []):
            task_name = task_result["task"]
            if task_name not in all_episodes:
                all_episodes[task_name] = {}
            for ep in task_result.get("episodes", []):
                ep_id = ep.get("episode_id", "")
                if ep_id in all_episodes[task_name]:
                    logger.warning(
                        "Duplicate episode_id %r in task %r (shard %s overwrites previous)", ep_id, task_name, shard_id
                    )
                all_episodes[task_name][ep_id] = ep

    # Build merged task results
    tasks = []
    total_episodes = 0
    total_successes = 0
    for task_name in sorted(all_episodes.keys()):
        episodes = list(all_episodes[task_name].values())
        n = len(episodes)
        successes = sum(1 for e in episodes if e.get("success") is True)
        total_steps = sum(e.get("steps", 0) for e in episodes)
        tasks.append(
            {
                "task": task_name,
                "episodes": episodes,
                "success_rate": successes / n if n else 0.0,
                "avg_steps": total_steps / n if n else 0.0,
            }
        )
        total_episodes += n
        total_successes += successes

    is_partial = bool(missing_ids) or any(s.get("partial") for s in shards)

    merged: dict[str, Any] = {
        "benchmark": benchmark_name,
        "mode": shards[0].get("mode", "sync"),
        "harness_version": __version__,
        "tasks": tasks,
        "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "config": shards[0].get("config", {}),
        "merge_info": {
            "num_shards": expected_total,
            "shards_found": found_ids,
            "shards_missing": missing_ids,
            "total_episodes": total_episodes,
        },
    }
    if is_partial:
        merged["partial"] = True

    return merged


def print_merge_report(merged: dict[str, Any]) -> None:
    """Print a human-readable merge report to stderr."""
    info = merged["merge_info"]
    total_shards = info["num_shards"]
    found = info["shards_found"]
    missing = info["shards_missing"]
    total_eps = info["total_episodes"]
    rate = merged["overall_success_rate"]

    out = sys.stderr
    if missing:
        print(f"\n⚠  Missing shards: {missing} (expected 0..{total_shards - 1})", file=out)
        print(f"Coverage: {total_eps} episodes (shards {len(found)}/{total_shards})", file=out)
        print(f"Merged result (PARTIAL): {rate:.1%}", file=out)
        for sid in missing:
            print(f"  To complete: vla-eval run -c <config> --shard-id {sid} --num-shards {total_shards}", file=out)
    else:
        print(f"\nAll {total_shards} shards complete. {total_eps} episodes.", file=out)
        print(f"Overall success rate: {rate:.1%}", file=out)

    # Per-task summary
    print(f"\n{'=' * 60}", file=out)
    print(f"Benchmark: {merged['benchmark']}", file=out)
    print(f"{'=' * 60}", file=out)
    for task in merged["tasks"]:
        n = len(task["episodes"])
        s = int(task["success_rate"] * n)
        print(f"  {task['task']:40s} {task['success_rate']:6.1%} ({s}/{n})", file=out)
    print(f"{'─' * 60}", file=out)
    print(f"  {'Overall':40s} {rate:6.1%}", file=out)
    print(f"{'=' * 60}\n", file=out)
