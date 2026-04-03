#!/usr/bin/env python3
"""Pretty-print evaluation results as a table. Accepts one or more JSON paths."""

import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <results.json> [results2.json ...]")
    sys.exit(1)

TASK_COL_W = 72
RESET = "\033[0m"
BOLD = "\033[1m"


def color_for_rate(rate):
    if rate >= 0.9:
        return "\033[92m"  # green
    elif rate >= 0.7:
        return "\033[93m"  # yellow
    return "\033[91m"  # red


def print_table(data):
    """Print a per-task results table for one subset. Returns (success_count, episode_count, rate)."""
    model = data.get("server_info", {}).get("model_server", "Unknown")
    benchmark = data["benchmark"]
    subset = data.get("config", {}).get("params", {}).get("suite", benchmark)
    created = data["created_at"][:19].replace("T", " ")
    seed = data.get("seed", "N/A")
    eps = data["config"].get("episodes_per_task", "?")

    print()
    print(f"  {BOLD}[ {subset} ]{RESET}")
    print(f"  Model: {model}  |  Date: {created}  |  Seed: {seed}  |  Episodes/Task: {eps}")
    print()

    header = (
        f"  {'#':>2}  {'Task':<{TASK_COL_W}}  {'Success':>7}  {'Rate':>6}  {'Avg Steps':>9}  {'Avg Time':>8}"
    )
    sep = "  " + "─" * (len(header) - 2)

    print(sep)
    print(header)
    print(sep)

    total_success = 0
    total_episodes = 0

    for i, task in enumerate(data["tasks"], 1):
        episodes = task["episodes"]
        n = len(episodes)
        successes = sum(1 for e in episodes if e["metrics"]["success"])
        rate = successes / n if n else 0
        avg_steps = sum(e["steps"] for e in episodes) / n if n else 0
        avg_time = sum(e["elapsed_sec"] for e in episodes) / n if n else 0

        task_name = task["task"]
        if len(task_name) > TASK_COL_W:
            task_name = task_name[: TASK_COL_W - 1] + "…"

        c = color_for_rate(rate)
        print(
            f"  {i:>2}  {task_name:<{TASK_COL_W}}  "
            f"{c}{successes:>3}/{n:<3}{RESET}  "
            f"{c}{rate:>5.0%}{RESET}  "
            f"{avg_steps:>9.1f}  "
            f"{avg_time:>7.1f}s"
        )

        total_success += successes
        total_episodes += n

    print(sep)

    overall_rate = total_success / total_episodes if total_episodes else 0
    c = color_for_rate(overall_rate)
    print(
        f"      {'OVERALL':<{TASK_COL_W}}  "
        f"{c}{total_success:>3}/{total_episodes:<3}{RESET}  "
        f"{c}{overall_rate:>5.1%}{RESET}"
    )
    print(sep)

    return subset, total_success, total_episodes, overall_rate


# Load all files
files = sys.argv[1:]
results = []
for path in sorted(files):
    with open(path) as f:
        data = json.load(f)
    results.append((path, data))

# Print each subset table
subset_summaries = []
for path, data in results:
    subset, succ, total, rate = print_table(data)
    subset_summaries.append((subset, succ, total, rate))

# Summary across all subsets
if len(subset_summaries) > 1:
    print()
    print(f"  {BOLD}[ Summary ]{RESET}")
    print()

    name_w = max(len(s[0]) for s in subset_summaries)
    header = f"  {'Subset':<{name_w}}  {'Success':>9}  {'Rate':>6}"
    sep = "  " + "─" * (len(header) - 2)

    print(sep)
    print(header)
    print(sep)

    grand_success = 0
    grand_total = 0

    for subset, succ, total, rate in subset_summaries:
        c = color_for_rate(rate)
        print(f"  {subset:<{name_w}}  {c}{succ:>4}/{total:<4}{RESET}  {c}{rate:>5.1%}{RESET}")
        grand_success += succ
        grand_total += total

    print(sep)

    grand_rate = grand_success / grand_total if grand_total else 0
    avg_rate = sum(s[3] for s in subset_summaries) / len(subset_summaries)
    c_grand = color_for_rate(grand_rate)
    c_avg = color_for_rate(avg_rate)
    print(f"  {'TOTAL':<{name_w}}  {c_grand}{grand_success:>4}/{grand_total:<4}{RESET}  {c_grand}{grand_rate:>5.1%}{RESET}")
    print(f"  {'AVG (per subset)':<{name_w}}  {'':>9}  {c_avg}{avg_rate:>5.1%}{RESET}")
    print(sep)
    print()
