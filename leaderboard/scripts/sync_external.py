#!/usr/bin/env python3
"""Sync external leaderboard data from RoboChallenge and RoboArena APIs.

Fetches live competition results and merges them into leaderboard.json.
Default is dry-run (prints changes). Use --apply to write.
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import NamedTuple

RESULTS_PATH = Path(__file__).parent.parent / "data" / "leaderboard.json"

ROBOCHALLENGE_API = "https://robochallenge.ai/api/leaderboard/leaderboard_all.json"
ROBOARENA_API = "https://roboarena-api-domain-name.online/api/leaderboard"

# (display_name, is_multi_task_model) → model_key
ROBOCHALLENGE_MODEL_MAP = {
    ("cogact", False): "cogact",
    ("pi0.5", False): "pi0_5_task",
    ("pi05_generalist", True): "pi0_5_multi",
    ("pi0", False): "pi0_task",
    ("pi0_generalist", True): "pi0_multi",
    ("openvla-oft", False): "openvla_oft_rc",
    ("X-VLA", False): "x_vla_rc",
    ("RDT-1B", False): "rdt_1b_rc",
}

# policy → model_key
ROBOARENA_MODEL_MAP = {
    "pi05_droid": "pi0_5_ra",
    "pi0_fast_droid": "pi0_fast_ra",
    "pi0_droid": "pi0_ra",
    "dreaming_zebra": "dreamzero_ra",
}

ROBOARENA_DISPLAY_NAMES = {
    "pi0_5_ra": "\u03c0\u2080.\u2085 (DROID)",
    "pi0_fast_ra": "\u03c0\u2080-FAST (DROID)",
    "pi0_ra": "\u03c0\u2080 (DROID)",
    "dreamzero_ra": "DreamZero",
}


def snake_case(name: str) -> str:
    """Convert display name to snake_case key."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()


def fetch_json(url: str) -> dict | list | None:
    """Fetch JSON from URL, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VLA-Leaderboard/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        print(f"  ERROR fetching {url}: {e}")
        return None


def sync_robochallenge(data: dict) -> list[str]:
    """Fetch RoboChallenge API and upsert into data. Returns change descriptions."""
    changes = []
    entries = fetch_json(ROBOCHALLENGE_API)
    if entries is None:
        print("  Skipping RoboChallenge (API unavailable)")
        return changes

    results = data["results"]
    existing = {(r["model"], r["benchmark"]): i for i, r in enumerate(results)}
    today = date.today().isoformat()

    for entry in entries:
        name = entry["display_name"]
        is_multi = entry["is_multi_task_model"]
        model_key = ROBOCHALLENGE_MODEL_MAP.get((name, is_multi), f"rc_{snake_case(name)}")

        overall = round(entry["success_ratio"] * 100, 1)
        progress = round(entry["score"], 1)

        note_parts = [f"user={entry['user_name']}", f"type={entry['user_type']}"]
        if is_multi:
            note_parts.append("multi-task")
        note = f"RoboChallenge API sync. {', '.join(note_parts)}."

        is_new_model = model_key not in {r["model"] for r in results}
        if is_new_model:
            changes.append(f"  NEW model: {model_key} ({name})")

        pair = (model_key, "robochallenge")
        result_entry = {
            "model": model_key,
            "display_name": name,
            "name_in_paper": name,
            "params": None,
            "model_paper": None,
            "benchmark": "robochallenge",
            "weight_type": "shared",
            "overall_score": overall,
            "suite_scores": {"progress_score": progress},
            "task_scores": {},
            "reported_paper": None,
            "reported_table": None,
            "curated_by": "robochallenge-api",
            "date_added": today,
            "notes": note,
        }

        if pair in existing:
            old = results[existing[pair]]
            old_score = old.get("overall_score")
            result_entry["date_added"] = old.get("date_added", today)
            results[existing[pair]] = result_entry
            changes.append(f"  UPDATE: {model_key}/robochallenge {old_score} -> {overall}")
        else:
            results.append(result_entry)
            existing[pair] = len(results) - 1
            changes.append(f"  NEW: {model_key}/robochallenge score={overall}")

    # Remove stale API-synced entries no longer in the API response
    synced_models = {
        ROBOCHALLENGE_MODEL_MAP.get(
            (e["display_name"], e["is_multi_task_model"]), f"rc_{snake_case(e['display_name'])}"
        )
        for e in entries
    }
    stale = [
        i
        for i, r in enumerate(results)
        if r["benchmark"] == "robochallenge"
        and r.get("curated_by") == "robochallenge-api"
        and r["model"] not in synced_models
    ]
    for i in sorted(stale, reverse=True):
        changes.append(f"  REMOVE stale: {results[i]['model']}/robochallenge")
        results.pop(i)

    return changes


def sync_roboarena(data: dict) -> list[str]:
    """Fetch RoboArena API and upsert into data. Returns change descriptions."""
    changes = []
    resp = fetch_json(ROBOARENA_API)
    if not isinstance(resp, dict):
        print("  Skipping RoboArena (API unavailable)")
        return changes

    board = resp.get("board", [])
    results = data["results"]
    existing = {(r["model"], r["benchmark"]): i for i, r in enumerate(results)}
    today = date.today().isoformat()

    for entry in board:
        policy = entry["policy"]
        model_key = ROBOARENA_MODEL_MAP.get(policy, f"ra_{policy}")

        elo = round(entry["score"], 1)
        num_evals = entry["num_evals"]
        std = round(entry["std"], 1)

        note_parts = [f"num_evals={num_evals}", f"std={std}"]
        if entry["open_source"]:
            note_parts.append("open-source")
        if num_evals < 20:
            note_parts.append("LOW_EVAL_COUNT")
        note = f"RoboArena API sync. {', '.join(note_parts)}."

        display = ROBOARENA_DISPLAY_NAMES.get(model_key, policy.replace("_", " ").title())
        is_new_model = model_key not in {r["model"] for r in results}
        if is_new_model:
            changes.append(f"  NEW model: {model_key} ({display})")

        pair = (model_key, "roboarena")
        result_entry = {
            "model": model_key,
            "display_name": display,
            "name_in_paper": policy,
            "params": None,
            "model_paper": None,
            "benchmark": "roboarena",
            "weight_type": "shared",
            "overall_score": elo,
            "suite_scores": {},
            "task_scores": {},
            "reported_paper": None,
            "reported_table": None,
            "curated_by": "roboarena-api",
            "date_added": today,
            "notes": note,
        }

        if pair in existing:
            old = results[existing[pair]]
            old_score = old.get("overall_score")
            result_entry["date_added"] = old.get("date_added", today)
            results[existing[pair]] = result_entry
            changes.append(f"  UPDATE: {model_key}/roboarena {old_score} -> {elo}")
        else:
            results.append(result_entry)
            existing[pair] = len(results) - 1
            changes.append(f"  NEW: {model_key}/roboarena elo={elo}")

    # Remove stale API-synced entries no longer in the API response
    synced_models = {ROBOARENA_MODEL_MAP.get(e["policy"], f"ra_{e['policy']}") for e in board}
    stale = [
        i
        for i, r in enumerate(results)
        if r["benchmark"] == "roboarena" and r.get("curated_by") == "roboarena-api" and r["model"] not in synced_models
    ]
    for i in sorted(stale, reverse=True):
        changes.append(f"  REMOVE stale: {results[i]['model']}/roboarena")
        results.pop(i)

    return changes


# ---------------------------------------------------------------------------
# Source registry — single source of truth for available external leaderboards.
# Adding a new source: define a sync_*() function above, then add an entry here.
# CLI --source choices, PR title/body, and --list-sources all derive from this.
# ---------------------------------------------------------------------------


class Source(NamedTuple):
    display_name: str
    url: str
    sync: Callable[[dict], list[str]]


SOURCES: dict[str, Source] = {
    "robochallenge": Source("RoboChallenge", "https://robochallenge.ai/leaderboard", sync_robochallenge),
    "roboarena": Source("RoboArena", "https://robo-arena.github.io/", sync_roboarena),
}


def _set_github_output(synced: list[str], num_changes: int) -> None:
    """Write sync summary to $GITHUB_OUTPUT for the CI workflow."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    lines = []
    for key in synced:
        src = SOURCES[key]
        lines.append(f"- **{src.display_name}**: {src.url}")
    lines.append(f"- {num_changes} change(s)")
    with open(output_path, "a") as f:
        f.write("sync_summary<<GITHUB_OUTPUT_EOF\n")
        f.write("\n".join(lines) + "\n")
        f.write("GITHUB_OUTPUT_EOF\n")


def main():
    parser = argparse.ArgumentParser(description="Sync external leaderboard APIs into leaderboard.json.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    parser.add_argument("--source", choices=SOURCES, help="Sync only one source")
    parser.add_argument("--list-sources", action="store_true", help="Print available sources as JSON and exit")
    args = parser.parse_args()

    if args.list_sources:
        info = {k: {"display_name": v.display_name, "url": v.url} for k, v in SOURCES.items()}
        print(json.dumps(info, indent=2))
        return

    data = json.loads(RESULTS_PATH.read_text())
    all_changes = []
    synced = []

    for key in [args.source] if args.source else SOURCES:
        src = SOURCES[key]
        print(f"Syncing {src.display_name}...")
        changes = src.sync(data)
        all_changes.extend(changes)
        if changes:
            synced.append(key)

    if not all_changes:
        print("No changes.")
        return

    print(f"\n{len(all_changes)} change(s):")
    for c in all_changes:
        print(c)

    if args.apply:
        data["results"].sort(key=lambda r: (r["benchmark"], r["model"]))
        data["last_updated"] = date.today().isoformat()
        RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"\nWritten to {RESULTS_PATH}")
        _set_github_output(synced, len(all_changes))
    else:
        print("\nDry-run. Use --apply to write changes.")


if __name__ == "__main__":
    main()
