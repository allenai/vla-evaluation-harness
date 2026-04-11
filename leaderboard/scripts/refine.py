# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Build leaderboard.json from the extraction cache.

Reads all per-paper extraction files from ``extractions/``,
applies curation rules (protocol gate, overall_score computation,
dedup), and writes ``leaderboard.json``.

Usage::

    uv run build.py
    uv run build.py -o /tmp/results_new.json
    uv run build.py --benchmark libero
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Optional

import typer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTRACTIONS_DIR = ROOT / "extractions"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"

# ---------------------------------------------------------------------------
# Aggregation rules (same logic as validate.py ARITHMETIC_RULES)
# ---------------------------------------------------------------------------

ARITHMETIC_TOLERANCE = 0.5

ARITHMETIC_RULES: dict[str, dict] = {
    "libero": {
        "container": "suite_scores",
        "required_keys": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    },
    "libero_plus": {
        "container": "suite_scores",
        "required_keys": ["camera", "robot", "language", "light", "background", "noise", "layout"],
    },
    "libero_pro": {
        "container": "suite_scores",
        "required_keys": [
            f"{suite}_{pert}"
            for suite in ("goal", "spatial", "long", "object")
            for pert in ("ori", "obj", "pos", "sem", "task")
        ],
    },
    "libero_mem": {
        "container": "task_scores",
        "required_keys": [f"T{i}" for i in range(1, 11)],
    },
    "mikasa": {
        "container": "task_scores",
        "required_keys": ["ShellGameTouch", "InterceptMedium", "RememberColor3", "RememberColor5", "RememberColor9"],
    },
    "vlabench": {
        "container": "suite_scores",
        "required_keys": ["in_dist_PS", "cross_category_PS", "commonsense_PS", "semantic_instruction_PS"],
    },
}

FORBIDDEN_OVERALL: set[str] = {"simpler_env", "robotwin_v2"}

# Benchmarks where specific protocol violations mean entry rejection
REJECT_PROTOCOLS: dict[str, list[str]] = {
    "calvin": ["D→D", "ABCD→D", "D->D", "ABCD->D"],
}


# ---------------------------------------------------------------------------
# Entry building
# ---------------------------------------------------------------------------


def _compute_overall_score(benchmark: str, suite_scores: dict, task_scores: dict) -> float | None:
    """Compute overall_score from component scores using ARITHMETIC_RULES."""
    if benchmark in FORBIDDEN_OVERALL:
        return None
    rule = ARITHMETIC_RULES.get(benchmark)
    if rule is None:
        return None  # no aggregation rule → don't set overall_score
    container = suite_scores if rule["container"] == "suite_scores" else task_scores
    values = []
    for k in rule["required_keys"]:
        v = container.get(k)
        if v is None:
            return None  # missing required key → null
        values.append(float(v))
    return round(sum(values) / len(values), 2)


def _should_reject(benchmark: str, protocol: dict) -> bool:
    """Check if entry should be entirely excluded based on protocol violations."""
    patterns = REJECT_PROTOCOLS.get(benchmark, [])
    if not patterns:
        return False
    rationale = (protocol.get("rationale") or "").lower()
    for pat in patterns:
        if pat.lower() in rationale:
            return True
    return False


_GREEK = {
    "π": "pi",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "σ": "sigma",
    "λ": "lambda",
    "μ": "mu",
    "φ": "phi",
    "θ": "theta",
    "ℰ": "E",
    "ℒ": "L",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
}


def _normalize_model_key(label: str) -> str:
    s = label
    for greek, ascii_val in _GREEK.items():
        s = s.replace(greek, ascii_val)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _flatten_scores(scores_obj: dict) -> tuple[dict, dict]:
    """Extract flat suite_scores and task_scores dicts from extraction format."""
    suite = {}
    for k, v in (scores_obj.get("suite_scores") or {}).items():
        if isinstance(v, dict):
            suite[k] = v.get("value")
        else:
            suite[k] = v
    task = {}
    for k, v in (scores_obj.get("task_scores") or {}).items():
        if isinstance(v, dict):
            task[k] = v.get("value")
        else:
            task[k] = v
    return suite, task


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

# Priority: original evaluation > reproduction > cited baseline
_PROVENANCE_PRIORITY = {"original": 0, "reproduction": 1, "cited_baseline": 2, "unknown": 3}


def _dedup_entries(entries: list[dict]) -> list[dict]:
    """Deduplicate entries by (model, benchmark). Keep highest-priority provenance."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        key = (e["model"], e["benchmark"])
        by_key.setdefault(key, []).append(e)

    deduped = []
    for key, group in by_key.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Sort by provenance priority, then by date (newest first)
            group.sort(
                key=lambda e: (
                    _PROVENANCE_PRIORITY.get(e.get("_provenance", "unknown"), 3),
                    e.get("date_added", ""),
                )
            )
            deduped.append(group[0])
    return deduped


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_results(benchmark_filter: str | None = None) -> dict:
    """Build leaderboard.json from extraction cache."""
    entries: list[dict] = []

    for f in sorted(EXTRACTIONS_DIR.glob("*.json")):
        ext = json.loads(f.read_text())
        arxiv_id = ext.get("arxiv_id", "")
        source_url = f"https://arxiv.org/abs/{arxiv_id}"

        for bm_data in ext.get("benchmarks", []):
            benchmark = bm_data.get("benchmark", "")
            if benchmark_filter and benchmark != benchmark_filter:
                continue

            for model in bm_data.get("models", []):
                protocol = model.get("protocol") or {}

                # Reject check
                if _should_reject(benchmark, protocol):
                    continue

                scores_obj = model.get("scores") or {}
                suite_scores, task_scores = _flatten_scores(scores_obj)

                # Compute overall_score deterministically
                overall = _compute_overall_score(benchmark, suite_scores, task_scores)

                # If protocol is non-standard, force null
                if protocol.get("matches_standard") not in ("yes", None):
                    overall = None

                label = model.get("label", "")
                model_key = _normalize_model_key(label)
                if not model_key:
                    continue

                entry: dict = {
                    "model": model_key,
                    "display_name": label,
                    "name_in_paper": label,
                    "params": model.get("params"),
                    "model_paper": source_url,
                    "benchmark": benchmark,
                    "weight_type": model.get("weight_type", "finetuned")
                    if model.get("weight_type") != "unknown"
                    else "finetuned",
                    "overall_score": overall,
                    "source_paper": source_url,
                    "source_table": scores_obj.get("source_table"),
                    "curated_by": f"extract.py ({ext.get('model_used', 'sonnet')})",
                    "date_added": ext.get("extracted_at", "")[:10],
                    "_provenance": model.get("is_score_original", "unknown"),
                }
                if suite_scores:
                    entry["suite_scores"] = suite_scores
                if task_scores:
                    entry["task_scores"] = task_scores

                entries.append(entry)

    # Dedup
    entries = _dedup_entries(entries)

    # Remove internal fields
    for e in entries:
        e.pop("_provenance", None)

    # Sort
    entries.sort(key=lambda e: (e["benchmark"], e["model"]))

    # Build final structure
    benchmarks_json = DATA_DIR / "benchmarks.json"
    benchmarks_registry = json.loads(benchmarks_json.read_text()) if benchmarks_json.exists() else {}

    return {
        "last_updated": entries[0]["date_added"] if entries else "",
        "benchmarks": benchmarks_registry,
        "results": entries,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(data: dict) -> None:
    """Print build summary to stdout."""
    from collections import Counter

    results = data["results"]
    bm_counts = Counter(r["benchmark"] for r in results)
    n_null = sum(1 for r in results if r.get("overall_score") is None)
    n_scored = len(results) - n_null

    typer.echo(f"Total entries: {len(results)}")
    typer.echo(f"  with overall_score: {n_scored}")
    typer.echo(f"  overall_score=null: {n_null}")
    typer.echo("\nPer benchmark:")
    for bm, count in bm_counts.most_common():
        null_count = sum(1 for r in results if r["benchmark"] == bm and r.get("overall_score") is None)
        typer.echo(f"  {bm:20s} {count:5d} entries ({null_count} null)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Build leaderboard.json from extraction cache.", add_completion=False)


@app.command()
def main(
    output: Annotated[Path, typer.Option("-o", help="Output path.")] = LEADERBOARD_PATH,
    benchmark: Annotated[Optional[str], typer.Option(help="Only include this benchmark.")] = None,
) -> None:
    """Compile extraction cache into leaderboard.json."""
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    n_files = len(list(EXTRACTIONS_DIR.glob("*.json")))
    typer.echo(f"Reading {n_files} extraction cache files...")

    data = build_results(benchmark_filter=benchmark)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    typer.echo(f"Wrote {output}")
    _print_report(data)


if __name__ == "__main__":
    app()
