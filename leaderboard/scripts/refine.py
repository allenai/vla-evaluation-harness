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
# Dedup (rule-based fallback)
# ---------------------------------------------------------------------------

_PROVENANCE_PRIORITY = {"original": 0, "reproduction": 1, "cited_baseline": 2, "unknown": 3}


def _dedup_entries(entries: list[dict]) -> list[dict]:
    """Rule-based dedup by (model, benchmark). Keep highest-priority provenance."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        key = (e["model"], e["benchmark"])
        by_key.setdefault(key, []).append(e)

    deduped = []
    for key, group in by_key.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            group.sort(
                key=lambda e: (
                    _PROVENANCE_PRIORITY.get(e.get("_provenance", "unknown"), 3),
                    e.get("date_added", ""),
                )
            )
            deduped.append(group[0])
    return deduped


# ---------------------------------------------------------------------------
# LLM-assisted refinement (per-benchmark batch)
# ---------------------------------------------------------------------------

_REFINE_SCHEMA: dict = {
    "type": "object",
    "required": ["groups", "rejections"],
    "properties": {
        "groups": {
            "type": "array",
            "description": "Groups of entries that refer to the same model. Each group merges into one entry.",
            "items": {
                "type": "object",
                "required": ["canonical_key", "canonical_display_name", "entry_indices", "keep_index", "reason"],
                "properties": {
                    "canonical_key": {"type": "string", "description": "The model key to use for the merged entry"},
                    "canonical_display_name": {"type": "string", "description": "Clean display name for the model"},
                    "entry_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices of entries in this group (0-based)",
                    },
                    "keep_index": {
                        "type": "integer",
                        "description": "Index of the entry to keep (best provenance/scores)",
                    },
                    "reason": {"type": "string"},
                },
            },
        },
        "rejections": {
            "type": "array",
            "description": "Entries to reject entirely (protocol violations, invalid data).",
            "items": {
                "type": "object",
                "required": ["entry_index", "reason"],
                "properties": {
                    "entry_index": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


def _call_claude_cli(
    system_prompt: str, user_content: str, json_schema: dict, model: str = "sonnet", timeout: int = 300
) -> dict:
    import subprocess

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--json-schema",
        json.dumps(json_schema),
        "--output-format",
        "json",
        "--disallowedTools",
        "Bash,Write,Edit,NotebookEdit,WebFetch,WebSearch,Read,Grep,Glob",
        "--permission-mode",
        "default",
        "--no-session-persistence",
    ]
    result = subprocess.run(cmd, input=user_content, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:500]}")
    envelope = json.loads(result.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"CLI error: {envelope.get('subtype')}")
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        raise RuntimeError("no structured_output")
    return structured


def _llm_refine_benchmark(benchmark: str, entries: list[dict], model: str) -> list[dict]:
    """Use LLM to deduplicate and review entries for one benchmark."""
    if len(entries) <= 1:
        return entries

    # Build a compact summary for the LLM
    summary_lines = []
    for i, e in enumerate(entries):
        summary_lines.append(
            f"[{i}] key={e['model']} display={e['display_name']} "
            f"source={e.get('source_paper', '')} "
            f"provenance={e.get('_provenance', '?')} "
            f"overall={e.get('overall_score')} "
            f"weight={e.get('weight_type', '?')} "
            f"params={e.get('params', '?')}"
        )

    system_prompt = f"""You are refining leaderboard entries for the {benchmark} benchmark.

You will see a list of extracted entries. Your job:
1. Identify entries that refer to the SAME model under different names
   (e.g. "OpenVLA" and "OpenVLA-7B" and "OpenVLA (Kim et al.)" are the same model).
   Group them and pick the best entry (prefer original > reproduction > cited_baseline).
   Choose a clean canonical_key (lowercase, underscored) and display_name.
2. Flag entries that should be REJECTED (invalid protocol, corrupt data).
3. Entries that are genuinely different models should NOT be grouped.

Return groups (for merging) and rejections (for removal). Entries not mentioned
in any group or rejection are kept as-is."""

    user_content = f"Benchmark: {benchmark}\n{len(entries)} entries:\n\n" + "\n".join(summary_lines)

    try:
        result = _call_claude_cli(system_prompt, user_content, _REFINE_SCHEMA, model=model)
    except Exception as e:
        print(f"  LLM refine failed for {benchmark}: {e}")
        return entries

    # Apply rejections
    reject_indices = {r["entry_index"] for r in result.get("rejections", []) if 0 <= r["entry_index"] < len(entries)}

    # Apply groups
    consumed_indices: set[int] = set()
    refined: list[dict] = []
    for group in result.get("groups", []):
        indices = [i for i in group.get("entry_indices", []) if 0 <= i < len(entries)]
        keep_idx = group.get("keep_index")
        if keep_idx is None or keep_idx not in indices:
            continue
        if keep_idx in reject_indices:
            continue
        kept = dict(entries[keep_idx])
        kept["model"] = group.get("canonical_key", kept["model"])
        kept["display_name"] = group.get("canonical_display_name", kept["display_name"])
        refined.append(kept)
        consumed_indices.update(indices)

    # Add ungrouped, unrejected entries
    for i, e in enumerate(entries):
        if i not in consumed_indices and i not in reject_indices:
            refined.append(e)

    return refined


def _llm_refine_all(entries: list[dict], model: str) -> list[dict]:
    """Run LLM refinement per benchmark."""
    from collections import defaultdict

    by_bm: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_bm[e["benchmark"]].append(e)

    refined: list[dict] = []
    for bm in sorted(by_bm):
        bm_entries = by_bm[bm]
        print(f"  Refining {bm}: {len(bm_entries)} entries...")
        result = _llm_refine_benchmark(bm, bm_entries, model)
        print(f"    → {len(result)} entries ({len(bm_entries) - len(result)} removed)")
        refined.extend(result)
    return refined


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_results(benchmark_filter: str | None = None, llm_model: str | None = None) -> dict:
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

    # Dedup: rule-based first, then LLM if requested
    entries = _dedup_entries(entries)
    if llm_model:
        print(f"Running LLM refinement ({llm_model})...")
        entries = _llm_refine_all(entries, llm_model)

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
    model: Annotated[
        Optional[str],
        typer.Option(help="Claude model for LLM-assisted refinement (e.g. sonnet). Omit for rule-based only."),
    ] = None,
) -> None:
    """Compile extraction cache into leaderboard.json."""
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    n_files = len(list(EXTRACTIONS_DIR.glob("*.json")))
    typer.echo(f"Reading {n_files} extraction files...")

    data = build_results(benchmark_filter=benchmark, llm_model=model)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    typer.echo(f"Wrote {output}")
    _print_report(data)


if __name__ == "__main__":
    app()
