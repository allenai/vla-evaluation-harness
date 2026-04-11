# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Refine raw extractions into leaderboard.json via LLM.

Reads per-paper extraction files, sends each benchmark's entries to
the LLM for dedup, protocol review, and score curation.

Usage::

    uv run refine.py --model sonnet
    uv run refine.py --model sonnet --benchmark libero
    uv run refine.py --model sonnet -o /tmp/leaderboard.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTRACTIONS_DIR = ROOT / "extractions"
BENCHMARKS_DIR = ROOT / "benchmarks"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"

# ---------------------------------------------------------------------------
# Benchmark protocol loading
# ---------------------------------------------------------------------------


def _load_benchmark_md(bm_key: str) -> str:
    path = BENCHMARKS_DIR / f"{bm_key}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3 :].strip()
    return text


# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------

REFINE_SCHEMA: dict = {
    "type": "object",
    "required": ["entries"],
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["model", "display_name", "benchmark", "overall_score"],
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Canonical model key (lowercase, underscored, no Greek)",
                    },
                    "display_name": {"type": "string", "description": "Clean display name for the leaderboard"},
                    "name_in_paper": {"type": ["string", "null"], "description": "Exact label from the source paper"},
                    "params": {"type": ["string", "null"]},
                    "model_paper": {"type": ["string", "null"]},
                    "benchmark": {"type": "string"},
                    "weight_type": {"type": "string", "enum": ["shared", "finetuned"]},
                    "overall_score": {"type": ["number", "null"]},
                    "suite_scores": {"type": ["object", "null"]},
                    "task_scores": {"type": ["object", "null"]},
                    "source_paper": {"type": ["string", "null"]},
                    "source_table": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
            },
        },
    },
}


def _call_claude(system_prompt: str, user_content: str, model: str = "sonnet", timeout: int = 600) -> dict:
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--json-schema",
        json.dumps(REFINE_SCHEMA),
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


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------


def _load_extractions_for_benchmark(benchmark: str) -> list[dict]:
    """Collect all extracted models for a benchmark across all papers."""
    entries = []
    for f in sorted(EXTRACTIONS_DIR.glob("*.json")):
        ext = json.loads(f.read_text())
        arxiv_id = ext.get("arxiv_id", "")
        for bm_data in ext.get("benchmarks", []):
            if bm_data.get("benchmark") != benchmark:
                continue
            for model in bm_data.get("models", []):
                entries.append({"arxiv_id": arxiv_id, "source_paper": f"https://arxiv.org/abs/{arxiv_id}", **model})
    return entries


def refine_benchmark(benchmark: str, raw_entries: list[dict], protocol_md: str, model: str) -> list[dict]:
    """Send one benchmark's raw entries to LLM for refinement."""
    if not raw_entries:
        return []

    # Build compact input
    lines = []
    for i, e in enumerate(raw_entries):
        scores = e.get("scores") or {}
        suite = {
            k: v.get("value") if isinstance(v, dict) else v for k, v in (scores.get("suite_scores") or {}).items()
        }
        task = {k: v.get("value") if isinstance(v, dict) else v for k, v in (scores.get("task_scores") or {}).items()}
        lines.append(
            json.dumps(
                {
                    "idx": i,
                    "label": e.get("label"),
                    "params": e.get("params"),
                    "source": e.get("source_paper"),
                    "provenance": e.get("is_score_original"),
                    "weight_type": e.get("weight_type"),
                    "overall": scores.get("overall_score"),
                    "suite_scores": suite or None,
                    "task_scores": task or None,
                    "source_table": scores.get("source_table"),
                    "protocol": e.get("protocol"),
                },
                ensure_ascii=False,
            )
        )

    system_prompt = f"""You are curating leaderboard entries for the {benchmark} benchmark.

## Protocol rules for {benchmark}:
{protocol_md}

## Your job:
Given raw extracted entries from multiple papers, produce a clean list of
leaderboard entries. For each entry you output:

1. **Dedup**: If multiple entries are the same model (different names like
   "OpenVLA" vs "OpenVLA-7B" vs "OpenVLA (Kim et al.)"), keep only the
   best one (prefer original > reproduction > cited_baseline).
   Use a clean canonical model key (lowercase, underscored, transliterate
   Greek: π→pi, ₀→0, etc.) and a clean display_name.

2. **Protocol gate**: If the protocol doesn't match the standard (see rules
   above), set overall_score to null. If the protocol is a reject condition
   (e.g. CALVIN ABCD→D), exclude the entry entirely.

3. **overall_score**: Compute from component scores per the aggregation rule
   (e.g. LIBERO = mean of spatial/object/goal/10). Do NOT copy the paper's
   reported average — compute it yourself. If required components are missing,
   set null.

4. **Flatten scores**: suite_scores and task_scores should be flat dicts
   of key→number (no quotes/nested objects).

5. **weight_type**: Must be "shared" or "finetuned" (not "unknown").
   Judge from context.

6. **notes**: Add if there are protocol deviations worth documenting.

Output ONLY valid entries. Do not include rejected entries."""

    user_content = f"Benchmark: {benchmark}\n{len(raw_entries)} raw entries:\n\n" + "\n".join(lines)

    try:
        result = _call_claude(system_prompt, user_content, model=model)
    except Exception as e:
        print(f"  LLM error for {benchmark}: {e}")
        return []

    from datetime import datetime, timezone

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for entry in result.get("entries", []):
        entry["curated_by"] = f"refine.py ({model})"
        entry["date_added"] = date
        entry["benchmark"] = benchmark

    return result.get("entries", [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import typer  # noqa: E402
from typing import Annotated, Optional  # noqa: E402

app = typer.Typer(help="Refine raw extractions into leaderboard.json.", add_completion=False)


@app.command()
def main(
    output: Annotated[Path, typer.Option("-o", help="Output path.")] = LEADERBOARD_PATH,
    benchmark: Annotated[Optional[str], typer.Option(help="Only refine this benchmark.")] = None,
    model: Annotated[str, typer.Option(help="Claude model for refinement.")] = "opus",
) -> None:
    """Refine extractions into leaderboard.json via LLM."""
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    n_files = len(list(EXTRACTIONS_DIR.glob("*.json")))
    typer.echo(f"Reading {n_files} extraction files...")

    # Discover benchmarks from extraction data
    all_benchmarks: set[str] = set()
    for f in EXTRACTIONS_DIR.glob("*.json"):
        ext = json.loads(f.read_text())
        for bm_data in ext.get("benchmarks", []):
            all_benchmarks.add(bm_data["benchmark"])

    if benchmark:
        all_benchmarks = {benchmark} & all_benchmarks

    all_entries: list[dict] = []
    for bm in sorted(all_benchmarks):
        raw = _load_extractions_for_benchmark(bm)
        if not raw:
            continue
        protocol_md = _load_benchmark_md(bm)
        typer.echo(f"  {bm}: {len(raw)} raw entries → refining...")
        refined = refine_benchmark(bm, raw, protocol_md, model)
        typer.echo(f"    → {len(refined)} curated entries")
        all_entries.extend(refined)

    all_entries.sort(key=lambda e: (e.get("benchmark", ""), e.get("model", "")))

    benchmarks_json = DATA_DIR / "benchmarks.json"
    benchmarks_registry = json.loads(benchmarks_json.read_text()) if benchmarks_json.exists() else {}

    data = {
        "last_updated": all_entries[0]["date_added"] if all_entries else "",
        "benchmarks": benchmarks_registry,
        "results": all_entries,
    }
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    typer.echo(f"\nWrote {output}: {len(all_entries)} entries across {len(all_benchmarks)} benchmarks")


if __name__ == "__main__":
    app()
