# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Refine raw extractions into leaderboard.json via LLM agent.

Launches Claude as an agent with file system access. The agent reads
extractions, applies benchmark protocols, deduplicates, and writes
leaderboard.json directly.

Usage::

    uv run refine.py
    uv run refine.py --model opus --benchmark libero
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


def _build_system_prompt(benchmark_filter: str | None) -> str:
    # Collect all benchmark protocols
    protocols = []
    for f in sorted(BENCHMARKS_DIR.glob("*.md")):
        if f.stem == "_global":
            continue
        if benchmark_filter and f.stem != benchmark_filter:
            continue
        md = _load_benchmark_md(f.stem)
        if md:
            protocols.append(f"### {f.stem}\n{md}")

    global_md = _load_benchmark_md("_global")
    protocol_text = "\n\n".join(protocols)

    return f"""You are curating the VLA benchmark leaderboard.

Your job: read the raw extraction files in {EXTRACTIONS_DIR}/, apply
the benchmark protocols below, and write a clean leaderboard.json to
{LEADERBOARD_PATH}.

## What to do:

1. Read all extraction JSON files from {EXTRACTIONS_DIR}/.
   Each file is one paper with extracted benchmark results.

2. For each benchmark, collect all extracted models across papers.

3. **Deduplicate**: Same model under different names (e.g. "OpenVLA",
   "OpenVLA-7B", "OpenVLA (Kim et al.)") → keep the best entry
   (prefer original evaluation > reproduction > cited baseline).
   Use clean model keys (lowercase, underscored, π→pi, ₀→0).

4. **Protocol gate**: If a model's evaluation protocol doesn't match
   the standard (see rules below), set overall_score to null.
   If the protocol is a reject condition (e.g. CALVIN ABCD→D),
   exclude the entry entirely.

5. **overall_score**: Compute from component scores per each benchmark's
   aggregation rule. Do NOT copy the paper's reported average.
   If required components are missing, set null.

6. **weight_type**: Must be "shared" or "finetuned". Judge from context.

7. Write the output JSON to {LEADERBOARD_PATH} with this structure:
   {{"last_updated": "YYYY-MM-DD", "results": [...]}}

   Each entry: model, display_name, name_in_paper, params, model_paper,
   benchmark, weight_type, overall_score, suite_scores, task_scores,
   source_paper, source_table, curated_by, date_added, notes.

   Sort by (benchmark, model). Use "refine.py (opus)" as curated_by.

8. Also read {DATA_DIR}/benchmarks.json for the benchmark registry.
   Include it as the "benchmarks" key in the output.

## Global rules:
{global_md}

## Benchmark protocols:
{protocol_text}

Use Bash to run Python code for processing the data. Use Read to
inspect files. Use Write to create the output. Work systematically
benchmark by benchmark.
"""


def refine(
    model: str = "opus", benchmark: str | None = None, output: Path = LEADERBOARD_PATH, timeout: int = 1800
) -> None:
    system_prompt = _build_system_prompt(benchmark)

    scope = f"benchmark {benchmark}" if benchmark else "all benchmarks"
    user_msg = f"Refine extractions for {scope}. Write output to {output}."

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--allowedTools",
        "Bash,Read,Write",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        user_msg,
    ]

    print(f"Launching claude ({model}) to refine {scope}...")
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"claude exited with code {result.returncode}")
        raise SystemExit(result.returncode)

    if output.exists():
        data = json.loads(output.read_text())
        n = len(data.get("results", []))
        print(f"Done: {output} ({n} entries)")
    else:
        print(f"Warning: {output} was not created")


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
    model: Annotated[str, typer.Option(help="Claude model.")] = "opus",
    timeout: Annotated[int, typer.Option(help="Timeout in seconds.")] = 1800,
) -> None:
    """Refine extractions into leaderboard.json via LLM agent."""
    refine(model=model, benchmark=benchmark, output=output, timeout=timeout)


if __name__ == "__main__":
    app()
