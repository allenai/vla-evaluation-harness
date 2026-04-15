# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Refine raw extractions into leaderboard.json.

Two-stage pipeline:

1. `build_candidates()` — deterministic Python step. Applies the protocol
   gate (drop `matches_standard = no`, null out `partial`), computes
   `overall_score` arithmetically from components, and emits candidate
   entries in a pre-schema shape.

2. LLM agent (opus) — receives the candidate entries via a temp file and
   only handles FUZZY decisions: eligibility (drop "Ours"/ablation/quant
   labels), dedup across papers, cross-benchmark identity consistency,
   and substantive note composition.

Why this split: protocol gating and arithmetic are deterministic rules —
they do not need LLM judgment and the LLM was getting them wrong. The
LLM now only tackles problems that require fuzzy reasoning.

Usage::

    uv run refine.py
    uv run refine.py --model opus --benchmark libero
"""

from __future__ import annotations

import functools
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Annotated, Optional

import typer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTRACTIONS_DIR = ROOT / ".cache" / "extractions"
BENCHMARKS_DIR = ROOT / "benchmarks"
BENCHMARKS_JSON_PATH = DATA_DIR / "benchmarks.json"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"
SCHEMA_PATH = DATA_DIR / "leaderboard.schema.json"
CANDIDATES_PATH = ROOT / ".cache" / "refine_candidates.json"
REFINE_LOGS_DIR = ROOT / ".cache" / "refine_logs"

# Aggregation rules live in each benchmark's md frontmatter and are
# compiled into benchmarks.json. This function reads them from there —
# never hardcode a rule here; edit the benchmark's .md frontmatter instead.


@functools.cache
def _aggregation_rules() -> dict[str, dict | str]:
    """Return {benchmark: "forbidden" | {container, keys}} from benchmarks.json."""
    data = json.loads(BENCHMARKS_JSON_PATH.read_text())
    return {k: v["aggregation"] for k, v in data.items() if "aggregation" in v}


# ---------------------------------------------------------------------------
# Deterministic pre-step: build candidate entries from raw extractions
# ---------------------------------------------------------------------------


def _compute_overall(benchmark: str, suite: dict, task: dict) -> float | None:
    """Compute overall_score from component scores per aggregation rule.

    Returns None for: missing rule, `"forbidden"` rule, or partial component
    data. `isinstance(rule, dict)` handles all three non-computing cases in
    one branch (None / str / dict union from `_aggregation_rules`).
    """
    rule = _aggregation_rules().get(benchmark)
    if not isinstance(rule, dict):
        return None
    container = suite if rule["container"] == "suite_scores" else task
    values = [container[k] for k in rule["keys"] if k in container]
    if len(values) != len(rule["keys"]):
        return None
    return round(sum(values) / len(values), 1)


def _to_plain_scores(container: dict | None) -> dict:
    """Convert extraction score shape {k: {value, quote}} → {k: value}."""
    if not container:
        return {}
    out = {}
    for k, v in container.items():
        val = v["value"] if isinstance(v, dict) else v
        if isinstance(val, (int, float)):
            out[k] = val
    return out


def build_candidates(benchmark_filter: str | None = None) -> tuple[list[dict], dict]:
    """Read extractions and emit candidate entries.

    Applied here (deterministic):
    - Protocol gate: drop `matches_standard == "no"`; null out `overall_score`
      for `partial`/`unknown`; compute from components for `yes`.
    - Arithmetic: mean of required component keys per the benchmark's
      `aggregation` rule in benchmarks.json.
    - Forbidden-overall enforcement for benchmarks with aggregation `"forbidden"`.
    - Schema field population (reported_paper, reported_table, etc).

    NOT applied here (left to the LLM step):
    - Eligibility filter (junk labels, ablation variants)
    - Dedup across papers
    - Cross-benchmark identity consistency
    - Substantive note composition

    Returns (candidates, stats) where stats tracks how each paper and row
    was handled for pipeline audit.
    """
    candidates: list[dict] = []
    stats = {
        "extractions_total": 0,
        "papers_empty": 0,  # extraction file had benchmarks:[] — citing paper, no scores
        "papers_with_scores": 0,
        "rows_total": 0,
        "rows_drop_protocol_no": 0,
        "rows_drop_empty_after_conversion": 0,
        "rows_kept": 0,
    }
    for ext_file in sorted(EXTRACTIONS_DIR.glob("*.json")):
        ext = json.loads(ext_file.read_text())
        arxiv_id = ext.get("arxiv_id")
        if not arxiv_id:
            continue
        stats["extractions_total"] += 1
        reported_paper = f"https://arxiv.org/abs/{arxiv_id}"
        ext_benchmarks = ext.get("benchmarks") or []
        if not ext_benchmarks:
            stats["papers_empty"] += 1
            continue
        stats["papers_with_scores"] += 1
        for bm in ext_benchmarks:
            benchmark = bm.get("benchmark")
            if benchmark_filter and benchmark != benchmark_filter:
                continue
            for m in bm.get("models", []):
                stats["rows_total"] += 1
                protocol = m.get("protocol") or {}
                match = protocol.get("matches_standard", "unknown")
                # Hard reject: LLM already judged this protocol non-matching
                if match == "no":
                    stats["rows_drop_protocol_no"] += 1
                    continue
                scores_raw = m.get("scores") or {}
                suite_scores = _to_plain_scores(scores_raw.get("suite_scores"))
                task_scores = _to_plain_scores(scores_raw.get("task_scores"))

                # Arithmetic / protocol gate
                if match == "yes":
                    overall = _compute_overall(benchmark, suite_scores, task_scores)
                    # Fallback: if the benchmark has no aggregation rule at
                    # all, trust the LLM-extracted overall. Forbidden rules
                    # and partial-data cases keep overall=None.
                    if overall is None and _aggregation_rules().get(benchmark) is None:
                        raw_overall = scores_raw.get("overall_score")
                        if isinstance(raw_overall, (int, float)):
                            overall = raw_overall
                else:
                    overall = None

                # Skip entries with no score at all (schema requires >=1).
                if overall is None and not suite_scores and not task_scores:
                    stats["rows_drop_empty_after_conversion"] += 1
                    continue

                stats["rows_kept"] += 1
                candidates.append(
                    {
                        "name_in_paper": m.get("label", ""),
                        "params": m.get("params"),
                        "benchmark": benchmark,
                        "weight_type": m.get("weight_type")
                        if m.get("weight_type") in ("shared", "finetuned")
                        else "shared",
                        "overall_score": overall,
                        "suite_scores": suite_scores,
                        "task_scores": task_scores,
                        "reported_paper": reported_paper,
                        "reported_table": scores_raw.get("reported_table"),
                        "protocol_match": match,
                        "protocol_rationale": protocol.get("rationale", ""),
                        "is_score_original": m.get("is_score_original", "unknown"),
                    }
                )
    return candidates, stats


def _print_stats(stats: dict) -> None:
    """Print a one-page audit of what build_candidates() did."""
    print(
        f"Extractions scanned: {stats['extractions_total']}\n"
        f"  papers with scores:              {stats['papers_with_scores']}\n"
        f"  papers empty (cited, no scores): {stats['papers_empty']}\n"
        f"Model rows processed: {stats['rows_total']}\n"
        f"  dropped (protocol 'no'):         {stats['rows_drop_protocol_no']}\n"
        f"  dropped (no score after conv):   {stats['rows_drop_empty_after_conversion']}\n"
        f"  kept as candidates:              {stats['rows_kept']}"
    )


# ---------------------------------------------------------------------------
# LLM step: fuzzy decisions on pre-built candidates
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    return f"""You are the PRECISION stage of a two-stage VLA leaderboard pipeline.

## Pipeline role

The EXTRACT stage that produced these candidates was deliberately RECALL-FIRST:
it surfaced every row it could find with a number on a registry benchmark,
including cited baselines from related work, framework / architecture variants,
and rows whose protocol it marked `partial` or `unknown`. Many of those rows
should NOT end up on the public leaderboard.

A deterministic Python pre-step has already applied the hard protocol gate:
- `matches_standard == "no"` rows were dropped.
- `partial` / `unknown` rows kept with `overall_score = null`.
- `yes` rows have `overall_score` computed from component suite/task scores.

Your job is the FUZZY precision pass: drop everything else that doesn't
belong (junk labels, ablation variants, stale cited baselines), dedup
across papers, normalize identity across benchmarks, and write substantive
notes. Apply your filters aggressively. When in doubt, drop. A small,
clean leaderboard is the goal.

## Benchmark rule template

Each benchmark block (embedded elsewhere in this prompt or accessible via
`Read` on `leaderboard/benchmarks/*.md`) opens with a bold `**Standard**: ...`
line — the canonical protocol in one sentence. `Scoring` prescribes the
JSON shape. `Checks` are yes/no questions a row must pass; a failed check
means the row's `overall_score` must stay `null`. `Methodology axes` are
variance dimensions you must record in `notes` — they are NOT protocol
violations, so a row that merely differs along these axes keeps its
`overall_score` populated.

## Context

The candidate entries are in `{CANDIDATES_PATH}`. Each candidate is one
(paper × benchmark × model) row from a raw extraction, with these fields
already filled in:

- `name_in_paper`: exact label from the paper's table
- `params`, `benchmark`, `weight_type`
- `overall_score`: either computed from components or null if the
  protocol does not match standard. Do NOT recompute it.
- `suite_scores`, `task_scores`: component scores, already plain numbers
- `reported_paper`, `reported_table`
- `protocol_match`: "yes" / "partial" / "unknown" (candidates with "no"
  were already dropped by the python step)
- `protocol_rationale`: the LLM rationale from the extraction step —
  use this as the basis for your `notes` field
- `is_score_original`: "original" / "cited_baseline" / "reproduction" / "unknown"

## Your job (fuzzy decisions only)

1. **Eligibility filter**: drop candidates whose `name_in_paper` indicates
   junk. Specifically drop when `name_in_paper` is "Ours", "Our Method",
   "Our Model", "Proposed", "This Work", "Baseline", contains "(Ours)" /
   "(ours)", or is otherwise a placeholder with no public identity.
   Also drop ablation / variant rows whose differentiator is ONLY:
   - quantization (INT4, INT8, AWQ, PTQ, QAT, GPTQ, ...)
   - parameter-efficient tuning (LoRA, QLoRA, adapter, ...)
   - training-stage snapshots ("stage 1", "50% data", "w/o pretrain")
   - horizon / chunk / hyperparameter sweeps ("k=1", "chunk=8")
   - "+feature X", "w/o Y" style ablation tags
   - An unnamed "row (b)" / "(c)" style label
   Unless that variant IS clearly the paper's main contribution.

2. **Dedup**: distinct `reported_paper`s produce distinct entries. Never
   collapse a third-party measurement into a first-party canonical row.
   Two rows with the same model on the same benchmark, but different
   `reported_paper`, must remain separate. Within a single
   `(model, benchmark, reported_paper)` triple, collapse duplicates and
   prefer the row with more score detail (more suite/task keys, non-null
   overall).

3. **Cross-benchmark identity**: a model's first-party entries across
   different benchmarks must carry the same `display_name`, `params`, and
   `model_paper`. Pick the most detailed / most canonical values. (This
   rule applies inside the first-party set; third-party entries inherit
   the same canonical values for the underlying method.)

4. **Compose `notes`**: for each kept entry, write a substantive,
   human-readable note. Use the `protocol_rationale` field as the basis
   (trim if long), and append origin info. NEVER write generic labels
   like "partial protocol match" / "score cited" / empty. A reader
   hovering the score should learn something specific about what was
   evaluated and how.

   Good: "ABC→D split with 1000 evaluation chains; avg_len metric; reproduction"
   Good: "18/18 PerAct tasks, single camera view; cited from RVT paper Table 2"
   Bad: "partial protocol match"
   Bad: ""

5. **Assign `model` key and `display_name`**: the `model` field must be a
   BibTeX citation key that makes the entry's provenance self-explanatory.
   For first-party entries (`reported_paper == model_paper`), use the
   method's own citation key. For third-party measurements, the key must
   encode both the method and the measuring paper.

   `display_name` is human-readable. For third-party entries, the display
   name must make the source obvious to a reader scanning the leaderboard.

   Examples (illustrative, not prescriptive):
     first-party  →  model: `kim24openvla`,
                     display_name: "OpenVLA"
     third-party  →  model: `kim24openvla__black24xvla`,
                     display_name: "OpenVLA (from X-VLA)"

## Output

Write `{LEADERBOARD_PATH}` as JSON:

```
{{
  "last_updated": "{date.today().isoformat()}",
  "results": [<entry>, ...]
}}
```

Each entry must match the schema at `{SCHEMA_PATH}`:

- `model`, `display_name`, `name_in_paper`, `params`, `model_paper`,
  `benchmark`, `weight_type`, `overall_score`, `suite_scores`,
  `task_scores`, `reported_paper`, `reported_table`, `curated_by`,
  `date_added`, `notes`

Set `curated_by` to your own model alias (e.g. `"opus 4.6"` — the model
running this refine step; NOT the literal string `"refine.py"`) and
`date_added = "{date.today().isoformat()}"`.

`results` MUST be sorted by `(benchmark, model)`. The file should be
UTF-8 with a trailing newline.

## Bias toward dropping

When in doubt, DROP. A smaller leaderboard of canonical entries is much
better than a large one with ablation junk. A reader wants "what are
the main VLA models and how do they compare", not every table row.

## Constraints

- Do NOT touch `overall_score` — the python step computed it. Your
  changes are limited to which rows survive, how they are named, and
  what notes they carry.
- Report what you dropped and why when you are done.
"""


def refine(
    model: str = "opus",
    benchmark: str | None = None,
    output: Path = LEADERBOARD_PATH,
    timeout: int = 7200,
) -> None:
    # Stage 1: build candidates (deterministic)
    print("Stage 1: building candidates from extractions...")
    candidates, stats = build_candidates(benchmark_filter=benchmark)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n")
    _print_stats(stats)
    print(f"Wrote {CANDIDATES_PATH}")

    if not candidates:
        print("No candidates to refine. Exiting.")
        return

    # Stage 2: launch LLM agent for fuzzy decisions
    system_prompt = _build_system_prompt()
    scope = f"benchmark {benchmark}" if benchmark else "all benchmarks"
    user_msg = (
        f"Refine {len(candidates)} candidates for {scope}. "
        f"Candidates are at {CANDIDATES_PATH}. Write the final leaderboard to {output}."
    )

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        user_msg,
    ]

    log_path = REFINE_LOGS_DIR / f"refine_{date.today().isoformat()}.log"
    print(f"Stage 2: launching claude ({model}) for fuzzy decisions on {scope}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")

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

app = typer.Typer(help="Refine raw extractions into leaderboard.json.", add_completion=False)


@app.command()
def main(
    output: Annotated[Path, typer.Option("-o", help="Output path.")] = LEADERBOARD_PATH,
    benchmark: Annotated[Optional[str], typer.Option(help="Only refine this benchmark.")] = None,
    model: Annotated[str, typer.Option(help="Claude model for the fuzzy stage.")] = "opus",
    timeout: Annotated[int, typer.Option(help="LLM timeout in seconds.")] = 7200,
) -> None:
    """Refine extractions into leaderboard.json (python pre-step + LLM fuzzy stage)."""
    refine(model=model, benchmark=benchmark, output=output, timeout=timeout)


@app.command()
def candidates(
    benchmark: Annotated[Optional[str], typer.Option(help="Only build for this benchmark.")] = None,
) -> None:
    """Stage 1 only: build candidate entries and exit (no LLM call)."""
    cs, stats = build_candidates(benchmark_filter=benchmark)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(cs, indent=2, ensure_ascii=False) + "\n")
    _print_stats(stats)
    print(f"Wrote {CANDIDATES_PATH}")


if __name__ == "__main__":
    app()
