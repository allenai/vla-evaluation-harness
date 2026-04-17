# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Refine raw extractions into leaderboard.json.

Two-stage pipeline:

1. `build_candidates()` — deterministic Python step. Applies the protocol
   gate (`yes` → compute `overall_score` from components; `no`/`partial`/
   `unknown` → keep row with `overall_score = null`) and emits candidate
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
    - Protocol gate: `yes` computes `overall_score` from components;
      everything else (`no`/`partial`/`unknown`) keeps `overall_score = null`
      but the row is retained so non-standard subsets still surface on the
      leaderboard as unranked entries (see leaderboard/CONTRIBUTING.md).
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
        "rows_match_no_kept_null": 0,  # protocol=no rows kept with overall_score=null
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
                scores_raw = m.get("scores") or {}
                suite_scores = _to_plain_scores(scores_raw.get("suite_scores"))
                task_scores = _to_plain_scores(scores_raw.get("task_scores"))

                # Arithmetic / protocol gate.
                #
                # Only `matches_standard == "yes"` rows get an aggregated
                # `overall_score`. Everything else (including the hard "no"
                # case) keeps `overall_score = null` but stays in the
                # candidate pool so non-standard subsets still appear on
                # the leaderboard as non-ranked entries — see
                # leaderboard/CONTRIBUTING.md "non-standard entries must
                # set overall_score to null and store the original
                # aggregate in task_scores.reported_avg".
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
                    if match == "no":
                        stats["rows_match_no_kept_null"] += 1
                    # Non-standard-protocol preservation: per
                    # leaderboard/CONTRIBUTING.md, an entry with a
                    # reported aggregate but no component breakdown keeps
                    # the paper's number in `task_scores.reported_avg`
                    # (overall_score stays null so it does not rank).
                    # Without this recovery the row would be dropped by
                    # the empty-score check below.
                    raw_overall = scores_raw.get("overall_score")
                    if isinstance(raw_overall, (int, float)) and not task_scores and not suite_scores:
                        task_scores = {"reported_avg": raw_overall}

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
        f"  match=no kept with null overall: {stats['rows_match_no_kept_null']}\n"
        f"  dropped (no score after conv):   {stats['rows_drop_empty_after_conversion']}\n"
        f"  kept as candidates:              {stats['rows_kept']}"
    )


# ---------------------------------------------------------------------------
# LLM step: fuzzy decisions on pre-built candidates
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    return f"""You are the PRECISION stage of a two-stage VLA leaderboard pipeline.

An EXTRACT stage (with Read access to each paper) produced the candidates at
`{CANDIDATES_PATH}`. A deterministic Python step has already applied the
protocol gate: rows with `protocol_match == "yes"` have `overall_score`
computed from component scores; all other rows keep `overall_score = null`
but are retained.

You do NOT have paper access at this stage. All paper-derived context is
already in the candidate fields — rely on them.

## Candidate fields

Each candidate is one (paper × benchmark × model) row with:

- `name_in_paper`: the canonical method name the extract stage resolved
  from the paper. Treat as already cleaned — do not re-derive.
- `params`, `benchmark`, `weight_type`
- `overall_score`: computed by the python step. Never recompute or change.
- `suite_scores`, `task_scores`: component scores, plain numbers
- `reported_paper`, `reported_table`
- `protocol_match`: `"yes"` / `"no"` / `"partial"` / `"unknown"`
- `protocol_rationale`: the extract stage's reasoning — use as the basis
  for `notes`
- `is_score_original`: `"original"` / `"cited_baseline"` / `"reproduction"`
  / `"unknown"`

## Your job

Apply filters aggressively; when in doubt, drop. A small leaderboard of
canonical entries beats a large one with ablation junk.

### 1. Drop failed-resolution rows

Drop any row whose `name_in_paper` is still a generic label — "Ours",
"Our Method", "Our Model", "Proposed", "This Work", "Baseline", "(Ours)",
"Ablation", "(b)", "(c)", "variant X", or similar. The extract stage was
supposed to resolve these to the method's real name. Since you cannot
read the paper, a generic label means the row is not attributable.

### 2. Drop ablation / variant rows

Drop rows whose only differentiator is:
- quantization (INT4, INT8, AWQ, PTQ, QAT, GPTQ, ...)
- parameter-efficient tuning (LoRA, QLoRA, adapter, ...)
- training-stage snapshots ("stage 1", "50% data", "w/o pretrain")
- horizon / chunk / hyperparameter sweeps ("k=1", "chunk=8")
- "+feature X" / "w/o Y" style tags

Unless the variant IS the paper's main contribution.

### 3. Dedup

Distinct `reported_paper`s produce distinct entries — never collapse a
third-party measurement into a first-party canonical row. Within a single
`(model, benchmark, reported_paper)` triple, collapse duplicates and keep
the row with the most score detail.

### 4. Cross-benchmark identity

A model's first-party entries across benchmarks carry the same
`display_name`, `params`, and `model_paper`. Pick the most canonical
values. Third-party entries inherit the underlying method's canonical
values.

### 5. Assign `model` and `display_name`

`model` is a BibTeX-style key encoding provenance:

- First-party (`reported_paper == model_paper`):
  - `model: kim24openvla`, `display_name: "OpenVLA"`
- Third-party measurement:
  - `model: kim24openvla__black24xvla`, `display_name: "OpenVLA (from X-VLA)"`

`display_name` for third-party entries makes the measuring paper obvious
to a reader scanning the leaderboard.

### 6. Compose `notes`

Base on `protocol_rationale` (trim if long), append origin info. Write
something specific:

- OK: "ABC→D split with 1000 evaluation chains; avg_len metric; reproduction"
- OK: "18/18 PerAct tasks, single camera view; cited from RVT paper Table 2"
- Bad: "partial protocol match"
- Bad: ""

## Benchmark rules

Each block in `leaderboard/benchmarks/*.md` opens with `**Standard**: ...`
(the canonical protocol). You already have resolved scores, so consult
these only for dedup / identity / notes context.

## Output

Write `{LEADERBOARD_PATH}`:

```
{{
  "last_updated": "{date.today().isoformat()}",
  "results": [<entry>, ...]
}}
```

Each entry matches `{SCHEMA_PATH}` with fields: `model`, `display_name`,
`name_in_paper`, `params`, `model_paper`, `benchmark`, `weight_type`,
`overall_score`, `suite_scores`, `task_scores`, `reported_paper`,
`reported_table`, `curated_by`, `date_added`, `notes`.

- Copy `name_in_paper` verbatim from the candidate. Never blank it out,
  never synthesize from `display_name`.
- `curated_by` uses the form `"<family> <version>"` — e.g. `"opus 4.6"`,
  `"sonnet 4.6"`. The schema rejects `"claude-sonnet-4-6"` and similar.
- `date_added = "{date.today().isoformat()}"`.
- `results` sorted by `(benchmark, model)`. UTF-8, trailing newline.

Never touch `overall_score` — the python step computed it.

Report what you dropped and why when you finish.
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
