# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Refine raw extractions into leaderboard.json.

Two-stage pipeline:

1. ``build_candidates()`` — deterministic Python step. Applies the
   protocol gate, computes overall_score per the benchmark's aggregation
   rule, classifies each row (first_party vs third_party) per the
   decision table in candidates.schema.json, and resolves collapse
   candidates by cross-checking the original paper's extraction.

2. LLM agent (opus) — per-benchmark fuzzy decisions: eligibility (drop
   ablation / generic-label rows), dedup across papers, cross-benchmark
   identity consistency, notes composition, and model-key assignment.

Why per-benchmark: one LLM call for the entire leaderboard blows
context and is hard to debug. Each benchmark is a bounded workload.

Usage::

    uv run refine.py
    uv run refine.py --model opus --benchmark libero
"""

from __future__ import annotations

import functools
import json
import re
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
LEADERBOARD_SCHEMA_PATH = DATA_DIR / "leaderboard.schema.json"
CANDIDATES_SCHEMA_PATH = DATA_DIR / "candidates.schema.json"
CANDIDATES_PATH = ROOT / ".cache" / "refine_candidates.json"
REFINE_LOGS_DIR = ROOT / ".cache" / "refine_logs"

_ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d+\.\d+)")


def _arxiv_id_of(url: str | None) -> str | None:
    if not url:
        return None
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def _citing_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _norm_name(name: str) -> str:
    """Normalize a method name for cross-paper matching.

    Case-insensitive, strips trailing parentheticals ('(ours)'), common
    whitespace/punct variations.
    """
    s = name.strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"[^\w]+", "", s)
    return s


@functools.cache
def _aggregation_rules() -> dict[str, dict | str]:
    """Return {benchmark: "forbidden" | {container, keys}} from benchmarks.json."""
    data = json.loads(BENCHMARKS_JSON_PATH.read_text())
    return {k: v["aggregation"] for k, v in data.items() if "aggregation" in v}


# ---------------------------------------------------------------------------
# Deterministic pre-step: build candidate entries from raw extractions
# ---------------------------------------------------------------------------


def _compute_overall(benchmark: str, suite: dict, task: dict) -> float | None:
    """Compute overall_score from component scores per the aggregation rule.

    Returns None for missing rule, ``"forbidden"`` rule, or partial
    component data.
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


def _load_all_extractions() -> dict[str, dict]:
    """Load every .cache/extractions/*.json into memory keyed by arxiv_id."""
    out: dict[str, dict] = {}
    for p in sorted(EXTRACTIONS_DIR.glob("*.json")):
        ext = json.loads(p.read_text())
        aid = ext.get("arxiv_id")
        if aid:
            out[aid] = ext
    return out


def _build_row_index(extractions: dict[str, dict]) -> set[tuple[str, str, str]]:
    """Index of (arxiv_id, benchmark, normalized name) for collapse cross-check."""
    idx: set[tuple[str, str, str]] = set()
    for aid, ext in extractions.items():
        for bm in ext.get("benchmarks", []):
            for m in bm.get("models", []):
                idx.add((aid, bm["benchmark"], _norm_name(m.get("name_in_paper", ""))))
    return idx


def _classify_row(
    citing_url: str,
    is_score_original: str,
    model_paper: str | None,
    cited_paper: str | None,
    benchmark: str,
    name_in_paper: str,
    row_index: set[tuple[str, str, str]],
) -> tuple[str, str, str | None]:
    """Resolve a row's attribution case. Returns (row_type, reported_paper, effective_cited_paper).

    row_type is 'first_party', 'third_party', or 'drop' (canonical row
    exists in the original paper's extraction — this row is redundant).
    """
    # Case 1: this paper introduces the method and ran it
    if is_score_original == "original" and citing_url == model_paper:
        return "first_party", citing_url, cited_paper

    model_arxiv = _arxiv_id_of(model_paper)
    cited_arxiv = _arxiv_id_of(cited_paper)

    # Case 2: cited baseline pointing at the method's own paper
    if is_score_original == "cited_baseline" and model_arxiv and cited_arxiv and cited_arxiv == model_arxiv:
        # Cross-check: does the original paper's extraction contain this row?
        if (model_arxiv, benchmark, _norm_name(name_in_paper)) in row_index:
            return "drop", "", None
        # Cite unverified — demote to third_party and null out the cite
        # so downstream treats it as citing-paper measured.
        return "third_party", citing_url, None

    # Case 3a/3b: third-party via arxiv cite. cited_arxiv being truthy
    # means cited_paper parsed as an arxiv URL, so it is a str here.
    if is_score_original == "cited_baseline" and cited_arxiv and cited_paper is not None:
        return "third_party", cited_paper, cited_paper

    # Case 3c / reproduction: citing paper measured it
    return "third_party", citing_url, cited_paper


def build_candidates(benchmark_filter: str | None = None) -> tuple[list[dict], dict]:
    """Read extractions and emit candidate entries matching candidates.schema.json.

    Applied here (deterministic):
    - Protocol gate: match='yes' → compute overall_score from components
      (or use raw when no rule); everything else → overall_score=null,
      paper's raw aggregate preserved in task_scores.reported_avg.
    - Attribution decision table per candidates.schema.json row_type.
    - Collapse cross-check: cited_baseline rows whose cited_paper IS
      the method's own paper are dropped when the original paper's
      extraction already has the canonical row, and demoted to
      third_party otherwise.

    NOT applied here (for the refine LLM):
    - Eligibility filter (junk labels, ablation variants)
    - Dedup across papers
    - Cross-benchmark identity consistency
    - Notes composition

    Returns (candidates, stats).
    """
    extractions = _load_all_extractions()
    row_index = _build_row_index(extractions)

    candidates: list[dict] = []
    stats = {
        "extractions_total": 0,
        "papers_empty": 0,
        "papers_with_scores": 0,
        "rows_total": 0,
        "rows_dropped_collapse": 0,
        "rows_match_no_kept_null": 0,
        "rows_drop_empty_after_conversion": 0,
        "rows_first_party": 0,
        "rows_third_party": 0,
    }

    for aid, ext in extractions.items():
        stats["extractions_total"] += 1
        citing_url = _citing_url(aid)
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
                name = m.get("name_in_paper", "")
                protocol = m.get("protocol") or {}
                match = protocol.get("matches_standard", "unknown")
                scores_raw = m.get("scores") or {}
                suite_scores = _to_plain_scores(scores_raw.get("suite_scores"))
                task_scores = _to_plain_scores(scores_raw.get("task_scores"))
                raw_overall = scores_raw.get("overall_score")

                # Classify row first so we can short-circuit drops.
                row_type, reported_paper, effective_cited = _classify_row(
                    citing_url=citing_url,
                    is_score_original=m.get("is_score_original", "unknown"),
                    model_paper=m.get("model_paper"),
                    cited_paper=m.get("cited_paper"),
                    benchmark=benchmark,
                    name_in_paper=name,
                    row_index=row_index,
                )
                if row_type == "drop":
                    stats["rows_dropped_collapse"] += 1
                    continue

                # Arithmetic / protocol gate.
                if match == "yes":
                    overall = _compute_overall(benchmark, suite_scores, task_scores)
                    # Fallback: if the benchmark has no aggregation rule,
                    # trust the paper's raw overall.
                    if overall is None and _aggregation_rules().get(benchmark) is None:
                        if isinstance(raw_overall, (int, float)):
                            overall = raw_overall
                else:
                    overall = None
                    if match == "no":
                        stats["rows_match_no_kept_null"] += 1

                # reported_avg recovery. Any case where we did not end up
                # with a ranked overall_score but the paper did report
                # one → preserve that number in task_scores.reported_avg
                # so the row survives the empty-score gate with
                # overall_score=null. Applies to match='no'/'partial'/
                # 'unknown' as well as match='yes' with partial-component
                # data that couldn't satisfy the aggregation rule.
                if (
                    overall is None
                    and isinstance(raw_overall, (int, float))
                    and "reported_avg" not in task_scores
                    and "reported_avg" not in suite_scores
                ):
                    task_scores = {**task_scores, "reported_avg": raw_overall}

                if overall is None and not suite_scores and not task_scores:
                    stats["rows_drop_empty_after_conversion"] += 1
                    continue

                weight_type = m.get("weight_type")
                if weight_type not in ("shared", "finetuned"):
                    weight_type = "shared"

                candidates.append(
                    {
                        "benchmark": benchmark,
                        "name_in_paper": name,
                        "params": m.get("params"),
                        "weight_type": weight_type,
                        "overall_score": overall,
                        "suite_scores": suite_scores,
                        "task_scores": task_scores,
                        "reported_paper": reported_paper,
                        "reported_table": scores_raw.get("reported_table"),
                        "protocol_match": match,
                        "protocol_rationale": protocol.get("rationale", ""),
                        "is_score_original": m.get("is_score_original", "unknown"),
                        "model_paper": m.get("model_paper"),
                        "cited_paper": effective_cited,
                        "row_type": row_type,
                    }
                )
                if row_type == "first_party":
                    stats["rows_first_party"] += 1
                else:
                    stats["rows_third_party"] += 1

    return candidates, stats


def _print_stats(stats: dict) -> None:
    print(
        f"Extractions scanned: {stats['extractions_total']}\n"
        f"  papers with scores:              {stats['papers_with_scores']}\n"
        f"  papers empty (cited, no scores): {stats['papers_empty']}\n"
        f"Model rows processed: {stats['rows_total']}\n"
        f"  collapse → dropped (canonical row in original): {stats['rows_dropped_collapse']}\n"
        f"  match=no kept with null overall:                {stats['rows_match_no_kept_null']}\n"
        f"  dropped (no score after conv):                  {stats['rows_drop_empty_after_conversion']}\n"
        f"  kept first_party:                               {stats['rows_first_party']}\n"
        f"  kept third_party:                               {stats['rows_third_party']}"
    )


# ---------------------------------------------------------------------------
# LLM step: fuzzy decisions on pre-built candidates
# ---------------------------------------------------------------------------


def _benchmark_rules(benchmark: str) -> str:
    """Return the md body for one benchmark (frontmatter stripped)."""
    path = BENCHMARKS_DIR / f"{benchmark}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3 :].strip()
    return text


def _build_system_prompt(benchmark: str, rules: str) -> str:
    return f"""You are the PRECISION stage of a two-stage VLA leaderboard pipeline.

Your objective at this stage is precision, not recall. An upstream
EXTRACT stage optimized for recall and surfaced every candidate that
could belong on the leaderboard; you keep only the canonical ones.
Apply filters aggressively — when in doubt, drop. A small leaderboard
of canonical entries beats a large one with ablation junk.

Candidate rows are at `{CANDIDATES_PATH}`. Field semantics and per-row
invariants are defined in `{CANDIDATES_SCHEMA_PATH}` — read it, and
follow the `row_type` dispatch in the description of that field
exactly.

You do not have paper access at this stage. Rely on the candidate
fields. `overall_score` and `model_paper` are authoritative — pass
them through unchanged.

## Your job (in order)

1. **Drop ineligible rows.** Any candidate whose `name_in_paper` is
   still a generic label ("Ours", "Baseline", "(b)", "variant X",
   "Ablation") is unattributable — drop. Also drop variants whose only
   differentiator is quantization (INT4, AWQ), PEFT (LoRA), training
   stage ("w/o pretrain"), or hyperparameter sweep. When in doubt
   about whether a row is a genuine method vs an ablation, drop.

2. **Dedup within (model, benchmark, reported_paper).** Keep the row
   with the richest scores. Distinct reported_paper values always
   remain distinct entries.

3. **Assign `model`, `display_name`, `reported_paper`** per the
   row_type rules in the schema. Keep these consistent across a
   method's first-party entries.

4. **Compose `notes`** from `protocol_rationale` (trim long ones).
   Append origin context when useful — non-standard subset details,
   training budget, architecture class. When `cited_paper` is a
   non-arxiv URL, mention it in notes. Never leave notes blank or
   boilerplate.

## Benchmark scope: {benchmark}

{rules}

## Output

Write a JSON array of leaderboard entries to the output path specified
in the user message. Each entry matches `{LEADERBOARD_SCHEMA_PATH}`:

- Copy `name_in_paper` verbatim from the candidate.
- `curated_by = "opus 4.6"` (or the model you are).
- `date_added = "{date.today().isoformat()}"`.
- Do not emit a top-level wrapper — write the array directly.

Report what you dropped and why when you finish.
"""


def _refine_one_benchmark(
    benchmark: str,
    candidates: list[dict],
    output_path: Path,
    model: str,
    timeout: int,
) -> list[dict] | None:
    """Run the LLM refine step for a single benchmark's candidates.

    Returns the list of leaderboard entries on success (possibly empty
    if the LLM intentionally kept nothing). Returns None on LLM failure,
    so the caller can distinguish "LLM said drop everything" from "LLM
    crashed" and avoid wiping existing entries in the latter case.
    """
    bm_candidates = [c for c in candidates if c["benchmark"] == benchmark]
    if not bm_candidates:
        return None

    scratch_in = REFINE_LOGS_DIR / f"candidates_{benchmark}.json"
    scratch_out = REFINE_LOGS_DIR / f"out_{benchmark}.json"
    REFINE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(bm_candidates, indent=2, ensure_ascii=False) + "\n")
    scratch_in.write_text(json.dumps(bm_candidates, indent=2, ensure_ascii=False) + "\n")

    rules = _benchmark_rules(benchmark)
    system_prompt = _build_system_prompt(benchmark, rules)
    user_msg = (
        f"Refine {len(bm_candidates)} candidates for benchmark '{benchmark}'. "
        f"Read them from {CANDIDATES_PATH}. Write a JSON array of "
        f"leaderboard entries to {scratch_out}."
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
        # Restrict to Claude Code native tools; block MCP servers and
        # user skills that might delegate to outside knowledge sources.
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--add-dir",
        str(REFINE_LOGS_DIR.resolve()),
        "--add-dir",
        str(CANDIDATES_PATH.parent.resolve()),
    ]

    log_path = REFINE_LOGS_DIR / f"refine_{benchmark}_{date.today().isoformat()}.log"
    print(f"  [{benchmark}] {len(bm_candidates)} candidates → LLM ({model})...")
    result = subprocess.run(cmd, input=user_msg, capture_output=True, text=True, timeout=timeout)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        print(f"  [{benchmark}] claude exit {result.returncode}: {result.stderr[:300]}")
        return None

    if not scratch_out.exists():
        print(f"  [{benchmark}] no output file written")
        return None
    try:
        entries = json.loads(scratch_out.read_text())
    except json.JSONDecodeError as e:
        print(f"  [{benchmark}] invalid JSON output: {e}")
        return None
    if not isinstance(entries, list):
        print(f"  [{benchmark}] output is not an array (got {type(entries).__name__})")
        return None
    print(f"  [{benchmark}] LLM produced {len(entries)} entries")
    return entries


def _merge_leaderboard(
    new_entries: list[dict],
    benchmarks_touched: list[str],
    output_path: Path,
) -> None:
    """Merge per-benchmark results into leaderboard.json.

    Entries for benchmarks_touched are replaced; entries for other
    benchmarks are preserved from the existing file. Results are sorted
    by (benchmark, model).
    """
    existing: list[dict] = []
    if output_path.exists():
        data = json.loads(output_path.read_text())
        existing = data.get("results", [])
    kept = [e for e in existing if e.get("benchmark") not in benchmarks_touched]
    merged = kept + new_entries
    merged.sort(key=lambda r: (r.get("benchmark", ""), r.get("model", "")))
    out = {"last_updated": date.today().isoformat(), "results": merged}
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


def refine(
    model: str = "opus",
    benchmark: str | None = None,
    output: Path = LEADERBOARD_PATH,
    timeout: int = 7200,
) -> None:
    print("Stage 1: building candidates from extractions...")
    candidates, stats = build_candidates(benchmark_filter=benchmark)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n")
    _print_stats(stats)
    print(f"Wrote {CANDIDATES_PATH}")

    if not candidates:
        print("No candidates to refine. Exiting.")
        return

    # Group by benchmark and run LLM per-benchmark. Only benchmarks
    # whose LLM stage ran to completion are passed to the merge step;
    # on LLM failure, the existing leaderboard entries for that
    # benchmark are preserved rather than wiped to nothing.
    benchmarks = sorted({c["benchmark"] for c in candidates})
    print(f"\nStage 2: refining {len(benchmarks)} benchmark(s) with {model}...")
    all_entries: list[dict] = []
    touched: list[str] = []
    for bm in benchmarks:
        entries = _refine_one_benchmark(bm, candidates, output, model=model, timeout=timeout)
        if entries is None:
            print(f"  [{bm}] LLM step did not produce output — preserving existing entries")
            continue
        touched.append(bm)
        all_entries.extend(entries)

    _merge_leaderboard(all_entries, touched, output)
    print(
        f"\nDone: {output} refreshed {len(touched)} benchmark(s) "
        f"({len(all_entries)} new entries); {len(benchmarks) - len(touched)} preserved"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Refine raw extractions into leaderboard.json.", add_completion=False)


@app.command()
def main(
    output: Annotated[Path, typer.Option("-o", help="Output path.")] = LEADERBOARD_PATH,
    benchmark: Annotated[Optional[str], typer.Option(help="Only refine this benchmark.")] = None,
    model: Annotated[str, typer.Option(help="Claude model for the fuzzy stage.")] = "opus",
    timeout: Annotated[int, typer.Option(help="Per-benchmark LLM timeout in seconds.")] = 7200,
) -> None:
    """Refine extractions into leaderboard.json (python pre-step + per-benchmark LLM stage)."""
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
