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

2. LLM agent (opus) — per-benchmark cross-paper consolidation: canonical
   bibkey assignment, dedup across papers, notes composition, and
   resolution-failure drops (generic placeholders that could not be
   mapped to a canonical method). Content-eligibility judgments
   (ablation / sweep variants) belong in extract, not here.

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
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Annotated, Optional

import jsonschema
import typer

from authors import fetch_surnames

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTRACTIONS_JSON_PATH = DATA_DIR / "extractions.json"
PAPERS_DIR = ROOT / ".cache" / "papers"
BENCHMARKS_DIR = ROOT / "benchmarks"
BENCHMARKS_JSON_PATH = DATA_DIR / "benchmarks.json"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"
LEADERBOARD_SCHEMA_PATH = DATA_DIR / "leaderboard.schema.json"
CANDIDATES_SCHEMA_PATH = DATA_DIR / "candidates.schema.json"
CANDIDATES_PATH = ROOT / ".cache" / "refine_candidates.json"
REFINE_LOGS_DIR = ROOT / ".cache" / "refine_logs"
ARXIV_AUTHORS_CACHE = ROOT / ".cache" / "arxiv_authors.json"

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


_BIBKEY_PART_RE = re.compile(r"^([a-z]+)(\d{2,4})([a-z].*)$")


def _expand_bibkey_year(bibkey: str) -> str:
    """Force 4-digit year in a bibkey part like 'kim24openvla' → 'kim2024openvla'.

    Leaves 4-digit bibkeys alone. Years 15-30 map to 20XX (covers modern
    ML research); 0-14 would be ambiguous (could be 2000s) so we leave
    them as-is to avoid misreading a substring like 'gr00tn1'.
    """
    m = _BIBKEY_PART_RE.match(bibkey)
    if not m:
        return bibkey
    author, year, rest = m.group(1), m.group(2), m.group(3)
    if len(year) == 4:
        return bibkey  # already canonical
    yy = int(year)
    if 15 <= yy <= 30:
        return f"{author}20{year}{rest}"
    return bibkey


def _normalize_bibkey(model: str) -> str:
    """Apply 4-digit-year normalization to each `__`-joined bibkey part."""
    if not model:
        return model
    return "__".join(_expand_bibkey_year(part) for part in model.split("__"))


@functools.cache
def _benchmarks_registry() -> dict[str, dict]:
    """Full benchmarks.json registry, cached once per process."""
    return json.loads(BENCHMARKS_JSON_PATH.read_text())


def _aggregation_rules() -> dict[str, dict | str]:
    """Return {benchmark: "forbidden" | {container, keys}} from benchmarks.json."""
    return {k: v["aggregation"] for k, v in _benchmarks_registry().items() if "aggregation" in v}


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


def _score_key_suffixes(benchmark: str) -> tuple[str, ...]:
    """Return allowed suffixes for benchmarks that constrain task/suite keys.

    Read from ``benchmarks.json``'s per-benchmark ``score_key_suffixes``
    field. Empty tuple means the benchmark accepts unsuffixed keys
    (the common case). Explicit data; no central hardcode in this script.
    """
    return tuple(_benchmarks_registry().get(benchmark, {}).get("score_key_suffixes") or ())


def _reported_avg_key(benchmark: str, suite_scores: dict, task_scores: dict) -> str:
    """Return the key under which to store a paper's reported aggregate.

    For most benchmarks this is the reserved sentinel 'reported_avg'. For
    benchmarks whose keys are protocol-suffixed (declared via
    benchmarks.json's ``score_key_suffixes``), extend the sentinel with
    the dominant suffix seen on sibling keys so the row validates.
    """
    allowed = _score_key_suffixes(benchmark)
    if not allowed:
        return "reported_avg"
    seen = [
        k.rsplit("_", 1)[-1]
        for k in list(task_scores) + list(suite_scores)
        if "_" in k and k.rsplit("_", 1)[-1] in allowed
    ]
    suffix = max(set(seen), key=seen.count) if seen else allowed[0]
    return f"reported_avg_{suffix}"


def _load_all_extractions() -> dict[str, dict]:
    """Load data/extractions.json (authoritative) into memory keyed by arxiv_id.

    Returns an empty dict if the file is missing — caller handles the
    "nothing to refine" case.
    """
    if not EXTRACTIONS_JSON_PATH.exists():
        return {}
    entries = json.loads(EXTRACTIONS_JSON_PATH.read_text())
    return {e["arxiv_id"]: e for e in entries if e.get("arxiv_id")}


def _iter_scored(ext: dict) -> list[tuple[str, list[dict]]]:
    """Return [(benchmark_key, models)] for each scored entry in ext.benchmarks."""
    out: list[tuple[str, list[dict]]] = []
    for bm_key, entry in (ext.get("benchmarks") or {}).items():
        if entry.get("status") == "scored":
            out.append((bm_key, entry.get("models") or []))
    return out


def _build_row_index(extractions: dict[str, dict]) -> set[tuple[str, str, str]]:
    """Index of (arxiv_id, benchmark, normalized name) for collapse cross-check."""
    idx: set[tuple[str, str, str]] = set()
    for aid, ext in extractions.items():
        for bm_key, models in _iter_scored(ext):
            for m in models:
                idx.add((aid, bm_key, _norm_name(m.get("name_in_paper", ""))))
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

    NOT applied here (handled by the refine LLM):
    - Canonical bibkey assignment
    - Dedup across papers
    - Cross-benchmark identity consistency
    - Notes composition
    - Resolution-failure drops (generic placeholders, unresolvable names)

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
        scored = _iter_scored(ext)
        if not scored:
            stats["papers_empty"] += 1
            continue
        stats["papers_with_scores"] += 1
        for benchmark, models in scored:
            if benchmark_filter and benchmark != benchmark_filter:
                continue
            for m in models:
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
                #
                # Some benchmarks (simpler_env) constrain task_score keys
                # with a protocol suffix — pick the dominant suffix from
                # neighbouring keys so the fallback remains schema-valid.
                reported_avg_key = _reported_avg_key(benchmark, suite_scores, task_scores)
                if (
                    overall is None
                    and isinstance(raw_overall, (int, float))
                    and reported_avg_key not in task_scores
                    and reported_avg_key not in suite_scores
                ):
                    task_scores = {**task_scores, reported_avg_key: raw_overall}

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

    # Pre-compute first-author surnames for every arxiv URL seen so the
    # LLM can build bibkeys as {surname}{year}{shortname} directly. The
    # on-disk cache persists across runs.
    unique_ids: set[str] = {
        aid
        for c in candidates
        for k in ("model_paper", "cited_paper", "reported_paper")
        if (aid := _arxiv_id_of(c.get(k))) is not None
    }
    surnames = fetch_surnames(sorted(unique_ids), ARXIV_AUTHORS_CACHE)
    for c in candidates:
        for k in ("model_paper", "cited_paper", "reported_paper"):
            aid = _arxiv_id_of(c.get(k))
            c[f"{k}_surname"] = surnames.get(aid) if aid is not None else None

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
    return f"""You are the CROSS-PAPER CONSOLIDATION stage of a two-stage VLA leaderboard pipeline.

The upstream EXTRACT stage already evaluated each candidate's content
eligibility per-paper: ablation variants, quantization / PEFT /
hyperparameter-sweep rows, and generic-label rows it could resolve have
all been handled there. **Trust extract's content judgments — do not
re-evaluate whether a row is an ablation or sweep variant.**

Your job is cross-paper consolidation that extract cannot do with
per-paper context alone: canonical bibkey assignment, dedup across
papers, presentation (display_name, notes), and dropping rows that
truly cannot be resolved to a canonical method.

Candidate rows are at `{CANDIDATES_PATH}`. Field semantics and per-row
invariants are defined in `{CANDIDATES_SCHEMA_PATH}` — read it, and
follow the `row_type` dispatch in the description of that field
exactly. `overall_score` is authoritative — pass it through unchanged.

## Paper access

Papers are at `{PAPERS_DIR}/<arxiv_id>/paper.md`. Open them when you need
to verify:
- **Method identity** — e.g., "pi0" vs "pi0-FAST" are different methods
  with different `model_paper`s (2410.24164 vs 2501.09747). If a row's
  `name_in_paper` mentions a variant or tokenizer suffix, open the
  citing paper and its references to confirm the correct `model_paper`.
- **Attribution** — distinguish "this paper independently reproduced X"
  from "this paper quotes X's numbers verbatim from another source".
  If two candidates with the same `name_in_paper` show byte-identical
  scores across different `reported_paper`s, read both — the later one
  is usually re-citing the earlier; drop the later row.
- **Generic-label resolution** — when `name_in_paper` is still generic
  (e.g., "Ours" survived extract), the paper's abstract or methods
  section names the method.

Do NOT use paper access to re-filter ablations, variants, or
hyperparameter sweeps. Extract already handled scope and eligibility.
Trust those judgments.

## Your job (in order)

1. **Assign `model`, `display_name`, `reported_paper`** per the
   row_type rules in the schema. Keep `model` (bibkey) consistent
   across a method's first-party and third-party entries so dedup
   works downstream.

   Build each bibkey part as `{{surname}}{{year}}{{shortname}}` from the
   pre-fetched fields on the candidate:
   - Method's own part (first_party, or the leading part of a composite
     third_party bibkey): use `model_paper_surname` + year from
     `model_paper` (20YY from the arxiv id prefix).
   - Citing-paper part of a composite third_party bibkey: use
     `cited_paper_surname` when `cited_paper` is an arxiv URL, else
     `reported_paper_surname`.
   Never emit a bibkey part of the form `{{shortname}}{{year}}` that
   omits the surname.

2. **Dedup within (model, benchmark, reported_paper).** Keep the row
   with the richest scores. Distinct reported_paper values remain
   distinct entries (same method reproduced in a different paper is a
   separate row, not a duplicate).

3. **Compose `notes`** from `protocol_rationale` (trim long ones).
   Append origin context when useful — non-standard subset details,
   training budget, architecture class. When `cited_paper` is a
   non-arxiv URL, mention it in notes. Never leave notes blank or
   boilerplate.

## Drop only on these grounds

- **Resolution failure**: `name_in_paper` remains a generic placeholder
  ("Ours", "Baseline", "(b)", "variant X", "Ablation") that you
  cannot map to a canonical method name using the candidate's
  `model_paper` / `cited_paper` URLs.
- **Cross-paper exact duplicate**: another kept row has the same
  `(model, benchmark, reported_paper)` and identical scores — drop the
  redundant one.

Do NOT drop based on adjectival prefixes ("Naive", "Simple", "Basic"),
hyperparameter labels ("(h=1)", "(k=4)"), or any content-level concern
that extract already filtered. When uncertain, KEEP.

## Benchmark scope: {benchmark}

{rules}

## Output

Write a JSON array of leaderboard entries to the output path specified
in the user message. Each entry matches `{LEADERBOARD_SCHEMA_PATH}`:

- Copy `name_in_paper` verbatim from the candidate.
- `curated_by = "opus 4.6"` (or the model you are).
- `date_added = "{date.today().isoformat()}"`.
- Do not emit a top-level wrapper — write the array directly.

Report what you kept, dropped, and why when you finish.
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
        f"leaderboard entries to {scratch_out}.\n\n"
        f"Authoritative reference for existing entries: {LEADERBOARD_PATH} "
        f"(filter by `benchmark == '{benchmark}'`). When a candidate maps "
        "to a method that already has a row there, reuse the existing "
        "row's `model` (bibkey), `model_paper`, `display_name`, `notes`, "
        "and `params` verbatim — unless a new candidate provides explicit "
        "evidence for a correction (a newer measurement, a better-resolved "
        "paper URL, etc.). Emit one entry per final leaderboard row, "
        "covering every method the candidates + existing rows cover; "
        "scores come from the candidates, not the existing file."
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
        # data/ holds leaderboard.json (existing-rows reference) and
        # the schema files. The LLM reads those to follow the contract
        # and reuse existing bibkeys / metadata verbatim.
        "--add-dir",
        str(DATA_DIR.resolve()),
        # Paper source — refine verifies method identity / attribution
        # against paper.md. Read-only use from LLM side.
        "--add-dir",
        str(PAPERS_DIR.resolve()),
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
    # Enforce 4-digit year in bibkeys deterministically — the LLM has a
    # stubborn bias toward 2-digit years (e.g. 'kim24openvla') that the
    # schema description alone doesn't fix reliably.
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("model"), str):
            entry["model"] = _normalize_bibkey(entry["model"])
    print(f"  [{benchmark}] LLM produced {len(entries)} entries")
    entries = _collapse_duplicate_signatures(entries, benchmark)
    entries = _fill_first_party_model_paper(entries, benchmark)
    entries = _fill_third_party_model_paper(entries, benchmark)
    return entries


def _fill_first_party_model_paper(entries: list[dict], benchmark: str) -> list[dict]:
    """Set ``model_paper := reported_paper`` for first_party rows where the LLM
    forgot to.

    Invariant: bibkey without a ``__`` separator denotes first_party —
    the method's own paper is the citing paper, so model_paper and
    reported_paper must be the same URL. Losing model_paper breaks
    downstream links and schema expectations; this post-pass restores
    the equality deterministically.
    """
    fixed = 0
    for e in entries:
        model = e.get("model") or ""
        if "__" in model:
            continue
        if e.get("model_paper") is not None:
            continue
        rp = e.get("reported_paper")
        if not rp:
            continue
        e["model_paper"] = rp
        fixed += 1
    if fixed:
        print(f"  [{benchmark}] filled model_paper on {fixed} first_party row(s)")
    return entries


_METHOD_BIBKEY_RE = re.compile(r"^([a-z]+)(\d{4})")


@functools.cache
def _author_cache_by_surname_year() -> dict[tuple[str, int], list[str]]:
    """Reverse index ``.cache/arxiv_authors.json`` as (surname, year) → [arxiv_ids]."""
    if not ARXIV_AUTHORS_CACHE.exists():
        return {}
    cache: dict[str, str | None] = json.loads(ARXIV_AUTHORS_CACHE.read_text())
    by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for aid, surname in cache.items():
        if not surname:
            continue
        m = re.match(r"^(\d{2})\d{2}\.", aid)
        if not m:
            continue
        year = 2000 + int(m.group(1))
        by_key[(surname, year)].append(aid)
    return dict(by_key)


def _fill_third_party_model_paper(entries: list[dict], benchmark: str) -> list[dict]:
    """Deterministically fill ``model_paper`` on third-party rows (``__`` bibkey).

    Two resolution strategies, in order:

    1. **Cross-reference**: if another row in this same refine run carries
       the same method bibkey with a resolved ``model_paper``, copy that
       URL. This catches the common case where a paper appears both as
       first-party (its own paper) and third-party (cited in other papers);
       the first-party row always has ``model_paper`` set, so siblings
       inherit it.
    2. **Surname/year lookup**: parse ``{surname}{year}{shortname}`` from
       the method bibkey and look up the unique arxiv id with that
       first-author surname and year in ``.cache/arxiv_authors.json``.
       Ambiguous hits (common surnames like ``wang`` / ``liu``) are left
       null — those need a manual override in ``manual_overrides.json``.

    First-party rows (no ``__``) are handled by
    :func:`_fill_first_party_model_paper` — this pass is the symmetric
    safety net for compound bibkeys.
    """
    # Pass 1: collect bibkey → arxiv_id from already-resolved rows.
    bib_to_aid: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        mp = e.get("model_paper")
        if not mp:
            continue
        aid = _arxiv_id_of(mp)
        if not aid:
            continue
        method_bib = (e.get("model") or "").split("__")[0]
        if method_bib:
            bib_to_aid[method_bib].add(aid)
    crossref = {b: next(iter(s)) for b, s in bib_to_aid.items() if len(s) == 1}

    by_sn_year = _author_cache_by_surname_year()

    fixed_crossref = 0
    fixed_lookup = 0
    for e in entries:
        model = e.get("model") or ""
        if "__" not in model:
            continue  # first-party is someone else's problem
        if e.get("model_paper") is not None:
            continue
        method_bib = model.split("__", 1)[0]

        aid = crossref.get(method_bib)
        if aid is not None:
            e["model_paper"] = f"https://arxiv.org/abs/{aid}"
            fixed_crossref += 1
            continue

        m = _METHOD_BIBKEY_RE.match(method_bib)
        if not m:
            continue
        cands = by_sn_year.get((m.group(1), int(m.group(2))), [])
        if len(cands) == 1:
            e["model_paper"] = f"https://arxiv.org/abs/{cands[0]}"
            fixed_lookup += 1

    if fixed_crossref or fixed_lookup:
        print(
            f"  [{benchmark}] filled model_paper on third-party rows: "
            f"crossref={fixed_crossref} surname_year={fixed_lookup}"
        )
    return entries


def _score_signature(entry: dict) -> tuple:
    """Stable hashable fingerprint of an entry's numeric content."""
    return (
        entry.get("overall_score"),
        tuple(sorted((entry.get("suite_scores") or {}).items())),
        tuple(sorted((entry.get("task_scores") or {}).items())),
    )


def _collapse_duplicate_signatures(entries: list[dict], benchmark: str) -> list[dict]:
    """Collapse rows that share `(name_in_paper, score_signature)`.

    When multiple rows report identical numbers under the same method
    name, they are near-certainly re-citations of a single original
    measurement rather than independent reproductions. Keep one
    canonical row per group and drop the rest, noting the alternative
    `reported_paper` URLs on the survivor so provenance isn't lost.

    Canonical-row preference (in order):
      1. Row with ``reported_paper == model_paper`` (the method's own
         paper — the original source of record).
      2. Row with the earliest arxiv id in ``reported_paper`` (earlier
         publication is more likely the original).
      3. First row in input order.
    """
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        key = (e.get("name_in_paper", ""), _score_signature(e))
        groups[key].append(e)

    kept: list[dict] = []
    collapsed_count = 0
    for rows in groups.values():
        if len(rows) == 1:
            kept.append(rows[0])
            continue

        def _rank(e: dict) -> tuple:
            rp = e.get("reported_paper")
            mp = e.get("model_paper")
            # Prefer canonical: reported_paper is the method's own paper.
            is_canonical = 0 if rp and mp and rp == mp else 1
            rp_aid = _arxiv_id_of(rp)
            # Earlier arxiv id (smaller YY.MMMMM lex order works because
            # the id format is zero-padded per month).
            return (is_canonical, rp_aid or "9999.99999")

        rows_sorted = sorted(rows, key=_rank)
        winner = rows_sorted[0]
        losers = rows_sorted[1:]
        loser_rps: list[str] = [rp for e in losers if isinstance(rp := e.get("reported_paper"), str)]
        if loser_rps:
            extra = f" Same score also reported in: {', '.join(loser_rps)}."
            winner["notes"] = (winner.get("notes") or "").rstrip() + extra
        kept.append(winner)
        collapsed_count += len(losers)

    if collapsed_count:
        print(
            f"  [{benchmark}] collapsed {collapsed_count} duplicate-signature row(s) "
            f"into {len(entries) - collapsed_count} canonical entries"
        )
    return kept


@functools.cache
def _leaderboard_schema() -> dict:
    return json.loads(LEADERBOARD_SCHEMA_PATH.read_text())


PRESERVE_ON_NULL_FIELDS: tuple[str, ...] = (
    "model_paper",
    "display_name",
    "params",
    "notes",
    "weight_type",
)


def preserve_on_null(new_entries: list[dict], old_entries: list[dict]) -> int:
    """Restore preserve-listed fields on new rows from matching old rows when null.

    Match key: ``(benchmark, model)``. For each new row with a matching
    old row, any field in :data:`PRESERVE_ON_NULL_FIELDS` that's null on
    the new row and non-null on the old row is copied over from the old
    row. Curator-set metadata (a filled ``model_paper``, a corrected
    ``notes``) survives the LLM re-emitting a row with that field null.
    Scores are never preserved — the LLM's number is the authoritative
    latest extraction.

    Mutates ``new_entries`` in place and returns the restore count for
    logging.
    """
    old_by_key = {(e.get("benchmark"), e.get("model")): e for e in old_entries}
    restored = 0
    for new in new_entries:
        key = (new.get("benchmark"), new.get("model"))
        old = old_by_key.get(key)
        if old is None:
            continue
        for f in PRESERVE_ON_NULL_FIELDS:
            if new.get(f) is None and old.get(f) is not None:
                new[f] = old[f]
                restored += 1
    return restored


def _merge_leaderboard(
    new_entries: list[dict],
    benchmarks_touched: list[str],
    output_path: Path,
) -> None:
    """Merge per-benchmark results into leaderboard.json.

    Entries for benchmarks_touched are replaced; entries for other
    benchmarks are preserved from the existing file. Results are sorted
    by (benchmark, model).

    Before writing, the merged payload is schema-validated against
    ``leaderboard.schema.json`` so pipeline-level regressions fail here
    instead of surfacing much later via CI's ``validate.py``. Curator-
    set metadata fields are preserved on null via
    :func:`_preserve_on_null` — the LLM receives the existing rows in
    its prompt as authoritative reference, but we still guard against
    occasional null drift in a deterministic post-step.
    """
    existing: list[dict] = []
    if output_path.exists():
        data = json.loads(output_path.read_text())
        existing = data.get("results", [])
    touched_existing = [e for e in existing if e.get("benchmark") in benchmarks_touched]
    restored = preserve_on_null(new_entries, touched_existing)
    if restored:
        print(f"  preserve-on-null: restored {restored} field(s) from existing rows")
    kept = [e for e in existing if e.get("benchmark") not in benchmarks_touched]
    merged = kept + new_entries
    merged.sort(key=lambda r: (r.get("benchmark", ""), r.get("model", "")))
    out = {"last_updated": date.today().isoformat(), "results": merged}
    jsonschema.validate(out, _leaderboard_schema())
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
