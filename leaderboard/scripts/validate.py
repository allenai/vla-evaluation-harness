#!/usr/bin/env python3
"""Validate leaderboard.json against the JSON schema and check score ranges."""

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"
BENCHMARKS_PATH = DATA_DIR / "benchmarks.json"
LEADERBOARD_SCHEMA_PATH = DATA_DIR / "leaderboard.schema.json"
BENCHMARKS_SCHEMA_PATH = DATA_DIR / "benchmarks.schema.json"
CITATIONS_PATH = DATA_DIR / "citations.json"

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def canonical_json(data: dict) -> str:
    """Return the canonical JSON serialization used by leaderboard.json."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def validate_schema(data: dict, schema: dict) -> list[str]:
    """Validate data against JSON schema. Returns list of error messages."""
    validator = jsonschema.Draft7Validator(schema)
    return [f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in validator.iter_errors(data)]


def validate_score_ranges(data: dict) -> list[str]:
    """Check that all scores fall within their benchmark's declared range."""
    errors = []
    benchmarks = data["benchmarks"]

    seen_pairs: set[tuple[str, str, str]] = set()

    for i, result in enumerate(data["results"]):
        prefix = f"results[{i}]"

        # Check weight_type is valid
        wt = result.get("weight_type")
        if wt not in ("shared", "finetuned"):
            errors.append(f"{prefix}: weight_type '{wt}' must be 'shared' or 'finetuned'")

        # Check benchmark exists
        bm_key = result["benchmark"]
        if bm_key not in benchmarks:
            errors.append(f"{prefix}: benchmark '{bm_key}' not in benchmarks registry")
            continue

        bm = benchmarks[bm_key]
        metric = bm["metric"]
        lo, hi = metric["range"]

        # Check overall score (null is allowed when suite_scores provide the detail)
        score = result.get("overall_score")
        if score is not None and not (lo <= score <= hi):
            errors.append(f"{prefix}: overall_score {score} outside range [{lo}, {hi}]")

        # Every entry must have at least one score
        has_score = score is not None or result.get("suite_scores") or result.get("task_scores")
        if not has_score:
            errors.append(f"{prefix}: no score (overall_score, suite_scores, or task_scores required)")

        # Non-standard protocol entries (overall_score=null) may use task/suite
        # keys outside the declared set, so only validate keys for standard entries
        is_standard = score is not None

        # Check suite_scores: values must be in range, keys must match declared suites
        declared_suites = set(bm.get("suites", []))
        for suite, val in (result.get("suite_scores") or {}).items():
            if is_standard and declared_suites and suite not in declared_suites:
                errors.append(f"{prefix}: suite_scores.{suite} not in declared suites {sorted(declared_suites)}")
            if not (0 <= val <= 100):
                errors.append(f"{prefix}: suite_scores.{suite} = {val} outside range [0, 100]")

        # Check task_scores: values must be in range, keys must match declared tasks
        declared_tasks = set(bm.get("tasks", []))
        for task, val in (result.get("task_scores") or {}).items():
            if is_standard and declared_tasks and task not in declared_tasks:
                errors.append(f"{prefix}: task_scores.{task} not in declared tasks {sorted(declared_tasks)}")
            if not (0 <= val <= 100):
                errors.append(f"{prefix}: task_scores.{task} = {val} outside range [0, 100]")
            if bm_key == "simpler_env" and not (task.endswith("_vm") or task.endswith("_va")):
                errors.append(
                    f"{prefix}: simpler_env task_scores key '{task}' "
                    "must end with _vm or _va to indicate evaluation protocol"
                )

        # Check no duplicate (model, benchmark, weight_type)
        pair = (result["model"], bm_key, result.get("weight_type", "shared"))
        if pair in seen_pairs:
            errors.append(f"{prefix}: duplicate entry for {pair}")
        seen_pairs.add(pair)

    return errors


def validate_sort_and_format(data: dict, raw_text: str) -> list[str]:
    """Check that results are sorted by (benchmark, model) and file uses canonical format."""
    errors = []
    results = data["results"]
    pairs = [(r["benchmark"], r["model"]) for r in results]
    if pairs != sorted(pairs):
        errors.append("results array is not sorted by (benchmark, model) — run with --fix to auto-sort")

    expected = canonical_json(data)
    if raw_text != expected and pairs == sorted(pairs):
        errors.append("file format does not match canonical style (indent=2, trailing newline) — run with --fix")

    return errors


def validate_official_leaderboard_policy(data: dict) -> list[str]:
    """Benchmarks with official_leaderboard must only have API-synced entries."""
    errors = []
    for bm_key, bm in data["benchmarks"].items():
        if not bm.get("official_leaderboard"):
            continue
        for i, r in enumerate(data["results"]):
            if r["benchmark"] == bm_key and not r["curated_by"].endswith("-api"):
                errors.append(
                    f"results[{i}]: {r['model']}/{bm_key} curated_by '{r['curated_by']}' "
                    f"but {bm_key} has official_leaderboard — only API-synced entries allowed"
                )
    return errors


def validate_papers_reviewed(data: dict) -> list[str]:
    """Validate papers_reviewed entries inside benchmarks registry."""
    errors = []
    for bm_key, bm in data["benchmarks"].items():
        reviewed = bm.get("papers_reviewed", [])
        seen = set()
        for arxiv_id in reviewed:
            if not ARXIV_ID_RE.match(arxiv_id):
                errors.append(f"benchmarks.{bm_key}.papers_reviewed: '{arxiv_id}' is not a valid arxiv ID")
            if arxiv_id in seen:
                errors.append(f"benchmarks.{bm_key}.papers_reviewed: duplicate '{arxiv_id}'")
            seen.add(arxiv_id)
    return errors


def validate_citations(data: dict) -> list[str]:
    """Validate that citations.json exists and covers every arxiv paper in results.

    When the leaderboard has zero results (e.g. a fresh rebuild before any
    refine has run), an empty citations file is acceptable.
    """
    errors = []
    if not CITATIONS_PATH.exists():
        errors.append("citations.json not found — run update_citations.py --fetch")
        return errors

    citations = json.loads(CITATIONS_PATH.read_text())
    papers = citations.get("papers", {})
    if not papers:
        if data.get("results"):
            errors.append("citations.json has no entries — run update_citations.py --fetch")
        return errors

    # Check coverage: every arxiv-based model_paper/reported_paper should have a citation entry
    missing = []
    for r in data["results"]:
        for field in ("model_paper", "reported_paper"):
            url = r.get(field)
            m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url or "")
            if m and m.group(1) not in papers:
                missing.append(m.group(1))
    missing = sorted(set(missing))
    if missing:
        errors.append(
            f"citations.json missing {len(missing)} arxiv papers: {', '.join(missing[:10])}"
            + (" ..." if len(missing) > 10 else "")
        )

    return errors


def validate_scale_sanity(data: dict) -> list[str]:
    """Catch probable scale leaks (e.g. paper 0-1 values copied verbatim to a 0-100 %-benchmark).

    Flag when a row has ≥2 non-zero numeric scores across task_scores and
    suite_scores, and ALL of them are ≤ 1.0, on a benchmark whose declared
    metric.range upper bound is ≥ 10. A single sub-1.0 task score is valid
    (hard task failure); multiple together almost always means the paper's
    0-1 scale was not converted to %.
    """
    errors = []
    for i, r in enumerate(data["results"]):
        bm = data["benchmarks"].get(r["benchmark"], {})
        rule_range = bm.get("metric", {}).get("range", [0, 100])
        if rule_range[1] < 10:
            # Benchmark itself runs on a small scale (e.g. CALVIN 0-5) — cannot use this heuristic
            continue
        values = []
        for score_dict in (r.get("task_scores") or {}, r.get("suite_scores") or {}):
            for k, v in score_dict.items():
                if k == "reported_avg":
                    continue
                if isinstance(v, (int, float)) and v > 0:
                    values.append(v)
        if len(values) >= 2 and all(v <= 1.0 for v in values):
            errors.append(
                f"results[{i}] ({r['model']}/{r['benchmark']}): probable scale leak — "
                f"all {len(values)} non-zero scores are ≤ 1.0 on a benchmark with range {rule_range}. "
                f"The paper likely reports values on a 0-1 scale; multiply by 100 before emitting."
            )
    return errors


def validate_aggregation_rules(data: dict) -> list[str]:
    """Check each entry's overall_score against the benchmark's aggregation rule.

    Two shapes of rule live in benchmarks.json (sourced from md frontmatter):

    - ``"forbidden"`` — overall_score MUST be null (e.g. simpler_env, robotwin_v2).
    - ``{"container": "suite_scores"|"task_scores", "keys": [...]}`` — if all
      required keys are present on the entry, overall_score must match their
      arithmetic mean within a small tolerance. Entries missing any key are
      skipped (their overall_score may legitimately be null for non-standard
      protocols).

    Tolerance is 0.25: theoretical worst case for independent 1-decimal
    rounding is about 0.10, and papers frequently round sub-scores and
    overall at different steps, producing legitimate ±0.2 disagreements.
    Anything beyond ±0.25 indicates a real data inconsistency.
    """
    errors = []
    tolerance = 0.25
    for i, r in enumerate(data["results"]):
        rule = data["benchmarks"].get(r["benchmark"], {}).get("aggregation")
        if rule is None:
            continue
        score = r.get("overall_score")
        prefix = f"results[{i}] ({r['model']}/{r['benchmark']})"
        if rule == "forbidden":
            if score is not None:
                errors.append(f"{prefix}: overall_score must be null (benchmark aggregation: forbidden)")
            continue
        container = r.get(rule["container"]) or {}
        if not all(k in container for k in rule["keys"]):
            continue
        if score is None:
            continue
        mean = round(sum(container[k] for k in rule["keys"]) / len(rule["keys"]), 1)
        if abs(score - mean) > tolerance:
            errors.append(
                f"{prefix}: overall_score {score} disagrees with mean of "
                f"{rule['container']}{rule['keys']} = {mean} (diff {score - mean:+.2f}, tolerance ±{tolerance})"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate leaderboard.json against schema and leaderboard rules.")
    parser.add_argument(
        "leaderboard_file", nargs="?", default=None, help="Path to leaderboard.json (default: auto-detect)"
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix sort order and canonical formatting")
    args = parser.parse_args()

    results_path = Path(args.leaderboard_file) if args.leaderboard_file else LEADERBOARD_PATH
    raw_text = results_path.read_text()
    data = json.loads(raw_text)

    with open(LEADERBOARD_SCHEMA_PATH) as f:
        leaderboard_schema = json.load(f)

    # Load and validate the benchmarks registry separately
    benchmarks = json.loads(BENCHMARKS_PATH.read_text())
    benchmarks_errors: list[str] = []
    if BENCHMARKS_SCHEMA_PATH.exists():
        with open(BENCHMARKS_SCHEMA_PATH) as f:
            benchmarks_schema = json.load(f)
        benchmarks_errors = validate_schema(benchmarks, benchmarks_schema)

    if args.fix:
        data["results"].sort(key=lambda r: (r["benchmark"], r["model"]))
        fixed_text = canonical_json(data)
        if fixed_text != raw_text:
            results_path.write_text(fixed_text)
            raw_text = fixed_text
            print(f"Fixed: sorted results and wrote canonical format to {results_path}")
        else:
            print("Nothing to fix: already sorted and canonical.")

    errors: list[str] = []
    errors += validate_schema(data, leaderboard_schema)
    errors += [f"benchmarks.json: {e}" for e in benchmarks_errors]
    errors += validate_sort_and_format(data, raw_text)

    # Inject benchmarks for validators that need the registry (score_ranges,
    # official_policy, papers_reviewed). Done AFTER --fix so the benchmarks
    # registry never leaks back into leaderboard.json.
    data["benchmarks"] = benchmarks
    errors += validate_score_ranges(data)
    errors += validate_scale_sanity(data)
    errors += validate_aggregation_rules(data)
    errors += validate_official_leaderboard_policy(data)
    errors += validate_papers_reviewed(data)
    errors += validate_citations(data)

    if errors:
        print(f"FAILED: {len(errors)} error(s) found:")
        for e in errors:
            print(f"  - {e}")
        return 1

    n_models = len({r["model"] for r in data["results"]})
    n_benchmarks = len(data["benchmarks"])
    n_results = len(data["results"])
    print(f"OK: {n_results} results across {n_models} models and {n_benchmarks} benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
