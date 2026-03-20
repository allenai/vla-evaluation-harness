#!/usr/bin/env python3
"""Validate results.json against the JSON schema and check score ranges."""

import argparse
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results.json"
SCHEMA_PATH = DATA_DIR / "schema.json"
CITATIONS_PATH = DATA_DIR / "citations.json"

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def canonical_json(data: dict) -> str:
    """Return the canonical JSON serialization for results data."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def validate_schema(data: dict, schema: dict) -> list[str]:
    """Validate data against JSON schema. Returns list of error messages."""
    try:
        import jsonschema
    except ImportError:
        print("WARNING: jsonschema not installed, skipping schema validation")
        print("  Install with: uv pip install jsonschema")
        return []

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
    """Validate papers_reviewed entries."""
    errors = []
    for bm_key, bm in data["benchmarks"].items():
        no_results = bm.get("papers_reviewed", [])
        seen = set()
        for arxiv_id in no_results:
            if not ARXIV_ID_RE.match(arxiv_id):
                errors.append(f"benchmarks.{bm_key}.papers_reviewed: '{arxiv_id}' is not a valid arxiv ID")
            if arxiv_id in seen:
                errors.append(f"benchmarks.{bm_key}.papers_reviewed: duplicate '{arxiv_id}'")
            seen.add(arxiv_id)
    return errors


def validate_citations(data: dict) -> list[str]:
    """Validate that citations.json exists, is non-empty, and covers all arxiv papers in results."""
    errors = []
    if not CITATIONS_PATH.exists():
        errors.append("citations.json not found — run update_citations.py --fetch")
        return errors

    citations = json.loads(CITATIONS_PATH.read_text())
    papers = citations.get("papers", {})
    if not papers:
        errors.append("citations.json has no entries — run update_citations.py --fetch")
        return errors

    # Check coverage: every arxiv-based model_paper/source_paper should have a citation entry
    missing = []
    for r in data["results"]:
        for field in ("model_paper", "source_paper"):
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate results.json against schema and leaderboard rules.")
    parser.add_argument("results_file", nargs="?", default=None, help="Path to results.json (default: auto-detect)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix sort order and canonical formatting")
    args = parser.parse_args()

    results_path = Path(args.results_file) if args.results_file else RESULTS_PATH
    raw_text = results_path.read_text()
    data = json.loads(raw_text)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    if args.fix:
        data["results"].sort(key=lambda r: (r["benchmark"], r["model"]))
        fixed_text = canonical_json(data)
        if fixed_text != raw_text:
            results_path.write_text(fixed_text)
            raw_text = fixed_text
            print(f"Fixed: sorted results and wrote canonical format to {results_path}")
        else:
            print("Nothing to fix: already sorted and canonical.")

    errors = (
        validate_schema(data, schema)
        + validate_score_ranges(data)
        + validate_sort_and_format(data, raw_text)
        + validate_official_leaderboard_policy(data)
        + validate_papers_reviewed(data)
        + validate_citations(data)
    )

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
