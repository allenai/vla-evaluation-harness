#!/usr/bin/env python3
"""Validate leaderboard.json against the JSON schema and check score ranges.

Runs a layered set of checks:

1. JSON schema (draft 7) — structural validity
2. Score ranges — per-benchmark metric bounds, required score presence, duplicate keys
3. Sort and canonical format — deterministic on-disk representation
4. Official leaderboard policy — API-only benchmarks only accept `-api` entries
5. Citations coverage — every arxiv paper referenced has a citation entry
6. Arithmetic consistency — `overall_score` matches the benchmark's aggregation rule
7. Forbidden overall_score — benchmarks that must always use `null`
8. Cross-entry identity — same `model` key must carry consistent params/display_name/model_paper
9. Required notes (warnings) — per-benchmark mandatory note keywords

Errors block CI. Warnings are printed but do not affect exit code unless `--strict`.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import jsonschema


def canonical_json(data: dict) -> str:
    """Return the canonical JSON serialization used by leaderboard.json."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"
BENCHMARKS_PATH = DATA_DIR / "benchmarks.json"
SCHEMA_PATH = DATA_DIR / "leaderboard.schema.json"
BENCHMARKS_SCHEMA_PATH = DATA_DIR / "benchmarks.schema.json"
CITATIONS_PATH = DATA_DIR / "citations.json"


# ---------------------------------------------------------------------------
# Arithmetic aggregation rules
# ---------------------------------------------------------------------------

# Tolerance for arithmetic consistency — accommodates paper-side rounding
# and the occasional curator approximation (when a paper reports only the
# average and the individual suites were filled in from prose).
# LIBERO/LIBERO-Plus/etc. typically report to 1 decimal and have small suite
# counts, so 0.5 absorbs both rounding and mild approximation. LIBERO-Pro is
# noisier (20 cells, larger rounding cascade) so it gets a looser tolerance.
ARITHMETIC_TOLERANCE = 0.5
LIBERO_PRO_TOLERANCE = 1.0

# For each benchmark that has an exact aggregation rule, describe:
#   container:    "suite_scores" or "task_scores"
#   required_keys: list of keys that must all be present for overall_score to be non-null
#   excluded_keys: keys present in the registry but NOT aggregated into overall_score
#
# A benchmark not in this dict has no arithmetic rule enforced (either the
# overall_score is defined per-paper or the benchmark reports null by policy).
ARITHMETIC_RULES: dict[str, dict] = {
    "libero": {
        "container": "suite_scores",
        "required_keys": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "excluded_keys": ["libero_90"],
        "tolerance": ARITHMETIC_TOLERANCE,
    },
    "libero_plus": {
        "container": "suite_scores",
        "required_keys": ["camera", "robot", "language", "light", "background", "noise", "layout"],
        "excluded_keys": [],
        "tolerance": ARITHMETIC_TOLERANCE,
    },
    "libero_pro": {
        "container": "suite_scores",
        # 20 core cells: {goal, spatial, long, object} × {ori, obj, pos, sem, task}
        "required_keys": [
            f"{suite}_{pert}"
            for suite in ("goal", "spatial", "long", "object")
            for pert in ("ori", "obj", "pos", "sem", "task")
        ],
        # env cells are optional per CONTRIBUTING.md
        "excluded_keys": ["goal_env", "spatial_env", "long_env", "object_env"],
        "tolerance": LIBERO_PRO_TOLERANCE,
    },
    "libero_mem": {
        "container": "task_scores",
        "required_keys": [f"T{i}" for i in range(1, 11)],
        "excluded_keys": [],
        "tolerance": ARITHMETIC_TOLERANCE,
    },
    "mikasa": {
        "container": "task_scores",
        "required_keys": ["ShellGameTouch", "InterceptMedium", "RememberColor3", "RememberColor5", "RememberColor9"],
        "excluded_keys": [],
        "tolerance": ARITHMETIC_TOLERANCE,
    },
    "vlabench": {
        "container": "suite_scores",
        "required_keys": ["in_dist_PS", "cross_category_PS", "commonsense_PS", "semantic_instruction_PS"],
        "excluded_keys": [],
        "tolerance": ARITHMETIC_TOLERANCE,
    },
}

# Benchmarks whose overall_score must ALWAYS be null (per CONTRIBUTING.md).
# Any non-null overall_score for these is a hard error.
FORBIDDEN_OVERALL: set[str] = {"simpler_env", "robotwin_v2"}

# Per-benchmark required notes keywords. Each rule is a list of alternatives;
# at least one alternative keyword (case-insensitive substring match) must
# appear in `notes` or an "ok" marker otherwise the check emits a warning.
REQUIRED_NOTES: dict[str, list[list[str]]] = {
    "robotwin_v2": [["protocol a", "protocol b"]],
    "kinetix": [["d=", "delay", "execution_horizon", "inference_delay"]],
    "robocasa": [["demos", "task"]],
    "maniskill2": [["averaging", "averaged", "average method", "mean method", "unknown"]],
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _get_score_container(result: dict, container: str) -> dict:
    return result.get(container) or {}


# ---------------------------------------------------------------------------
# Existing checks (unchanged behavior)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Aggregation & consistency checks
# ---------------------------------------------------------------------------


def validate_arithmetic_consistency(data: dict) -> list[str]:
    """Verify overall_score matches the benchmark's aggregation rule.

    Applies only to benchmarks in ARITHMETIC_RULES. For each entry with
    overall_score != None:

      - If ALL required keys are present: compute the mean and compare to
        overall_score within per-benchmark tolerance.
      - If NO required keys are present: skip (the paper only reported the
        aggregate without a per-suite breakdown, which is permitted).
      - If SOME required keys are present: error — partial breakdown is
        ambiguous and must be either completed or set to null with the
        aggregate stored in task_scores.reported_avg.
    """
    errors = []
    for i, r in enumerate(data["results"]):
        rule = ARITHMETIC_RULES.get(r["benchmark"])
        if rule is None:
            continue
        if r.get("overall_score") is None:
            # Non-standard protocol — arithmetic check does not apply.
            continue

        container = _get_score_container(r, rule["container"])
        required = rule["required_keys"]
        present = [k for k in required if k in container]

        prefix = f"results[{i}] {r['model']}/{r['benchmark']}"

        if len(present) == 0:
            # Paper only reported the aggregate; no per-key breakdown.
            # Nothing to check arithmetically.
            continue

        if len(present) < len(required):
            missing = [k for k in required if k not in container]
            errors.append(
                f"{prefix}: overall_score={r['overall_score']} set with partial "
                f"{rule['container']} breakdown; missing: {missing}"
            )
            continue

        expected = _mean([container[k] for k in required])
        diff = abs(r["overall_score"] - expected)
        if diff > rule["tolerance"]:
            errors.append(
                f"{prefix}: overall_score {r['overall_score']} does not match "
                f"mean of {len(required)} required {rule['container']} keys "
                f"({expected:.2f}); diff={diff:.2f} > tolerance {rule['tolerance']}"
            )

    return errors


def validate_forbidden_overall(data: dict) -> list[str]:
    """Some benchmarks must always have overall_score == null."""
    errors = []
    for i, r in enumerate(data["results"]):
        if r["benchmark"] in FORBIDDEN_OVERALL and r.get("overall_score") is not None:
            errors.append(
                f"results[{i}] {r['model']}/{r['benchmark']}: overall_score must "
                f"always be null for {r['benchmark']}, got {r['overall_score']}"
            )
    return errors


def validate_cross_entry_consistency(data: dict) -> list[str]:
    """Same `model` key across benchmarks must carry consistent identity fields.

    Mismatched params/display_name/model_paper for the same model are almost
    always a copy-paste bug. API-synced entries (curated_by suffix `-api`) are
    excluded because different APIs may legitimately use different canonical
    identifiers for the same checkpoint.
    """
    errors = []
    by_model: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, r in enumerate(data["results"]):
        if r.get("curated_by", "").endswith("-api"):
            continue
        by_model[r["model"]].append((i, r))

    for model, entries in by_model.items():
        if len(entries) < 2:
            continue
        for field in ("display_name", "params", "model_paper"):
            values = {r.get(field) for _, r in entries}
            if len(values) > 1:
                # Point at the first row for each distinct value, for easy diffing.
                witnesses = {}
                for idx, r in entries:
                    v = r.get(field)
                    witnesses.setdefault(v, idx)
                witness_str = ", ".join(f"results[{idx}]={v!r}" for v, idx in witnesses.items())
                errors.append(f"model '{model}' has inconsistent {field} across entries: {witness_str}")

    return errors


def validate_required_notes(data: dict) -> list[str]:
    """Warnings: benchmarks whose notes should mention certain markers.

    Returns a list of warning strings. Callers decide whether to treat them
    as errors (via --strict).
    """
    warnings = []
    for i, r in enumerate(data["results"]):
        rules = REQUIRED_NOTES.get(r["benchmark"])
        if not rules:
            continue
        notes = (r.get("notes") or "").lower()
        for alternatives in rules:
            if not any(kw in notes for kw in alternatives):
                warnings.append(
                    f"results[{i}] {r['model']}/{r['benchmark']}: notes missing "
                    f"required marker (one of {alternatives})"
                )
    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate leaderboard.json against schema and leaderboard rules.")
    parser.add_argument(
        "leaderboard_file", nargs="?", default=None, help="Path to leaderboard.json (default: auto-detect)"
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix sort order and canonical formatting")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (required notes) as errors",
    )
    args = parser.parse_args()

    results_path = Path(args.leaderboard_file) if args.leaderboard_file else LEADERBOARD_PATH
    raw_text = results_path.read_text()
    data = json.loads(raw_text)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    # Load and validate benchmarks registry
    benchmarks = json.loads(BENCHMARKS_PATH.read_text()) if BENCHMARKS_PATH.exists() else {}
    if BENCHMARKS_SCHEMA_PATH.exists():
        bm_schema = json.loads(BENCHMARKS_SCHEMA_PATH.read_text())
        bm_errors = validate_schema(benchmarks, bm_schema)
        if bm_errors:
            print(f"benchmarks.json schema errors: {len(bm_errors)}")
            for e in bm_errors:
                print(f"  - {e}")

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
    warnings: list[str] = []

    errors += validate_schema(data, schema)
    errors += validate_sort_and_format(data, raw_text)

    # Inject benchmarks for validators that need them (score_ranges, official_policy, etc.)
    data["benchmarks"] = benchmarks
    errors += validate_score_ranges(data)
    errors += validate_official_leaderboard_policy(data)
    errors += validate_citations(data)
    errors += validate_arithmetic_consistency(data)
    errors += validate_forbidden_overall(data)
    errors += validate_cross_entry_consistency(data)
    warnings += validate_required_notes(data)

    if args.strict:
        errors += warnings
        warnings = []

    if warnings:
        print(f"WARNINGS: {len(warnings)} found:")
        for w in warnings:
            print(f"  - {w}")
        print()

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
