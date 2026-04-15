#!/usr/bin/env python3
"""Build benchmarks.json from per-benchmark leaderboard/benchmarks/*.md frontmatter.

Source of truth: each benchmark's `.md` file.
  - YAML frontmatter holds ALL authored structured config (display_name,
    metric, suites, tasks, aggregation, detail_notes, paper_url, etc).
  - Markdown body holds the LLM-consumed protocol prose (Standard /
    Scoring / Checks / Methodology).

`benchmarks.json` is a **build artifact** — never edit it by hand. This
script merges all md frontmatters plus the `papers_reviewed` lists (owned
by `update_coverage.py`) into the final JSON, validates against
`benchmarks.schema.json`, and writes the output.

Usage::

    python build_benchmarks_json.py            # build
    python build_benchmarks_json.py --check    # exit 1 on drift or validation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BENCHMARKS_DIR = ROOT / "benchmarks"
BENCHMARKS_JSON_PATH = DATA_DIR / "benchmarks.json"
BENCHMARKS_SCHEMA_PATH = DATA_DIR / "benchmarks.schema.json"

# Canonical field order for each benchmark in the generated JSON.
FIELD_ORDER = [
    "display_name",
    "paper_url",
    "metric",
    "suites",
    "tasks",
    "aggregation",
    "official_leaderboard",
    "expand_suites",
    "avg_position",
    "avg_label",
    "papers_reviewed",
    "detail_notes",
]


def _parse_frontmatter(path: Path) -> dict:
    """Return {**frontmatter} from a `---`-delimited md file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter (no leading `---`)")
    end = text.find("---", 3)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter (no closing `---`)")
    try:
        data = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: YAML parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    return data


def load_existing_papers_reviewed() -> dict[str, list[str]]:
    """Read the current benchmarks.json and return papers_reviewed per benchmark.

    This field is owned by `update_coverage.py` and preserved across builds.
    """
    if not BENCHMARKS_JSON_PATH.exists():
        return {}
    data = json.loads(BENCHMARKS_JSON_PATH.read_text())
    return {k: list(v.get("papers_reviewed", [])) for k, v in data.items()}


def _ordered(entry: dict) -> dict:
    """Return entry with keys in FIELD_ORDER (unknown keys trail alphabetically)."""
    out = {k: entry[k] for k in FIELD_ORDER if k in entry}
    for k in sorted(entry):
        if k not in out:
            out[k] = entry[k]
    return out


def build(preserve_papers_reviewed: bool = True) -> dict:
    """Assemble benchmarks.json from md sources. Raises on YAML/schema errors."""
    papers_by_key = load_existing_papers_reviewed() if preserve_papers_reviewed else {}
    out: dict[str, dict] = {}

    for f in sorted(BENCHMARKS_DIR.glob("*.md")):
        if f.stem == "_global":
            continue
        fm = _parse_frontmatter(f)
        bm_key = fm.pop("benchmark", None) or f.stem
        if bm_key in out:
            raise ValueError(f"{f}: duplicate benchmark key {bm_key!r}")
        if preserve_papers_reviewed and bm_key in papers_by_key:
            fm.setdefault("papers_reviewed", papers_by_key[bm_key])
        out[bm_key] = _ordered(fm)

    ordered = {k: out[k] for k in sorted(out)}

    schema = json.loads(BENCHMARKS_SCHEMA_PATH.read_text())
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(ordered), key=lambda e: list(e.absolute_path))
    if errors:
        msgs = []
        for err in errors:
            path = list(err.absolute_path)
            if path:
                bm = path[0]
                field = ".".join(str(p) for p in path[1:]) or "<root>"
                loc = f"leaderboard/benchmarks/{bm}.md"
                msgs.append(f"{loc}: {field}: {err.message}")
            else:
                msgs.append(f"<root>: {err.message}")
        raise ValueError("schema validation failed:\n  " + "\n  ".join(msgs))

    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmarks.json from benchmarks/*.md frontmatter.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if benchmarks.json diverges from the md sources (no writes).",
    )
    args = parser.parse_args()

    try:
        generated = build()
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    serialized = json.dumps(generated, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        existing = BENCHMARKS_JSON_PATH.read_text() if BENCHMARKS_JSON_PATH.exists() else ""
        if existing != serialized:
            print(
                f"ERROR: benchmarks.json is out of sync with md sources. "
                f"Run `python {Path(__file__).name}` to rebuild."
            )
            return 1
        print(f"OK: benchmarks.json in sync with {len(generated)} benchmarks.")
        return 0

    BENCHMARKS_JSON_PATH.write_text(serialized)
    print(f"Wrote {BENCHMARKS_JSON_PATH} ({len(generated)} benchmarks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
