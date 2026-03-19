#!/usr/bin/env python3
"""Compute coverage stats, optionally fetching citation counts from Semantic Scholar.

Without --fetch: updates entry counts from results.json, keeps cached citing_papers.
With --fetch: also fetches live citation counts from Semantic Scholar batch API.

Writes coverage data to leaderboard/data/coverage.json for display on the leaderboard site.
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "data" / "results.json"
COVERAGE_PATH = Path(__file__).parent.parent / "data" / "coverage.json"

S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"


def extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url or "")
    return m.group(1) if m else None


def fetch_citation_counts_batch(arxiv_ids: list[str]) -> dict[str, int | None]:
    """Fetch citation counts for multiple arxiv papers in a single batch request."""
    if not arxiv_ids:
        return {}
    ids = [f"ARXIV:{aid}" for aid in arxiv_ids]
    body = json.dumps({"ids": ids}).encode()
    url = f"{S2_BATCH_API}?fields=citationCount,externalIds"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "VLA-Leaderboard/1.0",
        },
    )
    results = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                results = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                print("  Rate limited, retrying in 10s...")
                time.sleep(10)
                continue
            print(f"  Batch API error: {e}")
            return {}
        except (urllib.error.URLError, OSError) as e:
            print(f"  Batch API error: {e}")
            return {}

    if results is None:
        return {}

    counts = {}
    for paper, aid in zip(results, arxiv_ids):
        if paper is not None:
            counts[aid] = paper.get("citationCount")
        else:
            counts[aid] = None
    return counts


def load_cached_coverage() -> dict:
    """Load existing coverage.json for cached citing_papers values."""
    if COVERAGE_PATH.exists():
        return json.loads(COVERAGE_PATH.read_text())
    return {}


def main():
    parser = argparse.ArgumentParser(description="Update leaderboard coverage stats.")
    parser.add_argument("--fetch", action="store_true", help="Fetch live citation counts from Semantic Scholar API")
    args = parser.parse_args()

    results_data = json.loads(RESULTS_PATH.read_text())
    benchmarks = results_data["benchmarks"]
    results = results_data["results"]
    cached = load_cached_coverage()
    cached_bm = cached.get("benchmarks", {})

    result_counts = Counter(r["benchmark"] for r in results)

    # Total unique papers reviewed across all benchmarks
    all_reviewed = set()
    for bm_info in benchmarks.values():
        all_reviewed.update(bm_info.get("papers_reviewed", []))
    papers_reviewed = len(all_reviewed)

    # Collect arxiv IDs and batch-fetch if requested
    bm_arxiv = {}
    for bm_key, bm_info in benchmarks.items():
        aid = extract_arxiv_id(bm_info.get("paper_url"))
        if aid:
            bm_arxiv[bm_key] = aid

    fetched_counts: dict[str, int | None] = {}
    if args.fetch and bm_arxiv:
        print(f"Batch-fetching citations for {len(bm_arxiv)} benchmarks...")
        fetched_counts = fetch_citation_counts_batch(list(bm_arxiv.values()))

    coverage = {
        "last_updated": results_data.get("last_updated", "unknown"),
        "total_models": len({r["model"] for r in results}),
        "total_results": len(results),
        "total_papers_reviewed": papers_reviewed,
        "benchmarks": {},
    }

    for bm_key, bm_info in benchmarks.items():
        arxiv_id = bm_arxiv.get(bm_key)
        citing_count = cached_bm.get(bm_key, {}).get("citing_papers")

        if args.fetch and arxiv_id and arxiv_id in fetched_counts:
            citing_count = fetched_counts[arxiv_id] or citing_count

        n_results = result_counts.get(bm_key, 0)
        n_papers = len(bm_info.get("papers_reviewed", []))
        coverage["benchmarks"][bm_key] = {
            "display_name": bm_info["display_name"],
            "arxiv_id": arxiv_id,
            "citing_papers": citing_count,
            "leaderboard_entries": n_results,
            "papers_reviewed": n_papers,
        }
        status = f"{citing_count} citations" if citing_count else "no data"
        source = "fetched" if args.fetch and arxiv_id else "cached"
        print(f"  {bm_key}: {n_results} entries, {n_papers} papers reviewed, {status} ({source})")

    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2) + "\n")
    print(f"\nCoverage written to {COVERAGE_PATH}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        total_citing = sum(b.get("citing_papers") or 0 for b in coverage["benchmarks"].values())
        total_reviewed = coverage["total_papers_reviewed"]
        pct = f"{total_reviewed / total_citing * 100:.1f}%" if total_citing else "N/A"
        lines = [
            f"- {len(coverage['benchmarks'])} benchmarks, {coverage['total_models']} models, {coverage['total_results']} results",
            f"- {total_reviewed} / {total_citing} citing papers reviewed ({pct})",
        ]
        with open(output_path, "a") as f:
            f.write("coverage_summary<<GITHUB_OUTPUT_EOF\n")
            f.write("\n".join(lines) + "\n")
            f.write("GITHUB_OUTPUT_EOF\n")


if __name__ == "__main__":
    main()
