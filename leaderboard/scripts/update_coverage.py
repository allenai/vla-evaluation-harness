#!/usr/bin/env python3
"""Compute coverage stats, optionally fetching citation counts from Semantic Scholar.

Without --fetch: updates entry counts from leaderboard.json + benchmarks.json,
keeps cached citing_papers / arxiv_citing_papers.
With --fetch: paginates the Semantic Scholar /citations endpoint per benchmark
to populate both:

- `citing_papers`: total citations (arXiv + non-arXiv publications)
- `arxiv_citing_papers`: subset with an arXiv preprint — this is the
  realistic denominator for coverage since non-arXiv papers cannot be
  reviewed via the arxiv reading pipeline.

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
from datetime import date
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "data" / "leaderboard.json"
BENCHMARKS_PATH = Path(__file__).parent.parent / "data" / "benchmarks.json"
COVERAGE_PATH = Path(__file__).parent.parent / "data" / "coverage.json"

S2_CITATIONS_URL = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/citations"


def extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url or "")
    return m.group(1) if m else None


def fetch_citation_counts(arxiv_id: str, limit: int = 1000) -> tuple[int, int] | None:
    """Paginate /citations for one benchmark paper.

    Returns (total_citing, arxiv_citing). None on persistent failure.
    Total counts every citation; arxiv-only counts those whose `externalIds`
    carry an ArXiv entry.
    """
    headers = {"Accept": "application/json", "User-Agent": "VLA-Leaderboard/1.0"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    total = 0
    arxiv_only = 0
    offset = 0
    while True:
        url = f"{S2_CITATIONS_URL.format(arxiv_id=arxiv_id)}?fields=externalIds&limit={limit}&offset={offset}"
        req = urllib.request.Request(url, headers=headers)
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                print(f"  {arxiv_id}: HTTP {e.code}")
                return None
            except (urllib.error.URLError, OSError) as e:
                if attempt < 2:
                    time.sleep(5)
                    continue
                print(f"  {arxiv_id}: {e}")
                return None
        if data is None:
            return None
        for item in data.get("data", []):
            total += 1
            ext = (item.get("citingPaper") or {}).get("externalIds") or {}
            if ext.get("ArXiv"):
                arxiv_only += 1
        if data.get("next") is None:
            break
        offset = data["next"]
        time.sleep(1)
    return total, arxiv_only


def load_cached_coverage() -> dict:
    """Load existing coverage.json for cached citing_papers values."""
    if COVERAGE_PATH.exists():
        return json.loads(COVERAGE_PATH.read_text())
    return {"last_updated": None, "benchmarks": {}}


def main():
    parser = argparse.ArgumentParser(description="Update leaderboard coverage stats.")
    parser.add_argument("--fetch", action="store_true", help="Fetch live citation counts from Semantic Scholar API")
    args = parser.parse_args()

    results_data = json.loads(RESULTS_PATH.read_text())
    benchmarks = json.loads(BENCHMARKS_PATH.read_text())
    results = results_data["results"]
    cached = load_cached_coverage()
    cached_bm = cached.get("benchmarks", {})

    result_counts = Counter(r["benchmark"] for r in results)

    # Total unique papers reviewed across all benchmarks
    all_reviewed = set()
    for bm_info in benchmarks.values():
        all_reviewed.update(bm_info.get("papers_reviewed", []))
    papers_reviewed = len(all_reviewed)

    bm_arxiv: dict[str, str] = {}
    for bm_key, bm_info in benchmarks.items():
        aid = extract_arxiv_id(bm_info.get("paper_url"))
        if aid:
            bm_arxiv[bm_key] = aid

    fetched: dict[str, tuple[int, int]] = {}
    if args.fetch and bm_arxiv:
        print(f"Fetching citations for {len(bm_arxiv)} benchmarks via /citations endpoint...")
        for bm_key, aid in bm_arxiv.items():
            result = fetch_citation_counts(aid)
            if result is not None:
                fetched[bm_key] = result
                total, arxiv_only = result
                print(f"  {bm_key}: total={total} arxiv={arxiv_only}")

    coverage = {
        "last_updated": date.today().isoformat() if (args.fetch and fetched) else cached.get("last_updated"),
        "total_models": len({r["model"] for r in results}),
        "total_results": len(results),
        "total_papers_reviewed": papers_reviewed,
        "benchmarks": {},
    }

    for bm_key, bm_info in benchmarks.items():
        arxiv_id = bm_arxiv.get(bm_key)
        cached_entry = cached_bm.get(bm_key, {})
        citing_count = cached_entry.get("citing_papers")
        arxiv_citing_count = cached_entry.get("arxiv_citing_papers")

        if bm_key in fetched:
            citing_count, arxiv_citing_count = fetched[bm_key]

        n_results = result_counts.get(bm_key, 0)
        n_papers = len(bm_info.get("papers_reviewed", []))
        coverage["benchmarks"][bm_key] = {
            "display_name": bm_info["display_name"],
            "arxiv_id": arxiv_id,
            "citing_papers": citing_count,
            "arxiv_citing_papers": arxiv_citing_count,
            "leaderboard_entries": n_results,
            "papers_reviewed": n_papers,
        }
        if citing_count is not None:
            arxiv_suffix = f" ({arxiv_citing_count} arxiv)" if arxiv_citing_count is not None else ""
            status = f"{citing_count} citations{arxiv_suffix}"
        else:
            status = "no data"
        source = "fetched" if bm_key in fetched else "cached"
        print(f"  {bm_key}: {n_results} entries, {n_papers} papers reviewed, {status} ({source})")

    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2) + "\n")
    print(f"\nCoverage written to {COVERAGE_PATH}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        total_citing = sum(
            b.get("arxiv_citing_papers") or b.get("citing_papers") or 0 for b in coverage["benchmarks"].values()
        )
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
