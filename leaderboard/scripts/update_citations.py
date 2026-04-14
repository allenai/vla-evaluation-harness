#!/usr/bin/env python3
"""Fetch per-model-paper citation counts from Semantic Scholar.

Reads all unique arxiv IDs from model_paper / reported_paper fields in leaderboard.json,
fetches citation counts from the Semantic Scholar batch API, and writes them to citations.json.

Usage:
    python update_citations.py              # use cached values, skip fetching
    python update_citations.py --fetch      # fetch live counts from Semantic Scholar
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "data" / "leaderboard.json"
CITATIONS_PATH = Path(__file__).parent.parent / "data" / "citations.json"

S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"

# Optional: increases rate limit but not required for batch citationCount queries
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
BATCH_SIZE = 500  # S2 batch API limit


def extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url or "")
    return m.group(1) if m else None


def fetch_citation_counts_batch(arxiv_ids: list[str]) -> dict[str, int | None]:
    """Fetch citation counts for multiple arxiv papers using the batch API."""
    all_counts: dict[str, int | None] = {}

    for i in range(0, len(arxiv_ids), BATCH_SIZE):
        batch = arxiv_ids[i : i + BATCH_SIZE]
        ids = [f"ARXIV:{aid}" for aid in batch]
        body = json.dumps({"ids": ids}).encode()
        url = f"{S2_BATCH_API}?fields=citationCount,externalIds"

        headers = {"Content-Type": "application/json", "User-Agent": "VLA-Leaderboard/1.0"}
        if API_KEY:
            headers["x-api-key"] = API_KEY

        req = urllib.request.Request(url, data=body, headers=headers)
        results = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    results = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    print(f"  Rate limited on batch {i // BATCH_SIZE + 1}, retrying in 10s...")
                    time.sleep(10)
                    req = urllib.request.Request(url, data=body, headers=headers)
                    continue
                print(f"  Batch API error (batch {i // BATCH_SIZE + 1}): {e}")
                break
            except (urllib.error.URLError, OSError) as e:
                print(f"  Batch API error (batch {i // BATCH_SIZE + 1}): {e}")
                break

        if results is None:
            continue

        for paper, aid in zip(results, batch):
            if paper is not None:
                all_counts[aid] = paper.get("citationCount")

        print(f"  Fetched batch {i // BATCH_SIZE + 1} ({len(batch)} papers)")
        if len(arxiv_ids) > BATCH_SIZE:
            time.sleep(1)  # rate limit between batches

    return all_counts


def load_cached() -> dict:
    if CITATIONS_PATH.exists():
        return json.loads(CITATIONS_PATH.read_text())
    return {"last_updated": None, "papers": {}}


def main():
    parser = argparse.ArgumentParser(description="Update per-paper citation counts.")
    parser.add_argument("--fetch", action="store_true", help="Fetch live counts from Semantic Scholar API")
    args = parser.parse_args()

    results_data = json.loads(RESULTS_PATH.read_text())
    cached = load_cached()
    cached_papers = cached.get("papers", {})

    # Collect all unique arxiv IDs from model_paper and reported_paper
    arxiv_ids = set()
    for r in results_data["results"]:
        for field in ("model_paper", "reported_paper"):
            aid = extract_arxiv_id(r.get(field))
            if aid:
                arxiv_ids.add(aid)

    print(f"Found {len(arxiv_ids)} unique arxiv papers across {len(results_data['results'])} results")

    papers = {}
    if args.fetch:
        sorted_ids = sorted(arxiv_ids)
        fetched = fetch_citation_counts_batch(sorted_ids)
        for aid in sorted_ids:
            if aid in fetched and fetched[aid] is not None:
                papers[aid] = fetched[aid]
            elif aid in cached_papers:
                papers[aid] = cached_papers[aid]  # keep cached on fetch failure
    else:
        for aid in sorted(arxiv_ids):
            if aid in cached_papers:
                papers[aid] = cached_papers[aid]

    output = {
        "last_updated": date.today().isoformat() if (args.fetch and fetched) else cached.get("last_updated"),
        "papers": papers,
    }
    CITATIONS_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(papers)} citation entries to {CITATIONS_PATH}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        changed = sum(1 for aid, count in papers.items() if cached_papers.get(aid) != count)
        summary = f"- {changed} of {len(papers)} papers updated"
        with open(output_path, "a") as f:
            f.write(f"citations_summary={summary}\n")


if __name__ == "__main__":
    main()
