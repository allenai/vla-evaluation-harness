#!/usr/bin/env python3
"""Refresh citation pools from Semantic Scholar and derive coverage stats.

Default mode: paginate Semantic Scholar's /citations endpoint per benchmark
and write both

- data/scan_results.json — per-benchmark `all_citing_ids` pool (the list
  consumed by extract.py's `run --from-scan`),
- data/coverage.json — counts for the leaderboard site.

`--check` mode: don't hit the S2 API; just re-derive data/coverage.json
from whatever is already on disk (existing scan_results.json pools and
the extractions cache / packed file). Use this after `extract.py run`
to reflect newly-extracted papers in the coverage bar, and in CI where
regenerating from committed data is enough.

`papers_reviewed` is the scan pool ∩ extractions — a paper counts as
reviewed for benchmark Y iff it is in Y's pool and an extraction exists
for it (cache preferred, packed fallback).
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

LEADERBOARD_ROOT = Path(__file__).parent.parent
DATA_DIR = LEADERBOARD_ROOT / "data"
RESULTS_PATH = DATA_DIR / "leaderboard.json"
BENCHMARKS_PATH = DATA_DIR / "benchmarks.json"
COVERAGE_PATH = DATA_DIR / "coverage.json"
EXTRACTIONS_CACHE_DIR = LEADERBOARD_ROOT / ".cache" / "extractions"
EXTRACTIONS_PACKED_PATH = DATA_DIR / "extractions.json"
SCAN_RESULTS_PATH = DATA_DIR / "scan_results.json"

S2_CITATIONS_URL = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/citations"


def extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url or "")
    return m.group(1) if m else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_citations(arxiv_id: str, limit: int = 1000) -> tuple[int, list[str]] | None:
    """Paginate /citations for one benchmark paper.

    Returns `(total_citing, arxiv_citing_ids)` — the full citation count
    and the ordered list of arxiv IDs among the citers. None on persistent
    failure. `total_citing` includes non-arxiv citations (those cannot be
    processed by the extract pipeline but are still the real citation count).
    """
    headers = {"Accept": "application/json", "User-Agent": "VLA-Leaderboard/1.0"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    total = 0
    arxiv_ids: list[str] = []
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
            aid = ext.get("ArXiv")
            if aid:
                arxiv_ids.append(aid)
        if data.get("next") is None:
            break
        offset = data["next"]
        time.sleep(1)
    return total, arxiv_ids


def load_cached_coverage() -> dict:
    """Load existing coverage.json for cached citing_papers values."""
    if COVERAGE_PATH.exists():
        return json.loads(COVERAGE_PATH.read_text())
    return {"last_updated": None, "benchmarks": {}}


def load_scan_results() -> dict:
    """Load existing data/scan_results.json. Returns the full file shape."""
    if SCAN_RESULTS_PATH.exists():
        try:
            return json.loads(SCAN_RESULTS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"scanned_at": None, "benchmarks": {}}


def _extracted_arxiv_ids() -> set[str]:
    """All arxiv IDs with an extraction record (cache preferred, packed fallback)."""
    if EXTRACTIONS_CACHE_DIR.exists():
        return {p.stem for p in EXTRACTIONS_CACHE_DIR.glob("*.json")}
    if EXTRACTIONS_PACKED_PATH.exists():
        try:
            return {rec["arxiv_id"] for rec in json.loads(EXTRACTIONS_PACKED_PATH.read_text()) if rec.get("arxiv_id")}
        except json.JSONDecodeError:
            return set()
    return set()


def load_reviewed_by_benchmark(benchmarks: dict, scan_by_bm: dict) -> dict[str, set[str]]:
    """Derive {benchmark_key → {arxiv_id}} from scan pools ∩ extractions.

    A paper counts as reviewed for benchmark Y iff an extraction exists
    for it AND it is in Y's reviewable pool (Y's citing papers per
    scan_results, or Y's own benchmark paper). This reflects the
    CLI-driven workflow — a paper pulled in by libero_mem's scan is
    credited to libero_mem, not to every benchmark the LLM happened to
    see rules for.
    """
    extracted = _extracted_arxiv_ids()
    reviewed: dict[str, set[str]] = {}
    for bm_key, bm_info in benchmarks.items():
        pool = set(scan_by_bm.get(bm_key, {}).get("all_citing_ids", []))
        bm_arxiv = extract_arxiv_id(bm_info.get("paper_url", ""))
        if bm_arxiv:
            pool.add(bm_arxiv)
        reviewed[bm_key] = pool & extracted
    return reviewed


def main():
    parser = argparse.ArgumentParser(
        description="Refresh S2 citation pools (scan_results.json) and coverage.json.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Skip the S2 fetch and only re-derive coverage.json from existing on-disk state.",
    )
    parser.add_argument(
        "--benchmark",
        help="Refresh only this benchmark's pool. Ignored with --check.",
    )
    args = parser.parse_args()
    do_fetch = not args.check

    results_data = json.loads(RESULTS_PATH.read_text())
    benchmarks = json.loads(BENCHMARKS_PATH.read_text())
    results = results_data["results"]
    cached_cov = load_cached_coverage().get("benchmarks", {})
    scan_file = load_scan_results()
    scan_by_bm: dict[str, dict] = dict(scan_file.get("benchmarks", {}))

    result_counts = Counter(r["benchmark"] for r in results)

    bm_arxiv: dict[str, str] = {}
    for bm_key, bm_info in benchmarks.items():
        aid = extract_arxiv_id(bm_info.get("paper_url"))
        if aid:
            bm_arxiv[bm_key] = aid

    if do_fetch and bm_arxiv:
        if args.benchmark and args.benchmark not in bm_arxiv:
            print(f"benchmark '{args.benchmark}' not in benchmarks.json or lacks an arxiv paper_url")
            return 1
        targets = {args.benchmark: bm_arxiv[args.benchmark]} if args.benchmark else bm_arxiv
        print(f"Fetching citations for {len(targets)} benchmark(s) via /citations endpoint...")
        for bm_key, aid in targets.items():
            result = fetch_citations(aid)
            if result is None:
                continue
            total, arxiv_citing_ids = result
            pool = sorted(set(arxiv_citing_ids) | {aid})  # include the benchmark paper itself
            scan_by_bm[bm_key] = {
                "arxiv_id": aid,
                "display_name": benchmarks[bm_key].get("display_name", bm_key),
                "citing_papers": total,
                "arxiv_citing_papers": len(arxiv_citing_ids),
                "all_citing_ids": pool,
            }
            print(f"  {bm_key}: total={total} arxiv={len(arxiv_citing_ids)} pool={len(pool)}")

        SCAN_RESULTS_PATH.write_text(
            json.dumps({"scanned_at": _now_iso(), "benchmarks": scan_by_bm}, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"Wrote {SCAN_RESULTS_PATH}")

    reviewed_by_bm = load_reviewed_by_benchmark(benchmarks, scan_by_bm)
    all_reviewed: set[str] = set()
    for papers in reviewed_by_bm.values():
        all_reviewed.update(papers)
    papers_reviewed = len(all_reviewed)

    coverage = {
        "last_updated": date.today().isoformat() if do_fetch else load_cached_coverage().get("last_updated"),
        "total_models": len({r["model"] for r in results}),
        "total_results": len(results),
        "total_papers_reviewed": papers_reviewed,
        "benchmarks": {},
    }

    for bm_key, bm_info in benchmarks.items():
        arxiv_id = bm_arxiv.get(bm_key)
        scan_entry = scan_by_bm.get(bm_key, {})
        # Prefer fresh scan counts; otherwise fall back to cached coverage.
        citing_count = scan_entry.get("citing_papers") or cached_cov.get(bm_key, {}).get("citing_papers")
        # Denominator for coverage. Use the pool size as written to
        # scan_results so it stays consistent with ``reviewed_by_bm``
        # (which includes the benchmark's own paper too). The raw S2
        # ``arxiv_citing_papers`` count doesn't reflect pool unions that
        # happen post-scan (robocasa after the gr1 merge) and excludes
        # self, which would otherwise make reviewed/citing > 100 %.
        pool_ids = scan_entry.get("all_citing_ids") or []
        if pool_ids:
            arxiv_citing_count = len(pool_ids)
        else:
            arxiv_citing_count = scan_entry.get("arxiv_citing_papers") or cached_cov.get(bm_key, {}).get(
                "arxiv_citing_papers"
            )

        n_results = result_counts.get(bm_key, 0)
        n_papers = len(reviewed_by_bm.get(bm_key, set()))
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
        print(f"  {bm_key}: {n_results} entries, {n_papers} papers reviewed, {status}")

    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2) + "\n")
    print(f"\nCoverage written to {COVERAGE_PATH}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        # ``arxiv_citing_papers`` now already includes the benchmark paper
        # itself (see the pool-size derivation above).
        total_reviewable = sum(
            b.get("arxiv_citing_papers") or b.get("citing_papers") or 0 for b in coverage["benchmarks"].values()
        )
        total_reviewed = coverage["total_papers_reviewed"]
        pct = f"{total_reviewed / total_reviewable * 100:.1f}%" if total_reviewable else "N/A"
        lines = [
            f"- {len(coverage['benchmarks'])} benchmarks, {coverage['total_models']} models, {coverage['total_results']} results",
            f"- {total_reviewed} / {total_reviewable} reviewable papers reviewed ({pct})",
        ]
        with open(output_path, "a") as f:
            f.write("coverage_summary<<GITHUB_OUTPUT_EOF\n")
            f.write("\n".join(lines) + "\n")
            f.write("GITHUB_OUTPUT_EOF\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
