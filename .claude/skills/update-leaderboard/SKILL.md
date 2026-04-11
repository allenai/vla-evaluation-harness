---
name: update-leaderboard
description: "Curate VLA benchmark scores from published arxiv papers into leaderboard/data/leaderboard.json. Use this skill for monthly bulk refresh, scanning for new citing papers, extracting scores, or any leaderboard update. Triggers on: 'update the leaderboard', 'monthly refresh', 'add this paper to the leaderboard', 'pull scores from 2406.09246'."
---

# Update leaderboard

Pipeline: `scan → run → refine → validate → sync_external`

Benchmark protocols live in `leaderboard/benchmarks/*.md`. Read them before manual edits.

## 1. Scan for new papers

```bash
uv run leaderboard/scripts/extract.py scan
```

Queries Semantic Scholar citation graph for all 17 benchmarks, discovers papers not yet extracted. Updates `coverage.json`.

## 2. Extract scores from papers

```bash
# All new papers from scan
uv run leaderboard/scripts/extract.py run --from-scan --workers 2

# Single paper
uv run leaderboard/scripts/extract.py run 2505.05800
```

One LLM call per paper (sonnet). Extracts ALL benchmark results found in the paper. Results stored in `leaderboard/extractions/{arxiv_id}.json` (git tracked, incremental). Papers with no benchmark results are stored as `"benchmarks": []`.

Auto-fetches paper HTML if not cached. `--resume` skips papers already extracted.

## 3. Refine into leaderboard.json

```bash
uv run leaderboard/scripts/refine.py --model opus
```

LLM (opus) processes each benchmark's raw entries:
- Deduplicates same model under different names
- Applies protocol rules (non-standard → `overall_score = null`, reject conditions)
- Computes `overall_score` from component scores
- Produces clean model keys and display names

## 4. Validate

```bash
uv run python leaderboard/scripts/validate.py
```

Static checks: schema, score ranges, arithmetic consistency, forbidden `overall_score`, cross-entry identity, required notes.

## 5. Sync API sources

```bash
python leaderboard/scripts/sync_external.py --apply
```

Adds RoboArena / RoboChallenge entries from API feeds.

## 6. Commit

Keep sync output and extraction-based entries in **separate commits**. Always push to a PR branch.

## Key files

| File | Purpose |
|------|---------|
| `leaderboard/scripts/extract.py` | scan + run (PEP 723, typer) |
| `leaderboard/scripts/refine.py` | extractions → leaderboard.json (PEP 723, typer) |
| `leaderboard/scripts/validate.py` | static validation |
| `leaderboard/scripts/sync_external.py` | RoboArena/RoboChallenge API sync |
| `leaderboard/scripts/update_citations.py` | Semantic Scholar citation counts |
| `leaderboard/benchmarks/*.md` | benchmark protocol definitions |
| `leaderboard/data/benchmarks.json` | benchmark registry (metrics, suites) |
| `leaderboard/data/leaderboard.json` | curated leaderboard entries |
| `leaderboard/extractions/` | per-paper extraction results (git tracked) |
