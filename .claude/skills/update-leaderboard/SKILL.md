---
name: update-leaderboard
description: "Run the monthly leaderboard update cycle: sync external APIs, scan new citing papers across all benchmarks, extract scores, and curate entries into results.json. Use this skill whenever the user wants to do a bulk leaderboard update, monthly refresh, scan for new papers, or otherwise batch-update leaderboard/data/results.json — even casually like 'update the leaderboard' or 'monthly refresh'. Also triggers for single-paper additions like 'add this paper to the leaderboard'."
---

# Update Leaderboard

Monthly bulk update cycle for the VLA leaderboard. All data rules and benchmark-specific caveats live in `leaderboard/CONTRIBUTING.md` — read it before every update.

## Phase 1: External sync (automated)

Run the existing scripts to sync API-sourced data and refresh metadata.

```bash
# 1. Sync RoboArena + RoboChallenge APIs
python leaderboard/scripts/sync_external.py --apply

# 2. Update citation counts from Semantic Scholar
python leaderboard/scripts/update_citations.py --fetch

# 3. Update coverage stats
python leaderboard/scripts/update_coverage.py --fetch
```

Review the diffs. These are mechanical updates — verify the numbers look reasonable (no dramatic drops, no missing models).

## Phase 2: Scan new citing papers

For each benchmark, find papers that cite the benchmark paper but haven't been reviewed yet.

1. Read `leaderboard/data/coverage.json` to get each benchmark's `arxiv_id` and current `papers_reviewed` count.
2. Read `leaderboard/data/results.json` to get the full `papers_reviewed` lists from each benchmark entry.
3. For each benchmark, use the Semantic Scholar citation graph (`mcp semantic-scholar get_paper` with `citations.title` fields, or `mcp arxiv citation_graph`) to fetch citing papers.
4. Filter out papers already in `papers_reviewed`.
5. Present a summary table to the user:

```
Benchmark       | Total citing | Reviewed | New to review
LIBERO          |          693 |      551 |           142
CALVIN          |          494 |      391 |           103
...
```

The user decides which benchmarks / papers to prioritize.

## Phase 3: Extract scores (semi-automated)

For each new paper the user selects:

1. `get_abstract` first — check if the paper is even relevant (many citing papers don't report benchmark scores).
2. If relevant, `download_paper` and `read_paper`. Focus on tables and results sections.
3. **Before building any entry**, read the "Benchmark-Specific Caveats" section of `leaderboard/CONTRIBUTING.md` for that benchmark. Each benchmark defines a standard evaluation protocol — the protocol determines whether `overall_score` can be set or must be `null`, which `suite_scores`/`task_scores` keys are required, and what to record in `notes`. Getting this wrong produces misleading rankings.
4. For each candidate entry, verify:
   - Does the paper's eval protocol match the benchmark's **standard protocol**? If not → `overall_score: null`.
   - Are all required score breakdowns present (e.g., LIBERO needs `suite_scores`, SimplerEnv needs 3 dimensions)?
   - Is the score provenance correct? (see "Score Provenance" in CONTRIBUTING.md)
5. Add the paper's arxiv ID to the benchmark's `papers_reviewed` regardless of whether scores were extracted.

Present candidate entries to the user as formatted JSON before applying. Highlight judgment calls (protocol deviations, ambiguous weight_type, missing fields).

## Phase 4: Apply and validate

After user approval:

1. Add entries to `leaderboard/data/results.json` and update `last_updated`.
2. Validate:
   ```bash
   python leaderboard/scripts/validate.py --fix
   ```
3. Show the final diff to the user.

## Commit hygiene

Separate automated sync changes from manual curation into distinct commits:

1. **First commit**: Phase 1 external sync output only (API sync, citations, coverage). These are mechanical, reproducible changes.
2. **Second commit**: Phase 3 curated entries from paper extraction. These involve human judgment.

This separation makes review easier and allows reverting one without the other. Always create a PR branch — never commit directly to main.

## Single-paper mode

When the user provides a specific arxiv ID or URL, skip Phases 1–2 and go directly to Phase 3 for that paper.
