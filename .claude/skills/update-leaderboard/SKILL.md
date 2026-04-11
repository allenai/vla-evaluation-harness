---
name: update-leaderboard
description: "Curate VLA benchmark scores from published arxiv papers into leaderboard/data/results.json and keep the leaderboard's backing state fresh. Use this skill whenever the user wants to update the leaderboard in any way — monthly bulk refresh, scanning for new citing papers, extracting scores from a specific paper, or any targeted edit to results.json. Triggers on casual phrasing too: 'update the leaderboard', 'monthly refresh', 'add this paper to the leaderboard', 'pull scores from 2406.09246', or questions about how the VLA benchmark results registry is maintained."
---

# Update leaderboard

Curate scores from published VLA papers into `leaderboard/data/results.json`. The registry of benchmarks, the per-benchmark evaluation protocols, and the rules for `overall_score` / `suite_scores` / `task_scores` all live in `leaderboard/CONTRIBUTING.md`. Read it before every update — each benchmark has sharp edges that will corrupt rankings if mishandled (CALVIN's ABC→D vs ABCD→D split, LIBERO excluding `libero_90` from the 4-suite mean, RoboTwin v2 requiring `overall_score = null` with Protocol A vs B noted, SimplerEnv never averaging across VM/VA/WidowX, and so on).

The workflow for a full update has five steps. For adding a single paper, jump straight to steps 3–4; for fixing an existing entry, jump to step 4.

## 1. Sync external data sources

RoboArena and RoboChallenge scores come from API feeds, not from paper extraction. Citation counts and coverage stats come from Semantic Scholar. Refresh all three first so the rest of the session works off fresh numbers.

```bash
python leaderboard/scripts/sync_external.py --apply
python leaderboard/scripts/update_citations.py --fetch
python leaderboard/scripts/update_coverage.py --fetch
```

Skim the diffs. These updates are mechanical and should look plausible: no dramatic Elo drops, no models vanishing, no RoboChallenge rows collapsing to zero.

## 2. Find unreviewed citing papers

For each benchmark, ask which papers cite its paper but are not yet in its `papers_reviewed` list:

1. Read `leaderboard/data/coverage.json` and `leaderboard/data/results.json` to get each benchmark's arxiv id and current `papers_reviewed` list.
2. Query the Semantic Scholar citation graph via `mcp semantic-scholar get_paper` (fields: `citations.title,citations.externalIds`) or `mcp arxiv citation_graph`. Semantic Scholar is authoritative; arxiv's own citation graph lags.
3. Subtract the already-reviewed arxiv ids.
4. Show the user a per-benchmark summary (total citing / already reviewed / new to review) and let them pick where to spend time.

Citing papers are noisy: most cite a benchmark in passing without running it. Treat the result as a backlog to triage, not a work queue.

## 3. Extract scores from each paper

For each paper the user picks up:

1. Read the abstract first via `get_abstract`. Many citing papers never actually run the benchmark — stop early when that's the case instead of downloading a full paper you will discard.
2. For the ones that do, `download_paper` and open the results section. Look for a table that names the benchmark and has the paper's own model as a row.
3. Re-read the benchmark's subsection in `leaderboard/CONTRIBUTING.md` before building the entry. The protocol decides whether `overall_score` can be set or must be `null`, which `suite_scores` / `task_scores` keys are allowed, and what has to appear in `notes`. A thirty-second re-read here eliminates the most common class of misfiled entries — scores are easy to copy, but protocol interpretation is where things silently break.
4. Mirror the paper's scores into a candidate entry as formatted JSON and show it to the user before touching `results.json`. Call out every judgment call explicitly — non-standard protocol, ambiguous `weight_type`, third-party reproduction vs self-reported, missing required breakdowns, name collisions with existing model keys.
5. Whether the paper yielded an entry or not, its arxiv id belongs in that benchmark's `papers_reviewed` list. Coverage tracking is only useful when it reflects what was actually looked at.

## 4. Extract and reconcile against source papers

Verification uses a two-phase pipeline that separates LLM extraction from comparison:

**Phase 1 — Extract** (`extract.py`): reads the source paper and benchmark rules, asks the LLM to independently extract all models' scores. Output goes to `extraction.json`. Does NOT read stored values from `results.json`.

```bash
# One-time: fetch missing source papers
python leaderboard/scripts/extract.py --fetch-missing

# Extract from a single paper for one benchmark
python leaderboard/scripts/extract.py --paper <arxiv_id> --benchmark <key> --model sonnet

# Extract all (paper, benchmark) pairs referenced in results.json
python leaderboard/scripts/extract.py --all --model sonnet --workers 2 --resume
```

**Phase 2 — Reconcile** (`reconcile.py`): compares `extraction.json` against `results.json`. Pure Python, no LLM, runs in seconds. Re-run freely after editing `results.json` without re-extracting.

```bash
python leaderboard/scripts/reconcile.py
python leaderboard/scripts/reconcile.py --benchmark libero --format detail
```

Both scripts use the Claude Code CLI (`claude --print --json-schema`) for LLM calls, so they run on the curator's subscription with no API key needed.

Skip extraction for rows produced by `sync_external.py` — those are mechanical API copies with no source paper to extract from.

## 5. Apply and validate

Once the candidates are reconciled clean:

1. Add them to `leaderboard/data/results.json` and bump `last_updated`.
2. Run `python leaderboard/scripts/validate.py --fix`. The validator runs schema, range, sort, arithmetic consistency, forbidden `overall_score`, cross-entry identity checks.
3. Show the user the final diff before committing.

## Committing

Keep automated sync output and human-curated entries in **separate commits**. Sync output is mechanical and reproducible; curated entries involve judgment. Mixing them forces a reviewer to context-switch between "is this number right?" and "are these scripts doing the right thing?", and prevents reverting one without the other. Always push to a PR branch — never commit directly to main.
