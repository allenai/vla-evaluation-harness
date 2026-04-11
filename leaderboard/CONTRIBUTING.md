# Contributing to the VLA Leaderboard

> **Note on evaluation protocols:** Benchmark evaluation protocols are not fully standardized across the VLA community. Different papers may use the same benchmark name but differ in training regimes, task subsets, or evaluation conditions — making scores not always directly comparable. This leaderboard records all available results transparently and documents known protocol differences, but gaps remain. We actively welcome contributions: score corrections, missing results, protocol clarifications, and proposals for standardization.

## Data Structure

All data lives in `leaderboard/data/results.json` — the single source of truth.

### Benchmarks

| Benchmark | Metric | Unit | Range |
|-----------|--------|------|-------|
| LIBERO, LIBERO-Plus, LIBERO-Pro | success_rate | % | 0–100 |
| LIBERO-Mem | subgoal_completion_rate | % | 0–100 |
| CALVIN | avg_len | subtasks | 0–5 |
| SimplerEnv, RLBench, ManiSkill2, RoboCasa, RoboTwin 1.0, RoboTwin 2.0, VLABench, MIKASA-Robo, Kinetix, RoboCerebra, RoboChallenge | success_rate | % | 0–100 |
| RoboArena | elo_rating | Elo | 0–2000 |

Each benchmark declares its metric, range, and optionally `suites`/`tasks`. See the JSON for the full registry.

Every benchmark has a `detail_notes` field displayed as a banner on the leaderboard frontend. When changing a benchmark's scoring rules or comparability notes, update `detail_notes` in `results.json` to match.

### Result Fields

Each result is **self-contained** — model metadata is inlined:

```json
{
  "model": "openvla",  "display_name": "OpenVLA",  "params": "7B",
  "model_paper": "https://arxiv.org/abs/2406.09246",
  "benchmark": "libero",  "weight_type": "finetuned",
  "overall_score": 85.7,
  "suite_scores": { "libero_spatial": 84.0, "libero_object": 88.0 },
  "source_paper": "https://arxiv.org/abs/2406.09246",
  "source_table": "Table 1",
  "curated_by": "opus 4.6",  "date_added": "2026-03-02"
}
```

**Required**: `model`, `display_name`, `benchmark`, `weight_type`, `curated_by`, `date_added`

**Key fields**:

| Field | Meaning | Null when |
|-------|---------|-----------|
| `model_paper` | Paper that **introduces the model** (architecture, training) | No arxiv paper (proprietary models) |
| `source_paper` | Paper where this **specific score was reported** | Score from official leaderboard API |
| `overall_score` | Aggregate score (controls ranking) | Non-standard protocol (→ `null`), or only per-suite scores available |
| `params` | Parameter count (e.g. `"7B"`) | Unknown |
| `name_in_paper` | Exact model label from the source paper's results table (e.g. `"Ours (π₀)"`) — used by `reconcile.py` for identity matching. `display_name` is the curator-controlled leaderboard label. | Not yet verified, or name matches `display_name` |

- `model_paper` / `source_paper` must be **full URLs** (`https://arxiv.org/abs/...`), not bare IDs — bare IDs render as broken links.
- `weight_type`: `"shared"` (same checkpoint across benchmarks) or `"finetuned"` (trained on this benchmark).
- `curated_by`: AI-extracted → model name (`"opus 4.6"`); human-verified → GitHub handle (`"@user"`).
- `notes`: Free-text for caveats (non-standard eval, different task subset, etc.).
- `overall_score` must only be set when the entry uses the benchmark's **standard evaluation protocol**. Entries using non-standard task subsets, different task counts, or incompatible evaluation setups must set `overall_score` to `null` and store the original aggregate in `task_scores.reported_avg` — this prevents misleading rankings while preserving the data. See [Benchmark-Specific Caveats](#benchmark-specific-caveats) for each benchmark's standard protocol.
- `validate.py` enforces: every entry must have at least one score (`overall_score`, `suite_scores`, or `task_scores`). For non-standard entries (`overall_score: null`), task/suite key names are not validated against the declared list since they use different protocols.

## Score Provenance

When adding scores, correctly attribute **who ran the evaluation**:

| Scenario | `model_paper` | `source_paper` | `model` key |
|----------|--------------|----------------|-------------|
| Authors evaluate their own model | Model's paper | Same paper | Original key (e.g. `openvla`) |
| Paper B re-trains/fine-tunes Model A from scratch | Model A's paper | Paper B | Separate key (e.g. `openvla_memoryvla`) |
| Paper B downloads Model A's checkpoint and evaluates as-is | Model A's paper | Paper B | Original key; note eval setup differences in `notes` |
| Paper B cites Paper A's score without re-running | Model A's paper | Paper A (original) | Original key |

**Rules**:
- Third-party reproductions always get a **separate model key** with a descriptive suffix (e.g. `openvla_memoryvla` = "OpenVLA reproduced by MemoryVLA authors"). Add `notes` explaining it is a reproduction.
- Baseline copies (citing without re-running) are acceptable only when the original score is not already in the leaderboard.
- When in doubt, create a separate entry — two entries can be merged later, but conflated runs cannot be separated.
- **Non-standard evaluation protocols** (different task subsets, custom metrics, modified benchmarks) must NOT be filed under the standard benchmark. Either create a separate benchmark or omit the entry.

## How to Add Results

1. **Add entries** to the `results` array (sorted by `benchmark, model`). Keep `display_name` and `params` consistent across entries for the same model.

2. **Update `last_updated`**: Set `last_updated` in `results.json` to today's date (`YYYY-MM-DD`) when adding or modifying result data. This is displayed on the frontend and must reflect the latest data change.

3. **Validate**: `python leaderboard/scripts/validate.py`
   - Auto-fix sort order and formatting: `python leaderboard/scripts/validate.py --fix`

4. **Update coverage** (optional): `python leaderboard/scripts/update_coverage.py [--fetch]`
   - `papers_reviewed` lists all arxiv IDs reviewed per benchmark (with or without results).

5. **Test locally**: `cd leaderboard/site && python -m http.server`

## Official Leaderboard Policy

Benchmarks with `official_leaderboard` in their registry entry require **API-synced entries only** — `curated_by` must end with `-api`. Manual paper extractions are prohibited. `validate.py` enforces this.

## CI/CD

- **`leaderboard-validate.yml`**: Runs `validate.py` on every PR touching `results.json` or `citations.json`
- **`pages.yml`**: Deploys to GitHub Pages on push to main; regenerates `coverage.json` and `citations.json`
- **`update-data.yml`**: Syncs external leaderboard sources weekly (Monday 06:00 UTC) and opens a PR with updates. Can also be triggered manually via `workflow_dispatch`.

## Benchmark Protocols

Each benchmark has its own protocol file in `leaderboard/benchmarks/`:

```
leaderboard/benchmarks/
  _global.md          # risky patterns that apply to every benchmark
  libero.md           # LIBERO protocol, caveats, risky patterns
  calvin.md           # CALVIN protocol, caveats, risky patterns
  ...                 # one file per benchmark
```

These files are the **single source of truth** for benchmark protocols. They are:
- Read by `extract.py` to build LLM prompts for score extraction
- Read by `extract.py audit` to produce protocol health reports
- The reference for human curators when adding entries

See each file for: standard protocol definition, caveats, and risky patterns checklist.

## Schema

JSON Schema: `leaderboard/data/schema.json`. Key nullable types: `overall_score`, `source_paper`, `source_table`, `params`, `model_paper` — all `["string"|"number", "null"]`.
