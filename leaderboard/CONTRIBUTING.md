# Contributing to the VLA Leaderboard

> **Note on evaluation protocols:** Benchmark evaluation protocols are not fully standardized across the VLA community. Different papers may use the same benchmark name but differ in training regimes, task subsets, or evaluation conditions — making scores not always directly comparable. This leaderboard records all available results transparently and documents known protocol differences, but gaps remain. We actively welcome contributions: score corrections, missing results, protocol clarifications, and proposals for standardization.

## Local Setup

`leaderboard/data/{leaderboard,extractions,scan_results}.json` are stored in **Git LFS** (see `leaderboard/data/.gitattributes`). Without LFS smudging, scripts will read pointer files and fail. Once per machine:

```
git lfs install                 # install the LFS hooks
git lfs pull                    # smudge any pointer files in the current checkout
```

CI workflows (`pages.yml`, `update-data.yml`, `leaderboard-validate.yml`) already pass `lfs: true` to `actions/checkout`.

## Data Structure

Data is split into focused files under `leaderboard/data/`:

| File | Contents |
|------|----------|
| `leaderboard.json` | Curated entries (`last_updated` + `results[]`) |
| `benchmarks.json` | Benchmark registry (build artifact — see below) |
| `citations.json` | Per-paper citation counts from Semantic Scholar |
| `coverage.json` | Per-benchmark coverage stats |
| `extractions.json` | Packed per-paper extractions (optional, for reproducibility) |

### Schemas (single source of truth)

Every field in the data files is defined in a JSON Schema with inline descriptions. Consult the schema before writing code, prompts, or docs — do not re-describe field semantics elsewhere.

| Schema | Covers |
|--------|--------|
| `leaderboard.schema.json` | `leaderboard.json` — final curated entries |
| `benchmarks.schema.json` | `benchmarks.json` — registry shape |
| `extraction.schema.json` | One paper's extract.py output. `extractions.json` is an array of these. |
| `candidates.schema.json` | `.cache/refine_candidates.json` — refine-stage input |

Per-benchmark protocol (Standard / Scoring / Checks / Methodology) lives in `leaderboard/benchmarks/{key}.md`. Frontmatter compiles into `benchmarks.json`; the markdown body is the LLM-facing protocol prose consumed by `extract.py` and `refine.py`.

**`benchmarks.json` is a build artifact — never edit it directly.** After editing any frontmatter, rebuild:

```
python leaderboard/scripts/build_benchmarks_json.py
```

CI runs `build_benchmarks_json.py --check` on every PR — if the committed `benchmarks.json` diverges from the md sources, the PR fails. Per-benchmark coverage (reviewed-paper counts) is derived from extraction records by `scan.py`, not stored in `benchmarks.json`.

## How to Add Results

1. **Add entries** to the `results` array (sorted by `benchmark, model`). Field shape and provenance rules are in `leaderboard.schema.json`; attribution cases (first-party vs third-party) are in `candidates.schema.json`'s `row_type` field description.

2. **Update `last_updated`**: Set `last_updated` in `leaderboard.json` to today's date (`YYYY-MM-DD`) when adding or modifying result data.

3. **Validate**: `python leaderboard/scripts/validate.py`
   - Auto-fix sort order and formatting: `python leaderboard/scripts/validate.py --fix`

4. **Update coverage** (optional): `python leaderboard/scripts/scan.py [--check]`  (default refreshes pools via S2; `--check` re-derives coverage only)

5. **Test locally**: `cd leaderboard/site && python -m http.server`

## Automated Extraction Pipeline

Paper-sourced entries are produced by:

```
scan.py                      # S2 /citations → data/scan_results.json + data/coverage.json
extract.py run --from-scan   # per-paper LLM extraction → .cache/extractions/
refine.py main               # protocol gate + per-benchmark LLM refinement → leaderboard.json
```

Field semantics live in the schema files above. `extract.py` and `refine.py` both load their respective schemas at runtime — the prompts reference the schema, not duplicate its field rules.

## Official Leaderboard Policy

Benchmarks with `official_leaderboard` in their registry entry require **API-synced entries only** — `curated_by` must end with `-api`. Manual paper extractions are prohibited. `validate.py` enforces this.

## CI/CD

- **`leaderboard-validate.yml`**: Runs `validate.py` on every PR touching `leaderboard.json` or `citations.json`
- **`pages.yml`**: Deploys to GitHub Pages on push to main; regenerates `coverage.json` and `citations.json`
- **`update-data.yml`**: Syncs external leaderboard sources weekly (Monday 06:00 UTC) and opens a PR with updates. Can also be triggered manually via `workflow_dispatch`.

## Benchmark Protocols

Per-benchmark Standard, Scoring, Checks, and Methodology axes live in `leaderboard/benchmarks/{key}.md`. That file is the single source — this document does not mirror it.
