# Contributing to the VLA Leaderboard

> **Note on evaluation protocols:** Benchmark evaluation protocols are not fully standardized across the VLA community. Different papers may use the same benchmark name but differ in training regimes, task subsets, or evaluation conditions — making scores not always directly comparable. This leaderboard records all available results transparently and documents known protocol differences, but gaps remain. We actively welcome contributions: score corrections, missing results, protocol clarifications, and proposals for standardization.

## Data Structure

All data lives in `leaderboard/data/results.json` — the single source of truth.

### Benchmarks

| Benchmark | Metric | Unit | Range |
|-----------|--------|------|-------|
| LIBERO, LIBERO-Plus, LIBERO-Pro, LIBERO-Mem | success_rate | % | 0–100 |
| CALVIN | avg_len | subtasks | 0–5 |
| SimplerEnv, RLBench, ManiSkill2, RoboCasa, RoboTwin 1.0, RoboTwin 2.0, VLABench, MIKASA, Kinetix, RoboCerebra | success_rate | % | 0–100 |

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

2. **Validate**: `python leaderboard/scripts/validate.py`
   - Auto-fix sort order and formatting: `python leaderboard/scripts/validate.py --fix`

3. **Update coverage** (optional): `python leaderboard/scripts/update_coverage.py [--fetch]`
   - `papers_reviewed` lists all arxiv IDs reviewed per benchmark (with or without results).

4. **Test locally**: `cd leaderboard/site && python -m http.server`

## Official Leaderboard Policy

Benchmarks with `official_leaderboard` in their registry entry require **API-synced entries only** — `curated_by` must end with `-api`. Manual paper extractions are prohibited. `validate.py` enforces this.

## CI/CD

- **`leaderboard-validate.yml`**: Runs `validate.py` on every PR touching `results.json` or `citations.json`
- **`pages.yml`**: Deploys to GitHub Pages on push to main; regenerates `coverage.json` and `citations.json`
- **`update-data.yml`**: Syncs external leaderboard sources weekly (Monday 06:00 UTC) and opens a PR with updates. Can also be triggered manually via `workflow_dispatch`.

## Benchmark-Specific Caveats

### SimplerEnv

- **Standard protocol**: 3 independent evaluation dimensions — **never average across them**. `overall_score` = always `null`; use `suite_scores` only.

| Dimension | Robot | Protocol | Benchmark key |
|-----------|-------|----------|---------------|
| Google Robot VM | Google Robot | Visual Matching | `suite_scores.google_robot_vm` |
| Google Robot VA | Google Robot | Variant Aggregation | `suite_scores.google_robot_va` |
| WidowX VM | WidowX (Bridge) | Visual Matching | `suite_scores.widowx_vm` |
- **Google Robot VM standardization**: `suite_scores.google_robot_vm` must always store the **3-task average** (Pick Coke Can, Move Near, Open/Close Drawer) for consistent ranking. Papers reporting 4 tasks (adding Place Apple in Drawer) should store the 4th task in `task_scores.place_apple_in_drawer_vm` and note the original 4-task average in `notes`. This ensures apples-to-apples comparison since 3-task is the dominant protocol (used by ~80% of papers).
- **Google Robot VA standardization**: `suite_scores.google_robot_va` follows the same rule — always store the **3-task average** (Pick Coke Can, Move Near, Open/Close Drawer). Papers reporting 4 tasks store the 4th in `task_scores.place_apple_in_drawer_va`. This ensures VM and VA scores are directly comparable.
- **task_scores protocol suffix**: All SimplerEnv `task_scores` keys **must** end with `_vm` or `_va` to indicate the evaluation protocol (e.g., `pick_coke_can_vm`, `move_near_va`). WidowX tasks always use `_vm`. `validate.py` enforces this. This prevents ambiguity since VM and VA evaluate the same tasks under different protocols with different scores.
- Don't confuse real-robot scores (e.g. OpenVLA's 12-task real eval) with SimplerEnv simulation.

### CALVIN

- **Standard protocol**: ABC→D split (train on A/B/C, eval on D), 1000 eval chains. ABCD→D inflates scores — do not add.
- Metric: avg completed subtasks in chain of 5 (0–5), not success rate.
- Note deviations from 1000 chains.

### LIBERO

- **Standard protocol**: 4-suite average (`spatial`, `object`, `goal`, `10`). Always include `suite_scores`. A 5th suite (`90`) exists but many papers skip it.
- `overall_score` = arithmetic mean of evaluated suites.
- LIBERO-Plus, LIBERO-Pro and LIBERO-Mem are **separate benchmarks**.

### LIBERO-Plus

- Robustness benchmark ([2510.13626](https://arxiv.org/abs/2510.13626)) with **7 perturbation dimensions**: Camera, Robot, Language, Light, Background, Noise, Layout.
- Models are trained on standard LIBERO and evaluated **zero-shot** under perturbations.
- `overall_score` = arithmetic mean of 7 perturbation dimensions. Always include `suite_scores`.
- `weight_type`: `"shared"` for zero-shot models (LIBERO-trained); `"finetuned"` for models trained on LIBERO-Plus data.
- Some papers (e.g. JEPA-VLA) use reduced training data (1/10 LIBERO) — record in `notes`.
- Partial evaluations (fewer than 7 dimensions) should NOT be filed under `libero_plus`.

### ManiSkill2

- **Standard protocol**: 5-task set (PickCube, StackCube, PickSingleYCB, PickSingleEGAD, PickClutterYCB). `overall_score` = `null` for other task subsets.
- Averaging varies (weighted vs arithmetic). Note method if known.

### RLBench

- **Standard protocol**: 18-task (PerAct) subset. `overall_score` = `null` for non-18-task evaluations. Record task count in `notes`.
- Multi-variation (e.g. 25 per task) vs single variation significantly affects scores.

### RoboCasa

- Papers may evaluate on different task subsets and episode counts. Record what was included.

### RoboTwin

- **v1 and v2 are separate benchmarks**. v1 = `robotwin_v1` ([2409.02920](https://arxiv.org/abs/2409.02920), ECCV 2024), v2 = `robotwin_v2` ([2506.18088](https://arxiv.org/abs/2506.18088), 2025).
- **v2 standard protocol**: `overall_score` = always `null`; use `suite_scores: {"easy": X, "hard": Y}`. Report both Easy (clean scenes) and Hard (5-axis domain randomization) when available.
- Task counts vary (v1: 4–17, v2: 3–50). Record task count in `notes`.
- Do not file CVPR 2025 Challenge results under standard v2 (different protocol).
- **Two v2 training protocols exist** — scores across them are **not comparable**:

  | | Protocol A (official) | Protocol B (Motus-style) |
  |---|---|---|
  | Source | [2506.18088](https://arxiv.org/abs/2506.18088) | [2512.13030](https://arxiv.org/abs/2512.13030) |
  | Training | Single-task, 50 clean demos/task | Multi-task, 50 clean + 500 DR demos/task |
  | Training data | 2,500 total | 27,500 total (11×) |
  | Hard/Rand meaning | OOD generalization (never seen DR) | In-distribution (trained on DR) |
  | Typical Easy/Hard gap | 3–10× (e.g. 55% / 5%) | Near-zero (e.g. 93% / 92%) |

  Always record which protocol in `notes` (prefix with `Protocol A` or `Protocol B`).

### MIKASA-Robo

- Some scores are third-party reproductions (e.g. MemoryVLA paper). Check `notes`.
- Ensure overall score reflects the paper's full aggregate, not a selective subset.

### RoboCerebra

- Includes both end-to-end VLAs and hierarchical systems (VLM planner + controller) — not directly comparable.
- Typical scores: 5–20%. Small absolute differences may be meaningful.

### Kinetix

- **Not the Kinetix simulator** — it's the 12-task eval protocol from the RTC paper ([2506.07339](https://arxiv.org/abs/2506.07339)). State-based, no vision/language.
- Scores depend on `(inference_delay, execution_horizon)` settings. Always record in `notes`.

### VLABench

- Official 6-track evaluation system ([OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)):
  - Track 1: `in_distribution` — task learning ability
  - Track 2: `cross_category` — object generalization
  - Track 3: `common_sense` — common sense understanding
  - Track 4: `semantic_instruction` — complex instruction understanding
  - Track 5: `cross_task` — skill transfer (kept open, not included in standard)
  - Track 6: `unseen_texture` — visual robustness (optional)
- **Two metrics**: IS (Intention Score, approached correct object) and PS (Progress Score, task completion). IS ≥ PS always.
- **Leaderboard standard**: `overall_score` = **Track 1-4 PS average**. Track 5-6 and IS values go in `suite_scores` as supplementary data.
- Original VLABench paper (2412.18194) uses a pre-track-system IS-based protocol (seen/unseen × base/commonsense). These entries have `overall_score: null`.
- Non-standard task subsets (e.g. cherry-picked tasks outside the official tracks) must NOT be filed under `vlabench`.
- Different papers evaluating the same model produce different scores due to fine-tuning setup and eval seeds. Use separate `model` keys per source paper (e.g. `pi0_acot_vlabench`, `pi0_xvla_vlabench`).

## Schema

JSON Schema: `leaderboard/data/schema.json`. Key nullable types: `overall_score`, `source_paper`, `source_table`, `params`, `model_paper` — all `["string"|"number", "null"]`.
