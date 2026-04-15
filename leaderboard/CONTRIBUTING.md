# Contributing to the VLA Leaderboard

> **Note on evaluation protocols:** Benchmark evaluation protocols are not fully standardized across the VLA community. Different papers may use the same benchmark name but differ in training regimes, task subsets, or evaluation conditions — making scores not always directly comparable. This leaderboard records all available results transparently and documents known protocol differences, but gaps remain. We actively welcome contributions: score corrections, missing results, protocol clarifications, and proposals for standardization.

## Data Structure

Data is split into focused files under `leaderboard/data/`:

| File | Contents |
|------|----------|
| `leaderboard.json` | Curated entries (`last_updated` + `results[]`) |
| `benchmarks.json` | Benchmark registry (metrics, suites, tasks, display config) |
| `leaderboard.schema.json` | JSON Schema for leaderboard.json |
| `benchmarks.schema.json` | JSON Schema for benchmarks.json |
| `citations.json` | Per-paper citation counts from Semantic Scholar |
| `coverage.json` | Per-benchmark coverage stats |

Per-benchmark protocol notes (standard split, metric formula, common
deviations) live alongside the registry at `leaderboard/benchmarks/{key}.md`.

### Benchmarks

17 benchmarks: LIBERO, LIBERO-Plus, LIBERO-Pro, LIBERO-Mem, CALVIN, SimplerEnv, RLBench, ManiSkill2, RoboCasa, RoboTwin v1, RoboTwin v2, VLABench, MIKASA-Robo, Kinetix, RoboCerebra, RoboArena, RoboChallenge.

Each benchmark declares its metric, range, and optionally `suites`/`tasks` in `benchmarks.json`. See `leaderboard/benchmarks/{key}.md` for the protocol and risky-pattern notes for each benchmark.

**`benchmarks.json` is a build artifact — never edit it directly.** Each benchmark's configuration (display_name, metric, suites, tasks, aggregation rule, detail_notes, etc.) lives in the YAML frontmatter of `leaderboard/benchmarks/{key}.md`. The markdown body of the same file holds the LLM-consumed protocol prose (Standard / Scoring / Checks / Methodology).

After editing any frontmatter, rebuild `benchmarks.json`:

```
python leaderboard/scripts/build_benchmarks_json.py
```

CI runs `build_benchmarks_json.py --check` on every PR — if the committed `benchmarks.json` diverges from the md sources, the PR fails. The only field in `benchmarks.json` that is NOT sourced from md is `papers_reviewed`, which is owned by `update_coverage.py` and preserved across builds.

### Result Fields

Each result is **self-contained** — model metadata is inlined:

```json
{
  "model": "openvla",  "display_name": "OpenVLA",  "params": "7B",
  "model_paper": "https://arxiv.org/abs/2406.09246",
  "benchmark": "libero",  "weight_type": "finetuned",
  "overall_score": 85.7,
  "suite_scores": { "libero_spatial": 84.0, "libero_object": 88.0 },
  "reported_paper": "https://arxiv.org/abs/2406.09246",
  "reported_table": "Table 1",
  "curated_by": "opus 4.6",  "date_added": "2026-03-02"
}
```

**Required**: `model`, `display_name`, `benchmark`, `weight_type`, `curated_by`, `date_added`

**Key fields**:

| Field | Meaning | Null when |
|-------|---------|-----------|
| `model_paper` | Paper that **introduces the model** (architecture, training) | No arxiv paper (proprietary models) |
| `reported_paper` | Paper where this **specific score was reported** | Score from official leaderboard API |
| `overall_score` | Aggregate score (controls ranking) | Non-standard protocol (→ `null`), or only per-suite scores available |
| `params` | Parameter count (e.g. `"7B"`) | Unknown |

- `model_paper` / `reported_paper` must be **full URLs** (`https://arxiv.org/abs/...`), not bare IDs — bare IDs render as broken links.
- `weight_type`: `"shared"` (same checkpoint across benchmarks) or `"finetuned"` (trained on this benchmark).
- `curated_by`: AI-extracted → model name (`"opus 4.6"`); human-verified → GitHub handle (`"@user"`).
- `notes`: Free-text for caveats (non-standard eval, different task subset, etc.).
- `overall_score` must only be set when the entry uses the benchmark's **standard evaluation protocol**. Entries using non-standard task subsets, different task counts, or incompatible evaluation setups must set `overall_score` to `null` and store the original aggregate in `task_scores.reported_avg` — this prevents misleading rankings while preserving the data. See [Benchmark-Specific Caveats](#benchmark-specific-caveats) for each benchmark's standard protocol.
- `validate.py` enforces: every entry must have at least one score (`overall_score`, `suite_scores`, or `task_scores`). For non-standard entries (`overall_score: null`), task/suite key names are not validated against the declared list since they use different protocols.

## Score Provenance

`model` keys use BibTeX citation key form. First-party entries use the method's own key (e.g. `kim24openvla`). Third-party measurements combine the method key with the measuring paper's key (e.g. `kim24openvla__black24xvla`) so every reproduction stays as its own row.

| Scenario | `model_paper` | `reported_paper` | `model` key | `display_name` |
|----------|--------------|----------------|-------------|----------------|
| Authors evaluate their own model | Model's paper | Same paper | `{method_bib}` (e.g. `kim24openvla`) | `OpenVLA` |
| Paper B re-trains/fine-tunes Model A from scratch | Model A's paper | Paper B | `{method_bib}__{B_bib}` (e.g. `kim24openvla__black24xvla`) | `OpenVLA (from X-VLA)` |
| Paper B downloads Model A's checkpoint and evaluates as-is | Model A's paper | Paper B | `{method_bib}__{B_bib}` (e.g. `kim24openvla__black24xvla`) | `OpenVLA (from X-VLA)` |
| Paper B cites Paper A's score without re-running | Model A's paper | Paper A (original) | `{method_bib}` (collapses to first-party) | `OpenVLA` |

**Rules**:
- Distinct `reported_paper`s always produce **distinct rows**. Never collapse a third-party measurement into a first-party canonical row.
- Citation-only rows (Paper B quoting Paper A without re-running) intentionally collapse to first-party because the measurement itself is the original — `reported_paper` should point at the original paper, not the citing one.
- The leaderboard frontend hides third-party rows by default behind an "Official results only" toggle, so the leaderboard stays clean while preserving variance for readers who want it.
- **Non-standard evaluation protocols** (different task subsets, custom metrics, modified benchmarks) must NOT be filed under the standard benchmark. Either create a separate benchmark or omit the entry.

## How to Add Results

1. **Add entries** to the `results` array (sorted by `benchmark, model`). Keep `display_name` and `params` consistent across entries for the same model.

2. **Update `last_updated`**: Set `last_updated` in `leaderboard.json` to today's date (`YYYY-MM-DD`) when adding or modifying result data. This is displayed on the frontend and must reflect the latest data change.

3. **Validate**: `python leaderboard/scripts/validate.py`
   - Auto-fix sort order and formatting: `python leaderboard/scripts/validate.py --fix`

4. **Update coverage** (optional): `python leaderboard/scripts/update_coverage.py [--fetch]`
   - `papers_reviewed` lists all arxiv IDs reviewed per benchmark (with or without results).

5. **Test locally**: `cd leaderboard/site && python -m http.server`

## Official Leaderboard Policy

Benchmarks with `official_leaderboard` in their registry entry require **API-synced entries only** — `curated_by` must end with `-api`. Manual paper extractions are prohibited. `validate.py` enforces this.

## CI/CD

- **`leaderboard-validate.yml`**: Runs `validate.py` on every PR touching `leaderboard.json` or `citations.json`
- **`pages.yml`**: Deploys to GitHub Pages on push to main; regenerates `coverage.json` and `citations.json`
- **`update-data.yml`**: Syncs external leaderboard sources weekly (Monday 06:00 UTC) and opens a PR with updates. Can also be triggered manually via `workflow_dispatch`.

## Benchmark-Specific Caveats

### SimplerEnv

- **Standard protocol**: 3 independent evaluation dimensions — **never average across them**. `overall_score` = always `null`; use `suite_scores` only. Store the paper's reported aggregate (if any) in `task_scores.reported_avg` per the global rule (see Result Fields).

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
- Record deviations from 1000 chains in `notes`.

### LIBERO

- **Standard protocol**: 4-suite average (`spatial`, `object`, `goal`, `10`). Always include `suite_scores`. A 5th suite (`90`) exists but many papers skip it.
- `overall_score` = arithmetic mean of the **4 standard suites only** (`spatial`, `object`, `goal`, `10`). Do NOT include `90` in the overall mean even when reported — store it in `suite_scores.libero_90` only. Entries reporting only a subset of the 4 standard suites must set `overall_score = null`.
- LIBERO-Plus, LIBERO-Pro and LIBERO-Mem are **separate benchmarks**.

### LIBERO-Plus

- Robustness benchmark ([2510.13626](https://arxiv.org/abs/2510.13626)) with **7 perturbation dimensions**: Camera, Robot, Language, Light, Background, Noise, Layout.
- Models are trained on standard LIBERO and evaluated **zero-shot** under perturbations.
- `overall_score` = arithmetic mean of **all 7** perturbation dimensions. Always include `suite_scores`. Entries with fewer than 7 dimensions must set `overall_score = null`.
- `weight_type`: `"shared"` for zero-shot models (LIBERO-trained); `"finetuned"` for models trained on LIBERO-Plus data.
- Some papers (e.g. JEPA-VLA) use reduced training data (1/10 LIBERO) — record in `notes`.

### LIBERO-Pro

- Robustness benchmark ([2510.03827](https://arxiv.org/abs/2510.03827)) evaluating generalization under perturbations across LIBERO suites.
- **Standard protocol**: 4 suites × 5 core perturbations = **20 cells**.
  - Suites: `goal`, `spatial`, `long`, `object`
  - Core perturbations: `original` (ori), `object_swap` (obj), `position` (pos), `semantic` (sem), `task`
  - Optional 6th perturbation: `environment` (env) — only available for `object` suite
- **suite_scores key format**: `{suite}_{perturbation}` (e.g., `goal_ori`, `spatial_obj`, `long_pos`). Use canonical short names: `ori`, `obj`, `pos`, `sem`, `task`, `env`.
- `overall_score` = arithmetic mean of the **20 core cells only** (excluding optional `env`). Set `overall_score = null` if any of the 20 core cells are absent.
- Non-standard perturbation types (e.g., `lang_aug`, `vision_aug`) should NOT be filed under `libero_pro`. Use a separate benchmark or omit.
- 50 evaluation episodes per task, consistent with standard LIBERO.

### LIBERO-Mem

- Memory benchmark ([2511.11478](https://arxiv.org/abs/2511.11478)) with **10 tasks** (T1–T10) across 4 types: OM (T1–T2), OS (T3–T5), OR (T6–T8), OO (T9–T10).
- **Metric**: subgoal completion rate (%), NOT task success rate. 20 rollouts per task.
- `overall_score` = unweighted arithmetic mean of T1–T10. Always include `task_scores`.
- Models using oracle subgoal information must note this in `notes`.

### ManiSkill2

- **Standard protocol**: 5-task set (PickCube, StackCube, PickSingleYCB, PickSingleEGAD, PickClutterYCB). `overall_score` = `null` for other task subsets.
- Always record the averaging method (weighted vs arithmetic) in `notes`. If unknown, note `'averaging method unknown'`.

### RLBench

- **Standard protocol**: 18-task PerAct subset ([2209.05451](https://arxiv.org/abs/2209.05451)), 249 total language-goal variations across the 18 tasks, 25 evaluation episodes per task (450 total), 100 training demos per task.
- `overall_score` = mean success rate across 18 tasks. Set `overall_score = null` for non-18-task evaluations. Always record task count in `notes`.
- **Variation count matters**: Multi-variation (e.g. 25 per task) vs single variation significantly affects scores. Record variation count in `notes` when known.
- Entries using single-task learning (training a separate policy per task) are not comparable to multi-task entries. Note the training regime.

### RoboCasa

- **Standard protocol**: 24 atomic tasks from the RoboCasa benchmark ([2406.02523](https://arxiv.org/abs/2406.02523)). `overall_score` = mean success rate across evaluated tasks.
- **Training data varies widely** (50–300 demos/task across papers). Always record `demos_per_task` and `task_count` in `notes`. Scores from different training budgets are not directly comparable.
- Entries evaluating non-standard task counts (e.g., 8 tasks, composite tasks) should note the deviation. Prefer `overall_score = null` for significantly non-standard subsets.
- Record episode count when known.

### RoboTwin

- **v1 and v2 are separate benchmarks**. v1 = `robotwin_v1` ([2409.02920](https://arxiv.org/abs/2409.02920), ECCV 2024), v2 = `robotwin_v2` ([2506.18088](https://arxiv.org/abs/2506.18088), 2025).
- **v2 standard protocol**: `overall_score` = always `null`; use `suite_scores: {"easy": X, "hard": Y}`. Report both Easy (clean scenes) and Hard (5-axis domain randomization) when available.
- **v1**: No standard task set — entries evaluate 4–17 tasks. Set `overall_score = null` unless the entry matches the original paper's exact task set. Always record task count in `notes`. Entries with different task counts are not comparable.
- **v2**: Task counts vary (3–50). Record task count in `notes`.
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

- **Standard protocol**: 5-task VLA evaluation ([2502.10550](https://arxiv.org/abs/2502.10550)): ShellGameTouch, InterceptMedium, RememberColor3, RememberColor5, RememberColor9. Endorsed as the standard by MemoryVLA (ICLR 2026). 100 evaluation episodes per task.
- `overall_score` = arithmetic mean of 5 task success rates. Always include `task_scores`.
- Entries using non-standard task sets (e.g., ELMUR 4-task: RC3/5/9 + TakeItBack) must set `overall_score = null`. Store the paper's reported aggregate in `suite_scores.reported_avg`.
- Some scores are third-party reproductions (e.g. MemoryVLA paper). Check `notes`.

### RoboCerebra

- Embodied reasoning benchmark ([2506.06677](https://arxiv.org/abs/2506.06677)) with **6 evaluation dimensions**: ideal, memory_execution, memory_exploration, mix, observation_mismatching, random_disturbance.
- `overall_score` = arithmetic mean of all 6 dimensions. Set `overall_score = null` if fewer than 6 dimensions are reported. Always include `suite_scores` when available.
- **Architecture types**: Entries include end-to-end VLAs, hierarchical systems (VLM planner + controller), and oracle (GT-Plan) upper bounds. These are not directly comparable. Note the architecture type in `notes`.
- Oracle entries (GT-Plan + VLA) represent non-deployable upper bounds. They should be clearly marked.
- Typical scores: 5–20%. Small absolute differences may be meaningful. All current entries are from the original paper (600 rollouts, same protocol).

### Kinetix

- **Not the Kinetix simulator** — it's the 12-task eval protocol from the RTC paper ([2506.07339](https://arxiv.org/abs/2506.07339)). State-based, no vision/language.
- Scores depend on `(inference_delay d, execution_horizon e)` settings. Always record both in `notes`.
- Entries at different `d` values are **not directly comparable** (e.g., d=0 scores ~11pp higher than d=4 for the same method). Prefer grouping by `d` when comparing.

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
- **Canonical suite_scores keys**: Use the track-based naming: `in_dist_IS`, `in_dist_PS`, `cross_category_IS`, `cross_category_PS`, `commonsense_IS`, `commonsense_PS`, `semantic_instruction_IS`, `semantic_instruction_PS`, `unseen_texture_IS`, `unseen_texture_PS`. Pre-track-system entries (2412.18194) use legacy keys (`seen_base`, `unseen_commonsense`, etc.) with `overall_score: null`.
- Original VLABench paper (2412.18194) uses a pre-track-system IS-based protocol (seen/unseen × base/commonsense). These entries have `overall_score: null`.
- Non-standard task subsets (e.g. cherry-picked tasks outside the official tracks) must NOT be filed under `vlabench`.
- Different papers evaluating the same model produce different scores due to fine-tuning setup and eval seeds. Use separate `model` keys per source paper (e.g. `pi0_acot_vlabench`, `pi0_xvla_vlabench`).

### RoboArena

- Elo-based pairwise comparison benchmark. Scores are Elo ratings (not success rates). Higher is better.
- All entries are API-synced (`curated_by` ends with `-api`). Manual entries are not accepted.
- Entries with fewer than 15 pairwise evaluations have high variance (std > 120 Elo) and should be interpreted cautiously.

### RoboChallenge

- Multi-task challenge benchmark with two scores: `overall_score` (success rate, binary task completion) and `suite_scores.progress_score` (partial credit for sub-goal progress).
- All entries are API-synced. Manual entries are not accepted.

## Schema

JSON Schema: `leaderboard/data/schema.json`. Key nullable types: `overall_score`, `reported_paper`, `reported_table`, `params`, `model_paper` — all `["string"|"number", "null"]`.
