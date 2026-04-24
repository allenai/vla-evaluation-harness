---
benchmark: vlabench
display_name: VLABench
paper_url: https://arxiv.org/abs/2412.18194
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- avg_IS
- avg_PS
- common_sense
- commonsense_IS
- commonsense_PS
- composite_task_avg
- cross_category
- cross_category_IS
- cross_category_PS
- in_dist_IS
- in_dist_PS
- in_distribution
- intention_score
- lang_generalization_avg
- level_2b_lang_generalization
- level_2c_unseen_task
- level_2d_composite
- progress_score
- seen_base
- seen_commonsense
- semantic_instruction
- semantic_instruction_IS
- semantic_instruction_PS
- unseen_base
- unseen_commonsense
- unseen_task_avg
- unseen_texture_IS
- unseen_texture_PS
tasks:
- add_condiment
- get_coffee
- select_apple
- select_chemistry_tube
- set_study_table
- texas_holdem
detail_notes: "Standard: Track 1-4 Progress Score (PS) avg. Official 6-track system from OpenMOSS/VLABench: (1) in_distribution, (2) cross_category, (3) common_sense, (4) semantic_instruction, (5) cross_task [open], (6) unseen_texture. Metrics: IS (Intention Score) = approached correct object; PS (Progress Score) = task completion. overall_score = Track 1-4 PS avg. Entries from the original VLABench paper (2412.18194) use a pre-track IS-based protocol and have overall_score=null."
aggregation:
  container: suite_scores
  keys:
  - in_dist_PS
  - cross_category_PS
  - commonsense_PS
  - semantic_instruction_PS
---

**Standard**: VLABench 6-track evaluation system ([OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)) with IS (Intention Score) and PS (Progress Score) metrics; `overall_score` = arithmetic mean of Tracks 1–4 PS (`in_dist_PS`, `cross_category_PS`, `commonsense_PS`, `semantic_instruction_PS`).

## Scoring
- `overall_score`: mean of the four Track 1–4 PS values; `null` if any is missing OR if the entry uses the legacy pre-track-system protocol.
- `suite_scores`: canonical keys for Tracks 1–4 PS & IS — `in_dist_PS`, `in_dist_IS`, `cross_category_PS`, `cross_category_IS`, `commonsense_PS`, `commonsense_IS`, `semantic_instruction_PS`, `semantic_instruction_IS`. Optional Track 5 (`cross_task`, skill transfer) and Track 6 (`unseen_texture_PS` / `unseen_texture_IS`) are supplementary. Pre-track-system entries (2412.18194) use legacy keys (`seen_base`, `unseen_commonsense`, etc.) with `overall_score: null`.
- `task_scores`: not used — metrics are reported at the track level only.

## Checks
- Is `overall_score` the mean of Tracks 1–4 PS (NOT IS)? IS-based aggregates must be nulled.
- Are canonical suite keys used (`in_dist_PS`, `cross_category_PS`, `commonsense_PS`, `semantic_instruction_PS`)?
- Is the original VLABench paper (2412.18194) entry stored with legacy keys and `overall_score: null` since it predates the 6-track system?
- Is the task set one of the official tracks? Cherry-picked tasks outside the official tracks must NOT be filed under `vlabench`.

## Methodology axes (record in `notes`, do not null)
- Different papers evaluating the same model produce different scores due to fine-tuning setup and eval seeds. Use separate `model` keys per source paper (e.g. `pi0_acot_vlabench`, `pi0_xvla_vlabench`). This is the benchmark's convention for handling third-party measurements explicitly.
