---
benchmark: libero
display_name: LIBERO
paper_url: https://arxiv.org/abs/2306.03310
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- libero_spatial
- libero_object
- libero_goal
- libero_10
- libero_90
expand_suites: true
avg_position: 4
avg_label: Avg (w/o 90)
detail_notes: "Standard protocol: 4-suite average (spatial, object, goal, 10). <code>overall_score</code> = mean of 4 standard suites only — <code>libero_90</code> is excluded from the mean even when reported."
aggregation:
  container: suite_scores
  keys:
  - libero_spatial
  - libero_object
  - libero_goal
  - libero_10
---

**Standard**: LIBERO 4-suite average (`spatial`, `object`, `goal`, `10`) trained on the standard 50-demo budget; `overall_score` = arithmetic mean of the four suites.

## Scoring
- `overall_score`: arithmetic mean of `suite_scores.libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; `null` if any of the four is missing.
- `suite_scores`: canonical keys `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. A reported 5th suite goes in `libero_90` and is EXCLUDED from `overall_score`.
- `task_scores`: not used at this level — per-task numbers belong on individual LIBERO sub-benchmarks.

## Checks
- Are all four standard suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) present? If not → `overall_score = null`.
- Is `libero_90` (when reported) stored only in `suite_scores.libero_90` and kept out of the 4-suite mean?
- Is this standard LIBERO and not LIBERO-Plus, LIBERO-Pro, or LIBERO-Mem (those are separate benchmarks)?

## Methodology axes (record in `notes`, do not null)
- Training data budget: standard is 50 demos/task. Reduced-data setups (e.g. 10 demos/task, 1/10 LIBERO, JEPA-VLA's reduced setting) are valid entries with the budget noted — not grounds for nulling.
- Evaluation rollouts per task: note any deviation from the common 50-rollout convention.
