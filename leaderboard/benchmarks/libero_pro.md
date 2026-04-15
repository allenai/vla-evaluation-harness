---
benchmark: libero_pro
display_name: LIBERO-Pro
paper_url: https://arxiv.org/abs/2510.03827
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- goal_env
- goal_lang_aug
- goal_obj
- goal_ori
- goal_original
- goal_pos
- goal_position
- goal_sem
- goal_task
- goal_vision_aug
- language_perturbation
- libero_10
- libero_goal
- libero_object
- libero_spatial
- long_env
- long_obj
- long_ori
- long_pos
- long_position
- long_sem
- long_task
- object_env
- object_obj
- object_ori
- object_perturbation
- object_pos
- object_position
- object_sem
- object_task
- pos_10
- pos_goal
- pos_object
- pos_spatial
- spatial_env
- spatial_lang_aug
- spatial_obj
- spatial_ori
- spatial_original
- spatial_pos
- spatial_position
- spatial_sem
- spatial_task
- spatial_vision_aug
- swap_perturbation
- task_10
- task_goal
- task_object
- task_perturbation
- task_spatial
detail_notes: "LIBERO-Pro robustness benchmark (<a href='https://arxiv.org/abs/2510.03827'>2510.03827</a>). Standard protocol: 4 suites (goal, spatial, long, object) × 5 perturbations (original, object_swap, position, semantic, task) = 20 cells. Optional 6th perturbation: environment (object suite only). <code>overall_score</code> = mean of 20 core cells (excl env). Null if &lt;20 cells reported."
aggregation:
  container: suite_scores
  keys:
  - goal_ori
  - goal_obj
  - goal_pos
  - goal_sem
  - goal_task
  - spatial_ori
  - spatial_obj
  - spatial_pos
  - spatial_sem
  - spatial_task
  - long_ori
  - long_obj
  - long_pos
  - long_sem
  - long_task
  - object_ori
  - object_obj
  - object_pos
  - object_sem
  - object_task
---

**Standard**: LIBERO-Pro robustness benchmark ([2510.03827](https://arxiv.org/abs/2510.03827)) — 4 suites (`goal`, `spatial`, `long`, `object`) × 5 core perturbations (`ori`, `obj`, `pos`, `sem`, `task`) = 20 core cells; `overall_score` = arithmetic mean of the 20 core cells.

## Scoring
- `overall_score`: arithmetic mean over the 20 cells (`{suite}_{pert}` for suites in {goal, spatial, long, object} × perturbations in {ori, obj, pos, sem, task}); `null` if any of the 20 cells is missing.
- `suite_scores`: canonical keys use format `{suite}_{perturbation}` with short names: `goal_ori`, `spatial_obj`, `long_pos`, etc. The optional 6th perturbation `env` (only available for `object` suite) goes in `object_env` and is EXCLUDED from the mean.
- `task_scores`: not used.

## Checks
- Are all 20 core cells present ({goal, spatial, long, object} × {ori, obj, pos, sem, task})?
- Is `object_env` (when reported) kept out of the 20-cell mean?
- Are non-standard perturbation types (e.g. `lang_aug`, `vision_aug`) excluded from `libero_pro` entirely (they belong in a separate benchmark or must be omitted)?
- Is the per-task evaluation count (50 episodes per task per standard) recorded when it deviates?
