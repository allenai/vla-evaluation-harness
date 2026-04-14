---
benchmark: libero_pro
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
