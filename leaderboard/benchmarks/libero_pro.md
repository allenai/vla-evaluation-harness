---
benchmark: libero_pro
---

## Protocol

- Robustness benchmark ([2510.03827](https://arxiv.org/abs/2510.03827)) evaluating generalization under perturbations across LIBERO suites.
- **Standard protocol**: 4 suites × 5 core perturbations = **20 cells**.
  - Suites: `goal`, `spatial`, `long`, `object`
  - Core perturbations: `original` (ori), `object_swap` (obj), `position` (pos), `semantic` (sem), `task`
  - Optional 6th perturbation: `environment` (env) — only available for `object` suite
- **suite_scores key format**: `{suite}_{perturbation}` (e.g., `goal_ori`, `spatial_obj`, `long_pos`). Use canonical short names: `ori`, `obj`, `pos`, `sem`, `task`, `env`.
- `overall_score` = arithmetic mean of the **20 core cells only** (excluding optional `env`). Set `overall_score = null` if any of the 20 core cells are absent.
- Non-standard perturbation types (e.g., `lang_aug`, `vision_aug`) should NOT be filed under `libero_pro`. Use a separate benchmark or omit.
- 50 evaluation episodes per task, consistent with standard LIBERO.

## Risky Patterns

- Are all 20 core cells present ({goal, spatial, long, object} × {ori, obj, pos, sem, task})? If any missing → `overall_score` must be `null`.
- Are non-standard perturbation types (`lang_aug`, `vision_aug`, `position`) excluded from the 20-cell mean?
