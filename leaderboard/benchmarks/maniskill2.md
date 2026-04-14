---
benchmark: maniskill2
---

**Standard**: ManiSkill2 5-task set (PickCube, StackCube, PickSingleYCB, PickSingleEGAD, PickClutterYCB); `overall_score` = mean success rate across the 5 tasks.

## Scoring
- `overall_score`: arithmetic mean over the 5 standard tasks; `null` if the task set differs.
- `suite_scores`: not used.
- `task_scores`: per-task success rates keyed `pick_cube`, `stack_cube`, `pick_single_ycb`, `pick_single_egad`, `pick_clutter_ycb` (snake_case).

## Checks
- Does the entry use exactly the 5 standard tasks? Other subsets → `null`.
- Is the averaging method (weighted vs arithmetic) recorded in `notes`? If unknown, note `'averaging method unknown'`.

## Methodology axes (record in `notes`, do not null)
- Averaging method: weighted vs arithmetic — papers sometimes differ. Record when known.
