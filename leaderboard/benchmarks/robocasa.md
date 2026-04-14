---
benchmark: robocasa
---

**Standard**: 24 atomic tasks on a Mobile Franka robot in RoboCasa kitchen environments ([2406.02523](https://arxiv.org/abs/2406.02523)); `overall_score` = mean success rate across the 24 tasks.

## Scoring
- `overall_score`: arithmetic mean of success rates over the 24 atomic tasks; `null` if the evaluated set is not the full 24.
- `suite_scores`: optional — use when a paper groups tasks by category (e.g. pick-and-place, open/close).
- `task_scores`: per-task success rates keyed by the atomic task name when the paper tabulates them.

## Checks
- Is the embodiment a Mobile Franka robot in a RoboCasa kitchen environment? Alternative embodiments or environments (GR1 Tabletop humanoid, other tabletop variants, non-kitchen scenes) are NOT this benchmark — `overall_score` must be `null`, and the row likely belongs to a different benchmark.
- Does the entry evaluate the full 24 atomic tasks? Subsets (< 24), supersets (composite + atomic), or relabeled sets → `overall_score = null`.
- Are `demos_per_task`, `trials_per_task`, and demo source recorded in `notes` when the paper states them?

## Methodology axes (record in `notes`, do not null)
- Training demo budget: papers report anywhere from 50 to 3000 demos/task. All budgets within the 24-task protocol are valid; record `demos_per_task` so readers can account for the budget. Scores across different budgets are not directly comparable.
- Evaluation trials per task: varies (50, 100, 150, ...). Record the per-task trial count when reported.
- Demo source: human teleoperation / synthetic / generated. Record when disclosed.
- `weight_type`: shared (same checkpoint across benchmarks) vs finetuned (trained on this benchmark's data).
