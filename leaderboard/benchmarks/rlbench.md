---
benchmark: rlbench
---

**Standard**: 18-task PerAct subset ([2209.05451](https://arxiv.org/abs/2209.05451)) with 249 total language-goal variations across the 18 tasks, 25 evaluation episodes per task (450 total), 100 training demos per task, multi-task learning; `overall_score` = mean success rate across the 18 tasks.

## Scoring
- `overall_score`: arithmetic mean over the 18 tasks; `null` for non-18-task evaluations.
- `suite_scores`: not used.
- `task_scores`: per-task success rates keyed by PerAct task name.

## Checks
- Does the entry follow the 18-task PerAct subset? Fewer or different tasks → `null`.
- Is this multi-task learning (one policy for all 18)? Single-task-per-policy training is not directly comparable — record in `notes`.
- Is the variation count recorded in `notes` when known (multi-variation e.g. 25 per task vs single-variation)?
- Is the training demo count recorded?

## Methodology axes (record in `notes`, do not null)
- Variation count: multi-variation (e.g. 25 per task) vs single-variation affects scores significantly. Record when known.
- Training regime: multi-task (standard) vs one-policy-per-task. The latter is a valid entry but must be annotated.
- Training demo count: standard is 100 demos/task. Record deviations.
