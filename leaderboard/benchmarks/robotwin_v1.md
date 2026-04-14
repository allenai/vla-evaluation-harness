---
benchmark: robotwin_v1
---

**Standard**: RoboTwin v1 ([2409.02920](https://arxiv.org/abs/2409.02920), ECCV 2024) with no fixed task set — entries evaluate 4–17 tasks from the original paper; `overall_score` = mean success rate across the evaluated tasks ONLY when the task set matches the original paper's exact set, otherwise `null`. (v1 and v2 are separate benchmarks — v2 lives at `robotwin_v2`.)

## Scoring
- `overall_score`: arithmetic mean over the evaluated tasks; `null` unless the task set matches the original paper's exact set.
- `suite_scores`: optional per-task-family groupings when provided.
- `task_scores`: per-task success rates keyed by task name.

## Checks
- Does the task count match the original RoboTwin v1 paper's exact set? If not → `overall_score = null`.
- Is the task count recorded in `notes`?
- Is this v1 (not v2)? v2 results must go to `robotwin_v2`.

## Methodology axes (record in `notes`, do not null)
- Task count: varies across papers (4–17 tasks on v1). Entries with different counts are not comparable; record the exact count.
