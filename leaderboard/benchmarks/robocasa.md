---
benchmark: robocasa
---

## Protocol

- **Standard protocol**: 24 atomic tasks from the RoboCasa benchmark ([2406.02523](https://arxiv.org/abs/2406.02523)). `overall_score` = mean success rate across evaluated tasks.
- **Training data varies widely** (50–300 demos/task across papers). Always record `demos_per_task` and `task_count` in `notes`. Scores from different training budgets are not directly comparable.
- Entries evaluating non-standard task counts (e.g., 8 tasks, composite tasks) should note the deviation. Prefer `overall_score = null` for significantly non-standard subsets.
- Record episode count when known.

## Risky Patterns

- Are `demos_per_task` and `task_count` recorded in `notes`? Training budgets vary widely and are not directly comparable.
