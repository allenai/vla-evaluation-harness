---
benchmark: rlbench
---

## Protocol

- **Standard protocol**: 18-task PerAct subset ([2209.05451](https://arxiv.org/abs/2209.05451)), 249 total language-goal variations across the 18 tasks, 25 evaluation episodes per task (450 total), 100 training demos per task.
- `overall_score` = mean success rate across 18 tasks. Set `overall_score = null` for non-18-task evaluations. Always record task count in `notes`.
- **Variation count matters**: Multi-variation (e.g. 25 per task) vs single variation significantly affects scores. Record variation count in `notes` when known.
- Entries using single-task learning (training a separate policy per task) are not comparable to multi-task entries. Note the training regime.

## Risky Patterns

- Does the entry follow the 18-task PerAct subset with multi-variation eval? Non-18-task runs → `overall_score` must be `null`.
