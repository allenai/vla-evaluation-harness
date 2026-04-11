---
benchmark: libero_mem
---

## Protocol

- Memory benchmark ([2511.11478](https://arxiv.org/abs/2511.11478)) with **10 tasks** (T1–T10) across 4 types: OM (T1–T2), OS (T3–T5), OR (T6–T8), OO (T9–T10).
- **Metric**: subgoal completion rate (%), NOT task success rate. 20 rollouts per task.
- `overall_score` = unweighted arithmetic mean of T1–T10. Always include `task_scores`.
- Models using oracle subgoal information must note this in `notes`.

## Risky Patterns

- Are all 10 tasks (T1–T10) present and is the metric subgoal completion rate (not task success rate)?
