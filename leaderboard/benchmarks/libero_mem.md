---
benchmark: libero_mem
display_name: LIBERO-Mem
paper_url: https://arxiv.org/abs/2511.11478
metric:
  name: subgoal_completion_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
tasks:
- T1
- T10
- T2
- T3
- T4
- T5
- T6
- T7
- T8
- T9
detail_notes: "LIBERO-Mem memory benchmark (<a href='https://arxiv.org/abs/2511.11478'>2511.11478</a>). 10 tasks (T1–T10). Metric: <strong>subgoal completion rate</strong> (not task success rate). 20 rollouts per task."
aggregation:
  container: task_scores
  keys:
  - T1
  - T2
  - T3
  - T4
  - T5
  - T6
  - T7
  - T8
  - T9
  - T10
---

**Standard**: LIBERO memory benchmark ([2511.11478](https://arxiv.org/abs/2511.11478)) — 10 tasks (T1–T10) across 4 memory types (OM: T1–T2, OS: T3–T5, OR: T6–T8, OO: T9–T10) with subgoal completion rate as the metric; `overall_score` = unweighted arithmetic mean of T1–T10.

## Scoring
- `overall_score`: arithmetic mean of `task_scores.T1` through `task_scores.T10`; `null` if any task is missing.
- `suite_scores`: optional — use memory-type groupings (`OM`, `OS`, `OR`, `OO`) when provided.
- `task_scores`: canonical keys `T1` through `T10`. Values are subgoal completion rates in percent.

## Checks
- Are all 10 tasks (T1–T10) present?
- Is the metric subgoal completion rate (not task success rate)? Success-rate reporters must be nulled.
- Is the rollout count (20 per task per the standard) recorded in `notes`?
- Is oracle subgoal usage disclosed in `notes` when applicable?

## Methodology axes (record in `notes`, do not null)
- Oracle subgoal access: models using oracle subgoal information are valid entries but must be annotated.
