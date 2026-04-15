---
benchmark: kinetix
display_name: Kinetix
paper_url: https://arxiv.org/abs/2506.07339
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- d0
- d1
- d2
- d3
- d4
detail_notes: "12-task eval protocol from the RTC paper (<a href='https://arxiv.org/abs/2506.07339'>2506.07339</a>). State-based, no vision/language. Scores depend on (inference_delay <em>d</em>, execution_horizon <em>e</em>). Entries at different <em>d</em> values are <strong>not directly comparable</strong> (~11pp gap between d=0 and d=4 for the same method)."
---

**Standard**: 12-task state-based eval protocol from the RTC paper ([2506.07339](https://arxiv.org/abs/2506.07339)), no vision/language input; `overall_score` = mean success rate across the 12 tasks.

## Scoring
- `overall_score`: arithmetic mean over the 12 tasks; `null` if fewer than 12 tasks or a different task set.
- `suite_scores`: not used.
- `task_scores`: per-task success rates keyed by task name.

## Checks
- Is this the 12-task RTC protocol (not the standalone Kinetix simulator with different tasks)?
- Are `inference_delay d` and `execution_horizon e` recorded in `notes`? An unlabeled entry cannot be placed correctly.
- Are all 12 tasks evaluated? Partial sets → `null`.

## Methodology axes (record in `notes`, do not null)
- `inference_delay d` (0, 2, 4, ...): scores at different `d` differ by ~11pp for the same method — direct cross-`d` comparison is misleading. Record the value; group by `d` when presenting.
- `execution_horizon e`: also affects scores. Record the value.
