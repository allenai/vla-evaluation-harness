---
benchmark: mikasa
display_name: MIKASA-Robo
paper_url: https://arxiv.org/abs/2502.10550
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
tasks:
- InterceptMedium
- RememberColor3
- RememberColor5
- RememberColor9
- ShellGameTouch
detail_notes: "Standard protocol: 5-task VLA evaluation (ShellGameTouch, InterceptMedium, RememberColor3/5/9) from the MIKASA paper, endorsed by MemoryVLA (ICLR 2026). <code>overall_score</code> = arithmetic mean of 5 tasks. Entries using non-standard task sets (e.g., ELMUR 4-task) have <code>overall_score</code> = null."
aggregation:
  container: task_scores
  keys:
  - ShellGameTouch
  - InterceptMedium
  - RememberColor3
  - RememberColor5
  - RememberColor9
---

**Standard**: 5-task VLA evaluation ([2502.10550](https://arxiv.org/abs/2502.10550)) on ShellGameTouch, InterceptMedium, RememberColor3, RememberColor5, RememberColor9 at 100 episodes per task; `overall_score` = arithmetic mean of the 5 task success rates.

## Scoring
- `overall_score`: arithmetic mean over the 5 standard tasks; `null` for non-standard task sets.
- `suite_scores`: not used at this level — exception: `suite_scores.reported_avg` stores the paper's own aggregate when the entry uses a non-standard set.
- `task_scores`: canonical keys `ShellGameTouch`, `InterceptMedium`, `RememberColor3`, `RememberColor5`, `RememberColor9`.

## Checks
- Is this the 5-task VLA protocol (ShellGameTouch, InterceptMedium, RememberColor3/5/9)? The ELMUR 4-task variant (RC3/5/9 + TakeItBack) → `overall_score = null`, paper aggregate stored in `suite_scores.reported_avg`.
- Are all 5 tasks present in `task_scores` with their own success rates?
- Is a non-standard aggregate preserved in `suite_scores.reported_avg` rather than `overall_score`?

## Methodology axes (record in `notes`, do not null)
- Evaluation episode count: standard is 100 per task. Record deviations.
