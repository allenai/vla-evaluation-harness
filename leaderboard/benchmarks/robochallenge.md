---
benchmark: robochallenge
display_name: RoboChallenge
paper_url: https://arxiv.org/abs/2510.17950
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- progress_score
official_leaderboard: https://robochallenge.ai/leaderboard
detail_notes: "RoboChallenge (<a href='https://arxiv.org/abs/2510.17950'>2510.17950</a>). Scores synced from official leaderboard API. Manual paper extractions are not accepted."
---

**Standard**: Multi-task challenge benchmark with API-synced entries only; `overall_score` = success rate (binary task completion) and `suite_scores.progress_score` = partial credit for sub-goal progress.

## Scoring
- `overall_score`: binary task completion success rate from the upstream API.
- `suite_scores`: `progress_score` = partial-credit sub-goal progress. Both fields come from the API.
- `task_scores`: not used.

## Checks
- Is this entry API-synced? `curated_by` must end with `-api`. Manual paper extractions are forbidden and must be rejected entirely — not retained with `overall_score = null`.
