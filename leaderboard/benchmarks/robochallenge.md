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
official_leaderboard: https://robochallenge.ai/leaderboard
external_only: true
detail_notes: "RoboChallenge (<a href='https://arxiv.org/abs/2510.17950'>2510.17950</a>). Real-robot multi-task challenge with submission-based evaluation. Results live on the official leaderboard; this site does not mirror them."
---

**External-only**: results are maintained exclusively on the [official leaderboard](https://robochallenge.ai/leaderboard). This registry entry exists to link out; `leaderboard.json` must contain **zero** rows for this benchmark.

A bi-weekly API mirror ran until 2026-07 and was retired: submissions arrive on the official board's own cadence, and mirrored rows needed manual model-to-paper mapping that decayed without attention.

## Checks
- Any candidate row for this benchmark must be rejected entirely, whatever its source (paper extraction or API). Do not retain rows with `overall_score = null`.
