---
benchmark: roboarena
display_name: RoboArena
paper_url: https://arxiv.org/abs/2506.18123
metric:
  name: elo_rating
  unit: Elo
  range:
  - 0
  - 2000
  higher_is_better: true
official_leaderboard: https://robo-arena.github.io/
external_only: true
detail_notes: "RoboArena (<a href='https://arxiv.org/abs/2506.18123'>2506.18123</a>). Elo-based ranking via pairwise human evaluation of real-robot policies. Rankings live on the official leaderboard; this site does not mirror them."
---

**External-only**: results are maintained exclusively on the [official leaderboard](https://robo-arena.github.io/). This registry entry exists to link out; `leaderboard.json` must contain **zero** rows for this benchmark.

Elo ratings change with every pairwise match, so any snapshot copied here is immediately stale. A bi-weekly API mirror ran until 2026-07 and was retired for that reason.

## Checks
- Any candidate row for this benchmark must be rejected entirely, whatever its source (paper extraction or API). Do not retain rows with `overall_score = null`.
