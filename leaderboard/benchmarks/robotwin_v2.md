---
benchmark: robotwin_v2
display_name: RoboTwin 2.0
paper_url: https://arxiv.org/abs/2506.18088
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
official_leaderboard: https://robotwin-platform.github.io/leaderboard
external_only: true
detail_notes: "&ldquo;RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation&rdquo; (<a href='https://arxiv.org/abs/2506.18088'>2506.18088</a>). Results live on the official leaderboard; this site does not mirror them."
---

**External-only**: results are maintained exclusively on the [official leaderboard](https://robotwin-platform.github.io/leaderboard) (fixed protocol: 50 `demo_clean` demos × 50 tasks, 100 trials/task under clean and randomized scenes; co-train and single-task settings; public code + weights + report required). This registry entry exists to link out; `leaderboard.json` must contain **zero** rows for this benchmark.

Paper-reported rows were retired in 2026-09. Papers mixed incompatible training regimes (clean-only vs. clean + domain-randomized demos, where Hard becomes in-distribution), task subsets of 3–50, and self-reported numbers that disagree with the official reproductions; only ~31 of 395 rows were mutually comparable. RoboTwin 1.0 is a separate benchmark at `robotwin_v1` and is unaffected.

## Checks
- Any candidate row for this benchmark must be rejected entirely while `external_only` is set. Do not retain rows with `overall_score = null`.
- RoboTwin 1.0 results still go to `robotwin_v1`.
