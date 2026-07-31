---
benchmark: molmospaces
display_name: MolmoSpaces-Bench
paper_url: https://arxiv.org/abs/2602.11337
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
official_leaderboard: https://molmospaces.allen.ai/leaderboard
external_only: true
detail_notes: "&ldquo;MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation&rdquo; (<a href='https://arxiv.org/abs/2602.11337'>2602.11337</a>). Results live on the official leaderboard; this site does not mirror them."
---

**External-only**: results are maintained exclusively on the [official leaderboard](https://molmospaces.allen.ai/leaderboard). This registry entry exists to link out; `leaderboard.json` must contain **zero** rows for this benchmark.

The official board ranks policies per task and separates entries that trained on MolmoSpaces in-distribution data from those that did not, so a flat mirror here would drop that distinction. vla-eval integrates MolmoSpaces-Bench for running evaluations (see `configs/benchmarks/molmospaces/`), which is independent of this leaderboard entry.

If papers begin reporting MolmoSpaces-Bench numbers routinely, paper extraction can be enabled by removing `external_only` and defining the full protocol here.

## Checks
- Any candidate row for this benchmark must be rejected entirely while `external_only` is set. Do not retain rows with `overall_score = null`.
