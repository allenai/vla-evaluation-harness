---
benchmark: robodojo
display_name: RoboDojo
paper_url: https://arxiv.org/abs/2607.04434
metric:
  name: score
  unit: pts
  range:
  - 0
  - 100
  higher_is_better: true
official_leaderboard: https://robodojo-benchmark.com/leaderboard
external_only: true
detail_notes: "&ldquo;RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies&rdquo; (<a href='https://arxiv.org/abs/2607.04434'>2607.04434</a>). Results live on the official leaderboard; this site does not mirror them."
---

**External-only**: results are maintained exclusively on the [official leaderboard](https://robodojo-benchmark.com/leaderboard) (frozen protocol, 3 training seeds). This registry entry exists to link out; `leaderboard.json` must contain **zero** rows for this benchmark.

The benchmark reports two metrics per dimension, score (partial progress) and success rate, both ×100; the overall number averages the five dimension means. vla-eval integrates RoboDojo for running evaluations (see `configs/benchmarks/robodojo/`), which is independent of this leaderboard entry.

If papers begin reporting RoboDojo numbers routinely, paper extraction can be enabled by removing `external_only` and defining the full protocol here.

## Checks
- Any candidate row for this benchmark must be rejected entirely while `external_only` is set. Do not retain rows with `overall_score = null`.
