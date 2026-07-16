---
benchmark: robocasa365
display_name: RoboCasa365
paper_url: https://arxiv.org/abs/2603.04356
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
official_leaderboard: https://robocasa.ai/leaderboard.html
external_only: true
detail_notes: "&ldquo;RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots&rdquo; (<a href='https://arxiv.org/abs/2603.04356'>2603.04356</a>). Results live on the official leaderboard; this site does not mirror them. Distinct protocol from the <code>robocasa</code> entry."
---

**External-only**: results are maintained exclusively on the [official leaderboard](https://robocasa.ai/leaderboard.html) (submissions via PR, reviewed by the RoboCasa team). This registry entry exists to link out; `leaderboard.json` must contain **zero** rows for this benchmark.

Not the same protocol as the `robocasa` entry (original two-embodiment 24-task benchmark): RoboCasa365 pretrains on 300 tasks across 2,500 kitchens and evaluates a 50-task multi-task split. Scores are not comparable across the two.

If papers begin reporting RoboCasa365 numbers routinely, paper extraction can be enabled by removing `external_only` and defining the full protocol here.

## Checks
- Any candidate row for this benchmark must be rejected entirely while `external_only` is set. Do not retain rows with `overall_score = null`.
- Do not route original-RoboCasa (Panda/GR1 24-task) results here; those belong to `robocasa`.
