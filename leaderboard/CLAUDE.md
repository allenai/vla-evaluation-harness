# leaderboard/

This directory is a **standalone static leaderboard site** — it is NOT vla-eval output.

## Key distinction

- `leaderboard/data/leaderboard.json` contains scores **reported in published papers**. These are manually curated or produced by the `extract.py` / `refine.py` pipeline — not produced by running `vla-eval`.
- `vla-eval` is the evaluation harness (the parent repo). Its runtime outputs go to `results/` or user-specified paths, never here.

Do not conflate leaderboard data with vla-eval reproduced results. They are independent.
