---
benchmark: calvin
---

**Standard**: CALVIN ABC→D split (train A/B/C, evaluate on D) over 1000 evaluation chains; `overall_score` = average number of completed subtasks in chain of 5 (range 0–5).

## Scoring
- `overall_score`: `avg_len` metric (0–5) over 1000 eval chains; `null` if the metric is not `avg_len` or the split is not ABC→D.
- `suite_scores`: optional — some papers report per-chain-length success rates; store under keys like `chain_1`, `chain_2`, ..., `chain_5` when provided.
- `task_scores`: not used — CALVIN's canonical metric is sequence-level, not task-level.

## Checks
- Is the training split `ABC→D`? `ABCD→D` and `D→D` inflate scores and must be `null`.
- Is the reported metric `avg_len` (0–5)? Rows reporting only success rate percentages without `avg_len` → `null`.
- Does the evaluation use 1000 chains? Any deviation must be recorded in `notes`.

## Methodology axes (record in `notes`, do not null)
- Chain count deviation: note if a paper evaluates on fewer/more than 1000 chains.
- Training data source: CALVIN language-annotated vs full-play subset.
