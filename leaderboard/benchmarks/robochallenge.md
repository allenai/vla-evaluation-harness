---
benchmark: robochallenge
---

**Standard**: Multi-task challenge benchmark with API-synced entries only; `overall_score` = success rate (binary task completion) and `suite_scores.progress_score` = partial credit for sub-goal progress.

## Scoring
- `overall_score`: binary task completion success rate from the upstream API.
- `suite_scores`: `progress_score` = partial-credit sub-goal progress. Both fields come from the API.
- `task_scores`: not used.

## Checks
- Is this entry API-synced? `curated_by` must end with `-api`. Manual paper extractions are forbidden and must be rejected entirely — not retained with `overall_score = null`.
