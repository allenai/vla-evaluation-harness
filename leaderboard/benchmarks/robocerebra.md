---
benchmark: robocerebra
---

**Standard**: Embodied reasoning benchmark ([2506.06677](https://arxiv.org/abs/2506.06677)) with 6 evaluation dimensions (ideal, memory_execution, memory_exploration, mix, observation_mismatching, random_disturbance); `overall_score` = arithmetic mean of all 6 dimensions.

## Scoring
- `overall_score`: arithmetic mean of the 6 suite keys; `null` if fewer than 6 dimensions reported.
- `suite_scores`: canonical keys `ideal`, `memory_execution`, `memory_exploration`, `mix`, `observation_mismatching`, `random_disturbance`.
- `task_scores`: not used.

## Checks
- Are all 6 dimensions present? Missing any → `null`.
- Is the architecture type recorded in `notes`? (end-to-end VLA / hierarchical / oracle)
- For oracle entries (GT-Plan + VLA): is the upper-bound / non-deployable status clearly marked?

## Methodology axes (record in `notes`, do not null)
- Architecture type: end-to-end VLA, hierarchical (VLM planner + controller), or oracle (GT-Plan upper bound). These are not directly comparable — readers must group by architecture. Oracle entries are non-deployable upper bounds.
- Typical score range is 5–20%, so small absolute differences can be meaningful. Current entries use 600 rollouts; record deviations.
