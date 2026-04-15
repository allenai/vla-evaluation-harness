---
benchmark: robocerebra
display_name: RoboCerebra
paper_url: https://arxiv.org/abs/2506.06677
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- ideal
- memory_execution
- memory_exploration
- mix
- observation_mismatching
- random_disturbance
tasks:
- Ideal
- Memory_Execution
- Memory_Exploration
- Mix
- Observation_Mismatching
- Random_Disturbance
detail_notes: "Embodied reasoning benchmark (<a href='https://arxiv.org/abs/2506.06677'>2506.06677</a>) with 6 evaluation dimensions. <code>overall_score</code> = mean of 6 dimensions. <strong>Architecture types</strong>: end-to-end VLAs, hierarchical (VLM+controller), and oracle (GT-Plan) upper bounds are not directly comparable. Check <em>notes</em> for architecture type."
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
