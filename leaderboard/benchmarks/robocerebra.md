---
benchmark: robocerebra
---

## Protocol

- Embodied reasoning benchmark ([2506.06677](https://arxiv.org/abs/2506.06677)) with **6 evaluation dimensions**: ideal, memory_execution, memory_exploration, mix, observation_mismatching, random_disturbance.
- `overall_score` = arithmetic mean of all 6 dimensions. Set `overall_score = null` if fewer than 6 dimensions are reported. Always include `suite_scores` when available.
- **Architecture types**: Entries include end-to-end VLAs, hierarchical systems (VLM planner + controller), and oracle (GT-Plan) upper bounds. These are not directly comparable. Note the architecture type in `notes`.
- Oracle entries (GT-Plan + VLA) represent non-deployable upper bounds. They should be clearly marked.
- Typical scores: 5–20%. Small absolute differences may be meaningful. All current entries are from the original paper (600 rollouts, same protocol).

## Risky Patterns

- Are all 6 dimensions present? If not → `overall_score` must be `null`. Is the architecture type (end-to-end VLA, hierarchical, oracle) noted?
