---
benchmark: robotwin_v1
---

## Protocol

- **v1 and v2 are separate benchmarks**. v1 = `robotwin_v1` ([2409.02920](https://arxiv.org/abs/2409.02920), ECCV 2024), v2 = `robotwin_v2` ([2506.18088](https://arxiv.org/abs/2506.18088), 2025).
- **v2 standard protocol**: `overall_score` = always `null`; use `suite_scores: {"easy": X, "hard": Y}`. Report both Easy (clean scenes) and Hard (5-axis domain randomization) when available.
- **v1**: No standard task set — entries evaluate 4–17 tasks. Set `overall_score = null` unless the entry matches the original paper's exact task set. Always record task count in `notes`. Entries with different task counts are not comparable.
- **v2**: Task counts vary (3–50). Record task count in `notes`.
- Do not file CVPR 2025 Challenge results under standard v2 (different protocol).
- **Two v2 training protocols exist** — scores across them are **not comparable**:

  | | Protocol A (official) | Protocol B (Motus-style) |
  |---|---|---|
  | Source | [2506.18088](https://arxiv.org/abs/2506.18088) | [2512.13030](https://arxiv.org/abs/2512.13030) |
  | Training | Single-task, 50 clean demos/task | Multi-task, 50 clean + 500 DR demos/task |
  | Training data | 2,500 total | 27,500 total (11×) |
  | Hard/Rand meaning | OOD generalization (never seen DR) | In-distribution (trained on DR) |
  | Typical Easy/Hard gap | 3–10× (e.g. 55% / 5%) | Near-zero (e.g. 93% / 92%) |

  Always record which protocol in `notes` (prefix with `Protocol A` or `Protocol B`).

## Risky Patterns

- Does the task count match the original paper's exact set? If not → `overall_score` must be `null`.
