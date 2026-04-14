---
benchmark: robotwin_v2
---

**Standard**: RoboTwin v2 ([2506.18088](https://arxiv.org/abs/2506.18088), 2025) reported as separate Easy (clean scenes) and Hard (5-axis domain randomization) scores; `overall_score` is always `null` by design and the two numbers live in `suite_scores.easy` and `suite_scores.hard`. (v1 is a separate benchmark at `robotwin_v1`.)

## Scoring
- `overall_score`: always `null`.
- `suite_scores`: `easy` (clean) and `hard` (domain randomization). Report both when available; a single-difficulty entry stores only the reported key.
- `task_scores`: optional per-task success rates when the paper tabulates them.

## Checks
- Is `overall_score` set to `null` with the numbers in `suite_scores.easy` / `suite_scores.hard`?
- Is the `notes` field prefixed with `Protocol A` or `Protocol B` (see Methodology axes)? An unlabeled entry cannot be correctly placed.
- Is this standard v2 and NOT a CVPR 2025 Challenge result? Challenge results follow a different protocol and must not be filed here.
- Is the task count (3–50 varies) recorded in `notes`?

## Methodology axes (record in `notes`, do not null)
- Training protocol tag: every entry must be prefixed `Protocol A` or `Protocol B`. The two protocols are NOT comparable — scores from one cannot be ranked against the other.

  | | Protocol A (official) | Protocol B (Motus-style) |
  |---|---|---|
  | Source | [2506.18088](https://arxiv.org/abs/2506.18088) | [2512.13030](https://arxiv.org/abs/2512.13030) |
  | Training | Single-task, 50 clean demos/task | Multi-task, 50 clean + 500 DR demos/task |
  | Training data | 2,500 total | 27,500 total (11×) |
  | Hard/Rand meaning | OOD generalization (never seen DR) | In-distribution (trained on DR) |
  | Typical Easy/Hard gap | 3–10× (e.g. 55% / 5%) | Near-zero (e.g. 93% / 92%) |

- Task count: v2 papers evaluate anywhere from 3 to 50 tasks; record the exact count.
