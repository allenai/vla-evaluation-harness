---
benchmark: robotwin_v2
display_name: RoboTwin 2.0
paper_url: https://arxiv.org/abs/2506.18088
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- easy
- hard
tasks:
- beat_block_hammer
- click_bell
- handover_mic
- move_can_pot
- move_pillbottles_pad
- move_playingcard_away
- pick_diverse_bottles
- place_burger_fries
- place_container_plate
- place_mouse_pad
- place_phone_stand
- shake_bottle
detail_notes: "RoboTwin 2.0 (2025). Easy = clean scenes; Hard = 5-axis domain randomization. <strong>Two training protocols exist — scores across them are not directly comparable.</strong> Protocol A (official, 2506.18088): single-task, 50 clean demos — Hard tests OOD generalization (3–10× gaps). Protocol B (Motus-style, 2512.13030): multi-task, 50 clean + 500 DR demos/task — Hard is in-distribution (near-zero gaps). Check entry notes for protocol."
aggregation: forbidden
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
