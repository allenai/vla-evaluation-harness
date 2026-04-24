---
benchmark: robotwin_v1
display_name: RoboTwin 1.0
paper_url: https://arxiv.org/abs/2409.02920
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
tasks:
- block_hammer_beat
- block_hammer_beat_20demo
- block_hammer_beat_50demo
- block_handover
- block_handover_20demo
- block_handover_50demo
- blocks_stack
- blocks_stack_easy
- blocks_stack_easy_20demo
- blocks_stack_easy_50demo
- blocks_stack_hard
- blocks_stack_hard_20demo
- blocks_stack_hard_50demo
- bottle_adjust
- container_place
- container_place_20demo
- container_place_50demo
- diverse_bottles_pick
- diverse_bottles_pick_20demo
- diverse_bottles_pick_50demo
- dual_bottles_pick_easy
- dual_bottles_pick_hard
- dual_shoes_place
- empty_cup_place
- empty_cup_place_messy
- mug_hanging_easy
- mug_hanging_hard
- mug_hanging_hard_20demo
- mug_hanging_hard_50demo
- pick_apple_messy
- pick_apple_messy_20demo
- pick_apple_messy_50demo
- put_apple_cabinet
- shoe_place
- tool_adjust
detail_notes: "RoboTwin 1.0 (ECCV 2024). Task counts vary by paper (4–17). Scores across different task subsets are not directly comparable."
---

**Standard**: RoboTwin v1 ([2409.02920](https://arxiv.org/abs/2409.02920), ECCV 2024) with no fixed task set — entries evaluate 4–17 tasks from the original paper; `overall_score` = mean success rate across the evaluated tasks ONLY when the task set matches the original paper's exact set, otherwise `null`. (v1 and v2 are separate benchmarks — v2 lives at `robotwin_v2`.)

## Scoring
- `overall_score`: arithmetic mean over the evaluated tasks; `null` unless the task set matches the original paper's exact set.
- `suite_scores`: optional per-task-family groupings when provided.
- `task_scores`: per-task success rates keyed by task name.

## Checks
- Does the task count match the original RoboTwin v1 paper's exact set? If not → `overall_score = null`.
- Is the task count recorded in `notes`?
- Is this v1 (not v2)? v2 results must go to `robotwin_v2`.

## Methodology axes (record in `notes`, do not null)
- Task count: varies across papers (4–17 tasks on v1). Entries with different counts are not comparable; record the exact count.
