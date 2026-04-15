---
benchmark: rlbench
display_name: RLBench
paper_url: https://arxiv.org/abs/1909.12271
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- close_box
- close_jar
- drag_stick
- insert_block
- insert_peg
- lamp_on
- meat_off_grill
- open_box
- open_drawer
- open_fridge
- open_jar
- open_wine
- phone_on_base
- pick_and_lift
- pickup_cup
- place_cups
- place_wine
- play_jenga
- push_block
- push_buttons
- put_block
- put_in_cupboard
- put_in_drawer
- put_in_safe
- put_knife
- screw_bulb
- slide_block
- sort_mustard
- sort_shape
- stack_blocks
- stack_cups
- sweep_to_dustpan
- take_umbrella
- turn_tap
tasks:
- close_jar
- drag_stick
- insert_peg
- meat_off_grill
- open_drawer
- place_cups
- place_wine
- push_buttons
- put_in_cupboard
- put_in_drawer
- put_in_safe
- screw_bulb
- slide_block
- sort_shape
- stack_blocks
- stack_cups
- sweep_to_dustpan
- turn_tap
detail_notes: "Standard: 18-task PerAct subset (<a href='https://arxiv.org/abs/2209.05451'>2209.05451</a>), 249 total variations, 25 eval episodes/task, 100 demos/task. <strong>Only 18-task entries have a sortable overall score</strong> — other task counts are shown but not ranked. Multi-variation vs single-variation significantly affects scores."
---

**Standard**: 18-task PerAct subset ([2209.05451](https://arxiv.org/abs/2209.05451)) with 249 total language-goal variations across the 18 tasks, 25 evaluation episodes per task (450 total), 100 training demos per task, multi-task learning; `overall_score` = mean success rate across the 18 tasks.

## Scoring
- `overall_score`: arithmetic mean over the 18 tasks; `null` for non-18-task evaluations.
- `suite_scores`: not used.
- `task_scores`: per-task success rates keyed by PerAct task name.

## Checks
- Does the entry follow the 18-task PerAct subset? Fewer or different tasks → `null`.
- Is this multi-task learning (one policy for all 18)? Single-task-per-policy training is not directly comparable — record in `notes`.
- Is the variation count recorded in `notes` when known (multi-variation e.g. 25 per task vs single-variation)?
- Is the training demo count recorded?

## Methodology axes (record in `notes`, do not null)
- Variation count: multi-variation (e.g. 25 per task) vs single-variation affects scores significantly. Record when known.
- Training regime: multi-task (standard) vs one-policy-per-task. The latter is a valid entry but must be annotated.
- Training demo count: standard is 100 demos/task. Record deviations.
