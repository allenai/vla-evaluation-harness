---
benchmark: simpler_env
display_name: SimplerEnv
paper_url: https://arxiv.org/abs/2405.05941
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- google_robot_vm
- google_robot_va
- widowx_vm
tasks:
- close_drawer_va
- move_near_va
- move_near_vm
- open_close_drawer_va
- open_close_drawer_vm
- open_drawer_va
- open_top_drawer_place_apple_vm
- pick_coke_can_va
- pick_coke_can_vm
- place_apple_in_drawer_va
- place_apple_in_drawer_vm
- put_carrot_on_plate_vm
- put_eggplant_in_basket_vm
- put_spoon_on_towel_vm
- reported_avg
- stack_block_vm
detail_notes: "3 independent evaluation dimensions (Google Robot VM, Google Robot VA, WidowX VM). Scores should never be averaged across them. <strong>Google Robot VM</strong> and <strong>VA</strong> are both standardized to the <strong>3-task average</strong> (Pick Coke Can, Move Near, Open/Close Drawer) for consistent ranking. Papers reporting 4 tasks (adding Place Apple in Drawer) have the 4th task shown in sub-scores; original 4-task averages are recorded in Notes. All <code>task_scores</code> keys must end with <code>_vm</code> or <code>_va</code> to indicate the evaluation protocol."
aggregation: forbidden
---

**Standard**: 3 independent evaluation dimensions — Google Robot Visual Matching, Google Robot Variant Aggregation, WidowX Visual Matching — reported separately; `overall_score` is always `null` by design because averaging across dimensions is forbidden.

## Scoring
- `overall_score`: always `null`. Any paper-reported cross-dimension aggregate goes in `task_scores.reported_avg`.
- `suite_scores`: canonical keys `google_robot_vm`, `google_robot_va`, `widowx_vm`. Each Google Robot key holds the **3-task average** (Pick Coke Can, Move Near, Open/Close Drawer) regardless of whether the paper reports 3 or 4 tasks — this keeps the ranking directly comparable across papers.
- `task_scores`: all keys MUST end in `_vm` or `_va` to disambiguate protocol (e.g. `pick_coke_can_vm`, `move_near_va`). WidowX tasks always use `_vm`. A 4th Google Robot task (Place Apple in Drawer) goes in `task_scores.place_apple_in_drawer_vm` / `_va`, never in `suite_scores`.

## Checks
- Is `overall_score` set to `null`? (Always — no exceptions.)
- Are VM, VA, and WidowX kept strictly in their own `suite_scores` keys with no cross-dimension math?
- Does `suite_scores.google_robot_vm` (and `_va`) hold the 3-task average, with any 4th task stored under `task_scores` and the original 4-task aggregate noted in `notes`?
- Do all `task_scores` keys end in `_vm` or `_va`?
- Is this genuinely SimplerEnv simulation and not a real-robot eval that reuses similar task names?

## Methodology axes (record in `notes`, do not null)
- Original paper aggregate: if the paper itself reports a 4-task Google Robot mean or a cross-dimension number, record the value and what it covered so the stored 3-task number is traceable.
