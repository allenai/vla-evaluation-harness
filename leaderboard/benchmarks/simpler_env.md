---
benchmark: simpler_env
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
