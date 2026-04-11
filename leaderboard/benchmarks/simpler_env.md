---
benchmark: simpler_env
---

## Protocol

- **Standard protocol**: 3 independent evaluation dimensions — **never average across them**. `overall_score` = always `null`; use `suite_scores` only. Store the paper's reported aggregate (if any) in `task_scores.reported_avg` per the global rule (see Result Fields).

| Dimension | Robot | Protocol | Benchmark key |
|-----------|-------|----------|---------------|
| Google Robot VM | Google Robot | Visual Matching | `suite_scores.google_robot_vm` |
| Google Robot VA | Google Robot | Variant Aggregation | `suite_scores.google_robot_va` |
| WidowX VM | WidowX (Bridge) | Visual Matching | `suite_scores.widowx_vm` |
- **Google Robot VM standardization**: `suite_scores.google_robot_vm` must always store the **3-task average** (Pick Coke Can, Move Near, Open/Close Drawer) for consistent ranking. Papers reporting 4 tasks (adding Place Apple in Drawer) should store the 4th task in `task_scores.place_apple_in_drawer_vm` and note the original 4-task average in `notes`. This ensures apples-to-apples comparison since 3-task is the dominant protocol (used by ~80% of papers).
- **Google Robot VA standardization**: `suite_scores.google_robot_va` follows the same rule — always store the **3-task average** (Pick Coke Can, Move Near, Open/Close Drawer). Papers reporting 4 tasks store the 4th in `task_scores.place_apple_in_drawer_va`. This ensures VM and VA scores are directly comparable.
- **task_scores protocol suffix**: All SimplerEnv `task_scores` keys **must** end with `_vm` or `_va` to indicate the evaluation protocol (e.g., `pick_coke_can_vm`, `move_near_va`). WidowX tasks always use `_vm`. `validate.py` enforces this. This prevents ambiguity since VM and VA evaluate the same tasks under different protocols with different scores.
- Don't confuse real-robot scores (e.g. OpenVLA's 12-task real eval) with SimplerEnv simulation.

## Risky Patterns

- Are VM/VA/WidowX dimensions kept strictly separate? Averaging across them is forbidden.
- Does `suite_scores.google_robot_vm` hold the 3-task average (Pick Coke Can, Move Near, Open/Close Drawer)?
- Do all `task_scores` keys end in `_vm` or `_va`?
