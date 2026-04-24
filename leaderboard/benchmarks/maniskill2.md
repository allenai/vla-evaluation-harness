---
benchmark: maniskill2
display_name: ManiSkill2
paper_url: https://arxiv.org/abs/2302.04659
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
tasks:
- AssemblingKits
- CloseCabinetDoor
- Excavate
- Fill
- Hang
- LiftCube
- MoveBucket
- OpenCabinetDoor
- OpenCabinetDrawer
- PegInsertionSide
- PickClutterYCB
- PickCube
- PickSingleEGAD
- PickSingleYCB
- Pinch
- PlugCharger
- Pour
- PushChair
- PushCube
- StackCube
- TurnFaucet
- TurnFaucetLeft
- TurnFaucetRight
- Write
aggregation:
  container: task_scores
  keys:
  - PickCube
  - StackCube
  - PickSingleYCB
  - PickSingleEGAD
  - PickClutterYCB
detail_notes: "Task subsets vary significantly — 14+ different combinations exist across papers. The most common 5-task set (PickCube, StackCube, PickSingleYCB, PickSingleEGAD, PickClutterYCB) covers ~24% of entries. A second common 5-task set (Fill, Hang, PickCube, PickSingleYCB, StackCube) covers ~19%. <strong>Scores across different task subsets are not directly comparable.</strong> Averaging methods also vary (weighted vs arithmetic). Check notes for which tasks were evaluated. <strong>Only entries using the standard 5-task set have a sortable overall score</strong> — other task subsets are shown but not ranked."
---

**Standard**: ManiSkill2 5-task set (`PickCube`, `StackCube`, `PickSingleYCB`, `PickSingleEGAD`, `PickClutterYCB`); `overall_score` = arithmetic mean across those 5 tasks, and `null` for any other subset.

## Scoring
- `overall_score`: arithmetic mean over the 5 standard `task_scores` keys; `null` if any of the 5 is missing or a different subset is used.
- `suite_scores`: not used — ManiSkill2 has no canonical sub-suites at this level.
- `task_scores`: per-task success rates keyed by the PascalCase task name exactly as it appears in the ManiSkill2 codebase (`PickCube`, `StackCube`, `PickSingleYCB`, `PickSingleEGAD`, `PickClutterYCB`, and any of the other names listed in `tasks` above).

## Checks
- Does the entry use exactly the 5 standard tasks? Other subsets → `null`.
- Is the averaging method (weighted vs arithmetic) recorded in `notes`? If unknown, note `'averaging method unknown'`.

## Methodology axes (record in `notes`, do not null)
- Averaging method: weighted vs arithmetic — papers sometimes differ. Record when known.
