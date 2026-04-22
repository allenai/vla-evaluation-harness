---
benchmark: robocasa
display_name: RoboCasa
paper_url: https://arxiv.org/abs/2406.02523
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- panda
- gr1
tasks:
- PnPCupToDrawerClose_gr1
- PnPPotatoToMicrowaveClose_gr1
- PnPMilkToMicrowaveClose_gr1
- PnPBottleToCabinetClose_gr1
- PnPWineToCabinetClose_gr1
- PnPCanToDrawerClose_gr1
- PosttrainPnPNovelFromCuttingboardToBasketSplitA_gr1
- PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_gr1
- PosttrainPnPNovelFromCuttingboardToPanSplitA_gr1
- PosttrainPnPNovelFromCuttingboardToPotSplitA_gr1
- PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_gr1
- PosttrainPnPNovelFromPlacematToBasketSplitA_gr1
- PosttrainPnPNovelFromPlacematToBowlSplitA_gr1
- PosttrainPnPNovelFromPlacematToPlateSplitA_gr1
- PosttrainPnPNovelFromPlacematToTieredshelfSplitA_gr1
- PosttrainPnPNovelFromPlateToBowlSplitA_gr1
- PosttrainPnPNovelFromPlateToCardboardboxSplitA_gr1
- PosttrainPnPNovelFromPlateToPanSplitA_gr1
- PosttrainPnPNovelFromPlateToPlateSplitA_gr1
- PosttrainPnPNovelFromTrayToCardboardboxSplitA_gr1
- PosttrainPnPNovelFromTrayToPlateSplitA_gr1
- PosttrainPnPNovelFromTrayToPotSplitA_gr1
- PosttrainPnPNovelFromTrayToTieredbasketSplitA_gr1
- PosttrainPnPNovelFromTrayToTieredshelfSplitA_gr1
score_key_suffixes:
- panda
- gr1
aggregation: forbidden
detail_notes: "Two embodiments share a single benchmark: <strong>Panda</strong> (Mobile Franka in RoboCasa kitchens, <a href='https://arxiv.org/abs/2406.02523'>2406.02523</a>, 24 atomic tasks) and <strong>GR1</strong> (Fourier GR1 humanoid tabletop pick-and-place, <a href='https://arxiv.org/abs/2503.14734'>2503.14734</a>, 24 PnP tasks). Scores across embodiments are not directly comparable; <code>overall_score</code> is always <code>null</code>. Each embodiment's 24-task mean goes in <code>suite_scores.&lt;panda|gr1&gt;</code>; per-task scores take the <code>_panda</code> or <code>_gr1</code> suffix to disambiguate."
---

**Standard**: two independent embodiments, each with its own 24-task protocol:

- **Panda** — 24 atomic tasks on a Mobile Franka robot in RoboCasa kitchen environments ([2406.02523](https://arxiv.org/abs/2406.02523)).
- **GR1** — 24 tabletop pick-and-place tasks (6 "standard" + 18 "post-training novel" SplitA variants) on a Fourier GR1 humanoid, introduced in GR00T N1 ([2503.14734](https://arxiv.org/abs/2503.14734)).

`overall_score` is always `null` — never average across embodiments. Per-embodiment 24-task means live in `suite_scores.panda` and `suite_scores.gr1`.

## Scoring
- `overall_score`: always `null` (aggregation across embodiments is forbidden). If a paper reports a combined number, record it in `notes`; do not place it in `suite_scores`.
- `suite_scores`:
  - `panda` — arithmetic mean of success rates over the 24 Panda atomic tasks; omit when the paper doesn't evaluate Panda or evaluates a subset of the 24.
  - `gr1` — arithmetic mean of success rates over the 24 GR1 tabletop tasks (100 trials per task, max of the last 5 checkpoints); omit when the paper doesn't evaluate GR1 or evaluates a subset.
- `task_scores`: per-task success rates. **All keys MUST end with `_panda` or `_gr1`** to disambiguate embodiment. Panda task names are the RoboCasa atomic task names (e.g. `PnPCounterToCab_panda`, `OpenDrawer_panda`, `CoffeePressButton_panda`). GR1 task names come from the [robocasa-gr1-tabletop-tasks repo](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) (e.g. `PnPCupToDrawerClose_gr1`, `PosttrainPnPNovelFromPlateToBowlSplitA_gr1`). If a paper reports only an overall mean for an embodiment, leave `task_scores` empty for that side; the raw aggregate lands under `suite_scores.<panda|gr1>` instead, or in `task_scores.reported_avg_<panda|gr1>` when the protocol is non-standard.

## Checks
- Is the row's embodiment Mobile Franka (kitchen) or Fourier GR1 (tabletop)? Other embodiments (non-kitchen scenes, other tabletop variants) are NOT this benchmark — set `overall_score = null` and route to the correct benchmark if one exists.
- **Panda rows**: does the entry cover the full 24 atomic tasks? Subsets, supersets (composite + atomic), or relabeled sets → the 24-task mean is not reportable; drop `suite_scores.panda` and keep only per-task numbers.
- **GR1 rows**: does the entry cover all 24 tasks (6 standard + 18 post-training novel) at 100 trials per task with the max of the last 5 checkpoints? Deviations (lower trial count, single-checkpoint rollout, partial coverage) → drop `suite_scores.gr1`.
- Do all `task_scores` keys end in `_panda` or `_gr1`?
- Are `demos_per_task`, `trials_per_task`, and demo source recorded in `notes` when the paper states them?

## Methodology axes (record in `notes`, do not null)
- Training demo budget: Panda protocol sees 50–3000 demos/task in the wild; GR1 typical points are 30/100/300 (paper sweeps) or 1000 (NVIDIA teleop set). Record `demos_per_task` so readers can account for budget. Scores across different budgets are not directly comparable.
- Evaluation trials per task: Panda is variable in the wild; GR1's standard is 100. Record deviations.
- Training data source: human teleoperation / synthetic / machine-generated (MimicGen) / paper-specific mixes. Record when disclosed.
- `weight_type`: `shared` (same checkpoint across benchmarks) vs `finetuned` (trained on this benchmark's data).
