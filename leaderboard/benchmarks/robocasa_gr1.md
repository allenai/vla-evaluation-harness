---
benchmark: robocasa_gr1
display_name: RoboCasa-GR1
paper_url: https://arxiv.org/abs/2503.14734
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
tasks:
- PnPCupToDrawerClose
- PnPPotatoToMicrowaveClose
- PnPMilkToMicrowaveClose
- PnPBottleToCabinetClose
- PnPWineToCabinetClose
- PnPCanToDrawerClose
- PosttrainPnPNovelFromCuttingboardToBasketSplitA
- PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
- PosttrainPnPNovelFromCuttingboardToPanSplitA
- PosttrainPnPNovelFromCuttingboardToPotSplitA
- PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
- PosttrainPnPNovelFromPlacematToBasketSplitA
- PosttrainPnPNovelFromPlacematToBowlSplitA
- PosttrainPnPNovelFromPlacematToPlateSplitA
- PosttrainPnPNovelFromPlacematToTieredshelfSplitA
- PosttrainPnPNovelFromPlateToBowlSplitA
- PosttrainPnPNovelFromPlateToCardboardboxSplitA
- PosttrainPnPNovelFromPlateToPanSplitA
- PosttrainPnPNovelFromPlateToPlateSplitA
- PosttrainPnPNovelFromTrayToCardboardboxSplitA
- PosttrainPnPNovelFromTrayToPlateSplitA
- PosttrainPnPNovelFromTrayToPotSplitA
- PosttrainPnPNovelFromTrayToTieredbasketSplitA
- PosttrainPnPNovelFromTrayToTieredshelfSplitA
detail_notes: "Standard: 24 pick-and-place tasks (6 standard + 18 post-training novel SplitA variants) on a <strong>Fourier GR1 humanoid</strong> tabletop setup, introduced in the <a href='https://arxiv.org/abs/2503.14734'>GR00T N1 paper</a>. Protocol is 100 trials per task, reporting the max over the last 5 checkpoints. Distinct benchmark from the original RoboCasa (Mobile Franka in kitchen) — do not conflate. Training demo budgets vary (30 / 100 / 300 demos per task in the paper's sweeps; 1000 per task in NVIDIA's released teleop dataset)."
---

**Standard**: 24 tabletop pick-and-place tasks (6 "standard" + 18 "post-training novel" SplitA variants) on a Fourier GR1 humanoid, introduced in GR00T N1 ([2503.14734](https://arxiv.org/abs/2503.14734)); `overall_score` = mean success rate across the 24 tasks, with 100 trials per task and the max of the last 5 checkpoints taken.

## Scoring
- `overall_score`: arithmetic mean of success rates over all 24 tasks; `null` if the evaluated set is not the full 24.
- `suite_scores`: not used — the benchmark has no canonical sub-suite grouping.
- `task_scores`: per-task success rates keyed by the environment name from the [robocasa-gr1-tabletop-tasks repo](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) (e.g. `PnPCupToDrawerClose`, `PosttrainPnPNovelFromPlateToBowlSplitA`). Many papers report only the overall mean — in that case leave `task_scores` empty and the paper's reported aggregate ends up in `task_scores.reported_avg` automatically when the protocol is non-standard.

## Checks
- Is the embodiment a Fourier GR1 humanoid on a tabletop? Alternative embodiments (e.g. Mobile Franka in RoboCasa kitchen) belong to the original `robocasa` benchmark, not this one — set `matches_standard = "no"` and route accordingly.
- Does the entry evaluate all 24 tasks (6 standard + 18 post-training novel)? Subsets (e.g. only the 6 standard tasks, or only the post-training novel set) → `overall_score = null`.
- Is the evaluation 100 trials per task with the max of the last 5 checkpoints? Deviations (lower trial count, single-checkpoint rollout) go in `notes`.

## Methodology axes (record in `notes`, do not null)
- Training demo budget: 30 / 100 / 300 demos per task are the GR00T N1 paper's sweep points; 1000 per task is the NVIDIA teleop dataset. All budgets within the 24-task protocol are valid; record `demos_per_task` so readers can account for it. Scores across different budgets are not directly comparable.
- Trial count per task: 100 is the paper's standard. Record deviations.
- Training data source: human teleop (1000 demos/task NVIDIA set), machine-generated (MimicGen), or paper-specific mixes. Record when disclosed.
- `weight_type`: `shared` (same checkpoint across benchmarks) vs `finetuned` (trained on this benchmark's data).
