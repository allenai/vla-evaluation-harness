---
benchmark: robomme
display_name: RoboMME
paper_url: https://arxiv.org/abs/2603.04639
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- counting
- permanence
- reference
- imitation
tasks:
- BinFill
- PickXtimes
- SwingXtimes
- StopCube
- VideoUnmask
- VideoUnmaskSwap
- ButtonUnmask
- ButtonUnmaskSwap
- PickHighlight
- VideoRepick
- VideoPlaceButton
- VideoPlaceOrder
- MoveCube
- InsertPeg
- PatternLock
- RouteStick
expand_suites: true
avg_position: 4
avg_label: Avg
detail_notes: "RoboMME (<a href='https://arxiv.org/abs/2603.04639'>2603.04639</a>) — memory-augmented manipulation benchmark on a ManiSkill / 7-DOF Franka Panda tabletop. 16 tasks across 4 suites that target distinct memory demands: <strong>Counting</strong> (temporal), <strong>Permanence</strong> (spatial), <strong>Reference</strong> (object), <strong>Imitation</strong> (procedural). Standard protocol: 50 episodes/task (800 total), max 1,300 steps, multi-task single model, results averaged over the last 3 checkpoints × 3 seeds (9 runs). <code>overall_score</code> = unweighted mean of the 4 suite scores."
aggregation:
  container: suite_scores
  keys:
  - counting
  - permanence
  - reference
  - imitation
---

**Standard**: RoboMME memory benchmark ([2603.04639](https://arxiv.org/abs/2603.04639)) — 16 manipulation tasks grouped into 4 suites (`counting`, `permanence`, `reference`, `imitation`) on a ManiSkill 7-DOF Franka Panda tabletop. Multi-task single model, 50 episodes per task (800 total), max 1,300 steps per episode, averaged over the last 3 checkpoints × 3 random seeds (9 runs total). `overall_score` = arithmetic mean of the 4 suite scores.

## Scoring
- `overall_score`: arithmetic mean of `suite_scores.counting`, `permanence`, `reference`, `imitation`; `null` if any of the four is missing.
- `suite_scores`: canonical keys `counting`, `permanence`, `reference`, `imitation`. Each is the unweighted mean of the 4 tasks in that suite.
- `task_scores`: canonical keys are the 16 task names declared in the registry (`BinFill`, `PickXtimes`, `SwingXtimes`, `StopCube`, `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap`, `PickHighlight`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`, `MoveCube`, `InsertPeg`, `PatternLock`, `RouteStick`). Values are success rates in percent.

## Checks
- Are all 4 suites (`counting`, `permanence`, `reference`, `imitation`) present? If not → `overall_score = null`.
- Is the metric task success rate (not subgoal completion or any partial-credit variant)?
- Is this RoboMME and **not** a confusable memory benchmark — `libero_mem` (LIBERO-Mem, 2511.11478), `mikasa` (MIKASA-Robo), MemoryBench, MemoryVLA's MemoryRTBench (2603.18494), or RMBench? Each has a separate registry entry or is excluded.
- Is the multi-task setup respected? Per-suite or single-suite training (e.g. counting-only) is non-standard → `overall_score = null`, raw aggregate goes in `task_scores.reported_avg`.
- Is `weight_type` set correctly? Papers training on RoboMME demos are `finetuned`; download-and-run of an off-the-shelf checkpoint is `shared`.

## Methodology axes (record in `notes`, do not null)
- Run averaging: standard reports 9-run mean (3 last checkpoints × 3 seeds). Single-seed / single-checkpoint reports are valid entries — record the deviation in `notes`.
- Backbone: paper baselines are π₀.5-based. Memory variants attach to that backbone (FrameSamp, TokenDrop, SimpleSG, GroundSG, TTT, RMT). Record the backbone family in `notes` when it deviates.
- Memory class: when reporting a memory-augmented variant, note the class — `perceptual` (FrameSamp / TokenDrop), `symbolic` (SimpleSG / GroundSG + a VLM), or `recurrent` (TTT / RMT) — to keep ablation rows distinguishable in the table.
