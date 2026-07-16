# RoboDojo — Reproduction Status

[Repo](https://github.com/RoboDojo-Benchmark/RoboDojo) |
[Paper](https://arxiv.org/abs/2607.04434) |
[Leaderboard](https://robodojo-benchmark.com/leaderboard) |
42 bimanual simulation tasks (dual ARX-X5, Isaac Lab) across five capability dimensions

## Status

**Integration:** ✅ Benchmark + configs + Docker recipe + smoke pass (echo server, result JSON verified).
**Trained-VLA reproduction:** 🚧 π₀.₅ (officially released leaderboard checkpoint, seed 0), Memory
dimension only, partial (see below). The other four dimensions use the same machinery but were not run.

### Results Summary

Each cell is `score / success-rate`, both ×100 (paper convention). Reproduced column is π₀.₅ seed 0;
Reported column is the paper's 3-seed mean.

| Dimension | Reproduced | Reported | Status |
|---|:--:|:--:|---|
| Memory | 6.4 / 5.6% | 5.78 / 4.56% | Partial: 4/6 tasks, 15/50 ep |
| Generalization | — | 13.37 / 8.17% | Not run |
| Precision | — | 12.40 / 5.50% | Not run |
| Long-Horizon | — | 23.54 / 14.67% | Not run |
| Open | — | 1.98 / 1.67% | Not run |
| **Average** | — | **11.41 / 6.91%** | — |

The reproduced Memory figure counts the two layout-blocked tasks (`press_by_number`,
`swap_blocks`; upstream bug, see Known gaps) as 0; over the four tasks that build the mean is
9.6 / 8.3%. Per-task breakdown below.

A probe also produced a real success on `classify_objects` (Open dimension: score 1.0, terminated
early at 774/1100 steps), with the recorded video showing purposeful bimanual sorting. The
observation mapping, chunked inference, action encoding, and native reward path are all exercised
end to end.

## Published protocol (leaderboard freeze 2026-07-03)

- 42 tasks × 50 episodes (Generalization: 25 standard + 25 `_random`), 3 training seeds per
  policy. Report mean ± std of score / success rate per dimension; overall = mean of the five
  dimension means (not the per-task mean).
- Metrics: `success` (binary, native reward manager) and `score` (1.0 on success, else the task's
  partial progress). Both are reported ×100.
- π₀.₅ reference numbers are in the Results Summary above (Appendix K: fine-tuned from `pi05_base`,
  batch 256, 60K steps, 3-seed mean).

## Reproduction setup

- Checkpoint: `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999` from the
  `RoboDojo-Benchmark/RoboDojo` HF dataset (all evaluated policies release their fine-tuned
  checkpoints there; π₀.₅ ships all three seeds).
- Server: `configs/model_servers/robodojo_pi05/pi05.yaml` — OpenPI direct inference through
  XPolicyLab's openpi fork (which carries the RoboDojo train configs), reproducing the upstream
  input contract (3 CHW cameras, 14-D packed qpos, instruction prompt, open-loop action chunk).
- Deviations from the published protocol: one training seed (0) instead of three (so no std),
  and the partial run below used 15 of the protocol's 50 episodes per task. Everything else
  (layout groups, step limits, metrics) follows the official protocol; `eval.yaml` encodes the
  full 50-episode counts.

### Memory dimension — reproduced (per task)

π₀.₅, seed 0. Scores ×100.

| Task | Episodes | Score | SR |
|---|--:|--:|--:|
| `cover_blocks` | 15 | 31.7 | 26.7% |
| `match_and_pick_from_conveyor` | 15 | 6.7 | 6.7% |
| `swap_T` | 15 | 0.0 | 0.0% |
| `imitate_sorting_sequence` | 15 | 0.0 | 0.0% |
| `press_by_number` | — | blocked | — |
| `swap_blocks` | — | blocked | — |
| **Mean (6 tasks, blocked = 0)** | | **6.4** | **5.6%** |

Episode counts are 15/50: a pipeline validation, not the full statistical protocol. `press_by_number`
and `swap_blocks` fail deterministically at layout build (every layout raises in
`check_layout_stability`; see Known gaps), yielding zero valid episodes; they are counted as 0 in the
mean. The four tasks that build average 9.6 / 8.3%.

## Integration findings (validated empirically)

- **One task per process.** In principle Isaac Lab supports teardown-and-recreate
  (`DirectRLEnv.close()` → `SimulationContext.clear_instance()`, then build a new env). In
  practice `close()` hangs: measured on a *clean* `stack_blocks` scene (dual-X5, `camera_config`,
  Isaac Sim 5.1), `close()` did not return within 180 s, so `clear_instance()` is never reached
  and the next `create_eval_env` raises `RuntimeError: Simulation context already exists`. So a
  task switch inside a run is not viable (upstream `eval_policy.sh` is one-task-per-process for the
  same reason). `scripts/run_robodojo_protocol.sh` drives one `vla-eval run` per task.
- **`utils` namespace collision.** RoboDojo and XPolicyLab both ship a top-level `utils` package,
  and XPolicyLab's `load_file` is a subset (no `load_object_metadata` / `load_desc_info` /
  `load_pkl`). The image bakes `PYTHONPATH=/workspace/RoboDojo`, so a naive skip-if-present
  `sys.path` insert leaves the RoboDojo root *behind* XPolicyLab and layout loading dies with a
  confusing `NameError`. The adapter force-repositions both roots and verifies the resolution.
- **Bad layouts are skipped, not counted.** Some published layouts fail to build or settle. The
  adapter skips them, but when *every* layout in a group fails the group is exhausted with zero
  episodes (`press_by_number`, `swap_blocks`; see Known gaps). Groups ship 55–65 layouts for 50 counted episodes;
  the adapter consumes them in order and skips failures, recording each episode's `layout_id`.
  Never `close()` the env to recover: teardown destroys the cameras and the next `reset()` dies
  in `init_cameras`.
- **cuRobo planners are unusable from the public assets** (only `curobo_tmp.yml` templates ship),
  and joint-space VLA evaluation does not need them — including for the Franka competition tasks,
  whose support arm replays recorded trajectories.
- **Throughput is the binding constraint.** Single-env stepping runs at ~30–50 steps/min
  (sim + 3-camera render + chunked inference), and failed episodes run to the task's `step_lim`
  (300–1900), so one task ≈ 12–20 GPU-hours and the Memory dimension ≈ 100 GPU-hours. Isaac
  renderers do not share a GPU gracefully: 4 lanes on one A100 collapsed per-lane throughput
  ~30× (aggregate ~8× *worse* than a single lane). Run one lane per GPU. The paper's own answer
  is batched parallel simulation (`num_envs > 1`, tiled rendering), which the harness cannot
  drive today — see below.
- **Hardware: A100 works, H100 does not.** On H100 (Hopper) the RTX renderer crashes the GPU
  (`ERROR_DEVICE_LOST`, GPU crash dump) as soon as a render product is read, so every episode
  errors out at the first observation. A100 renders fine (with a benign "DLSS-RR not supported"
  warning). Plan capacity on RTX-capable or A100-class GPUs.
- **Vulkan ICD pinning.** The upstream image hard-pins `VK_ICD_FILENAMES` *and* `VK_DRIVER_FILES`
  to `/etc/vulkan/icd.d/nvidia_icd.json`, which only exists if the host's container toolkit
  injects it. Where it doesn't, Vulkan dies with `ERROR_INCOMPATIBLE_DRIVER`. The configs
  override both to the image's own `/usr/share/vulkan/icd.d/nvidia_icd.json`.
- **Docker ignores Slurm's GPU cgroup.** `--gpus device=<index>` resolves against the *host* GPU
  list, so a container can grab a GPU allocated to another job. Under Slurm, always map by the
  UUIDs from `nvidia-smi` inside the allocation (`--gpus device=<UUID>`).

## Known gaps / future work

- **Batched evaluation.** RoboDojo's `EvalEnv` supports `num_envs > 1` with tiled rendering and
  XPolicyLab exposes `update_obs_batch` / `get_action_batch`. Exploiting it needs a vectorized
  benchmark interface in the harness (one benchmark instance driving N concurrent episodes).
  That is the single biggest available speedup for this benchmark.
- **Competition-task metric deviation.** The 3 `dual_x5_and_franka_competition` tasks
  (`imitate_sorting_sequence`, `make_kong`, `play_tic_tac_toe`) can mark an episode "unstable"
  when the scripted Franka move fails; upstream *excludes* those from the denominator, but
  vla-eval's collector counts every episode, so they score as failures here (slightly pessimistic).
- **`press_by_number` and `swap_blocks` fail deterministically at layout build.** Every layout in
  both groups (all 65 / all 55) raises in RoboDojo's `check_layout_stability` (`layout_manager.py:430`:
  `get_instance_pose` → `obj._get_object_transform()` on a `None` instance), so the adapter exhausts
  the group with zero valid episodes. This is upstream/asset-side (a layout references a scene
  instance that is never created), not a flaky crash, so re-running does not help; it needs an
  upstream fix. The other four Memory tasks build fine.
- **Isaac Sim process crashes are unrecoverable mid-task.** A run can also hit a breakpad crash that
  takes down the process; the aggregate is then incomplete. `run_robodojo_protocol.sh` runs one
  process per task, so any crash is isolated to that task.
- Only the Memory dimension is reproduced so far; the other four dimensions use the same
  machinery and configs (`configs/benchmarks/robodojo/eval.yaml` covers all 42 tasks).

## License rationale (NO_REDIST)

The image builds on RoboDojo's official Isaac Sim 5.1 image and therefore bundles NVIDIA
Omniverse/Isaac Sim binaries (NVIDIA EULA) — same rationale as `behavior1k`. RoboDojo itself is
non-commercial research licensed, and the eval assets are distributed separately (HF dataset
`RoboDojo-Benchmark/RoboDojo`) and mounted at runtime. Build locally, do not push.
