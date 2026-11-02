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

The reproduced Memory figure counts `press_by_number` and `swap_blocks` as 0: the adapter then
re-reset scenes without closing them, which broke both (fixed, see Integration findings; not rerun).
Over the other four tasks the mean is 9.6 / 8.3%. Per-task breakdown below.

A probe also produced a real success on `classify_objects` (Open dimension: score 1.0, terminated
early at 774/1100 steps), with the recorded video showing purposeful bimanual sorting. The
observation mapping, chunked inference, action encoding, and native reward path are all exercised
end to end.

## Published protocol (leaderboard freeze 2026-07-03)

- Upstream `src/eval_client/main.py` runs each task's layout queue in id order: per round it calls
  `env.close()` (sim, stage), then `reset()` relaunches the stage. Unstable layouts are skipped, not
  counted and not retried; the first `eval_num` (50, or 25) stable layouts count.
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
and `swap_blocks` produced no episode with the adapter of that time (every reset after a process's first
raised; cause and fix under Integration findings) and count as 0. The four other tasks average 9.6 / 8.3%.

## Integration findings (validated empirically)

- **Env lifecycle: close per reset, on one thread.** As upstream does per round, every reset after the
  first closes the used scene and `reset()` relaunches it; all Kit calls run on one thread, since `close()`
  deadlocks off the env's creating thread. Soft resets instead leaked render products (`Unable to allocate
  descriptor sets` after ~20), made layout stability history-dependent, and never respawned the objects of
  `press_by_number` / `swap_blocks` (`_get_object_transform` on None). With it (H100): 35 `cover_blocks`
  resets at a flat ~11 s and ~4.8 GB GPU, 16/16 `press_by_number` / `swap_blocks` layouts build, and
  `classify_objects` 0-15 in order fails exactly where fresh processes fail (0, 1, 3, 10).
- **One task per process.** A second `create_eval_env` in one Kit process raised `Simulation context
  already exists` (measured when `close()` still ran off-thread; not retried since).
  `scripts/run_robodojo_protocol.sh` drives one `vla-eval run` per task, as upstream `eval_policy.sh` does.
- **Kit segfaults at interpreter exit** (`omni.graph.core` shutdown), after results are written. The
  benchmark sets `hard_exit`, so `vla-eval run` ends with `os._exit` and keeps its exit code (0, or 3).
- **`utils` namespace collision.** RoboDojo and XPolicyLab both ship a top-level `utils` package,
  and XPolicyLab's `load_file` is a subset (no `load_object_metadata` / `load_desc_info` /
  `load_pkl`). The image bakes `PYTHONPATH=/workspace/RoboDojo`, so a naive skip-if-present
  `sys.path` insert leaves the RoboDojo root *behind* XPolicyLab and layout loading dies with a
  confusing `NameError`. The adapter force-repositions both roots and verifies the resolution.
- **Bad layouts are skipped, not counted.** Some published layouts fail to build or settle. Groups ship
  55–65 layouts for 50 counted episodes; episode `i` runs layout `i` and a failure falls back to the
  spares past `eval_num`, recording each episode's `layout_id`. With stability independent of history,
  the counted set is upstream's; only the choice of spare can differ (shards split them round-robin, a
  shard out of spares errors the episode, and a resumed shard skips spares already counted). Five failed
  resets in a row (unstable layouts aside) raise `EnvUnusableError`: the shard puts its unfinished items
  back on the queue and exits 3, and a rerun with the same eval id resumes.
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
- **Hardware: A100 and H100 both render.** Under Docker, H100 crashed the RTX renderer
  (`ERROR_DEVICE_LOST`) at the first render-product read; under Charliecloud (driver 580.95.05) H100
  steps with RGB observations. "DLSS-RR not supported" is benign.
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
