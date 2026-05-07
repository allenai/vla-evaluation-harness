# MolmoBot — Reproduction Report (MolmoSpaces-Bench)

AI2's MolmoBot VLA evaluated on MolmoSpaces-Bench. [MolmoBot](https://github.com/allenai/MolmoBot) |
[MolmoSpaces](https://github.com/allenai/molmospaces) |
[MolmoBot Paper](https://arxiv.org/abs/2603.16861) |
[MolmoSpaces Paper](https://arxiv.org/abs/2602.11337) |
[Weights](https://huggingface.co/collections/allenai/molmobot-models) |
Molmo2-4B + DiT flow-matching action head.

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| MolmoSpaces-Bench Pick&Place (procthor-objaverse, 200ep) | **57.0%** | 57.7% | ✅ |
| MolmoSpaces-Bench Pick | — | 64.0% | ⬜ |
| MolmoSpaces-Bench Open | — | — | ⬜ |
| MolmoSpaces-Bench Close | — | — | ⬜ |
| MolmoSpaces-Bench Door Opening (RB-Y1) | — | 77.7% | ⬜ |
| MolmoSpaces-Bench Navigation (RB-Y1) | — | — | ⬜ |

Paper numbers from [MolmoBot Table 6](https://arxiv.org/abs/2603.16861), MolmoBot (F=2) "final" column on procthor-10k (MSProc) unless noted.

### Pick-and-Place (procthor-objaverse, Franka FR3)

| | |
|---|---|
| **Model** | [`allenai/MolmoBot-DROID`](https://huggingface.co/allenai/MolmoBot-DROID) (Molmo2-4B + DiT, n_obs_steps=2) |
| **Server config** | [`configs/model_servers/molmobot/droid.yaml`](../../configs/model_servers/molmobot/droid.yaml) |
| **Benchmark config** | [`configs/benchmarks/molmospaces/pick_and_place.yaml`](../../configs/benchmarks/molmospaces/pick_and_place.yaml) |
| **Benchmark source** | `procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark` |
| **Eval config (molmo_spaces)** | `olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig` |
| **Results** | [`data/molmobot-molmospaces/pick_and_place_procthor_objaverse.json`](data/molmobot-molmospaces/pick_and_place_procthor_objaverse.json) |

200 episodes, 16-way parallel sharding (2 model servers × 8 shards each), 2× A100 80GB, ~3 h wall time.

Reproduced **57.0%** (114/200) vs reported **57.7%**. Difference of 0.7 pp is within run-to-run variance of flow-matching inference (no deterministic seed control in `SynthManipMolmoInferenceWrapper.get_action_chunk`).

## Configuration Notes

### Architectural mismatch: `ModelServer` vs `PredictModelServer`

The base `PredictModelServer` assumes stateless per-call inference, but MolmoBot's reference policy (`SynthVLAPolicy` in `olmo.eval.configure_molmo_spaces`) maintains **per-episode state** across calls:

- **Frame history**: `n_obs_steps=2` with `obs_step_delta=8` — the policy stores observations from each step and retrieves the frame from 8 steps ago at each refill.
- **Action buffer**: Predicts a 16-step action chunk, executes `execute_horizon=8` actions before re-querying.
- **Per-step safety clamp**: Per-action clipping of joint deltas relative to the current `qpos` (absolute-joint mode) uses the latest observation, not the observation at chunk prediction time.

To preserve this behavior the harness subclasses `ModelServer` directly and keys per-episode state by `ctx.session_id`, reset in `on_episode_start`.

**Future refactor: move the per-step clamp into the benchmark.**
Of the three pieces of state above, only the per-step safety clamp actually requires knowing the env's latest `qpos` — and the benchmark's `step()` method already has that information. If the clamp is moved into `MolmoSpacesBenchmark.step()` (it reads `self._task`'s current `qpos`, subtracts the incoming absolute target, scales the delta, adds it back), the model server becomes a stateless chunk producer that can inherit from `PredictModelServer`. That enables three wins currently left on the table:

1. **Cross-session batching** via `max_batch_size > 1` — the 16 sharded clients that today serialize through a single synchronous `get_action_chunk` call could be batched into one GPU forward pass. The reproduction run observed GPU utilization of 15–25% precisely because the current `ModelServer` path has no batching hook; a batched `PredictModelServer` path should saturate the GPU and cut wall time materially.
2. **Event-loop offloading** — `PredictModelServer` runs `predict()` in a dedicated thread pool, so one shard's blocking inference no longer freezes the other shards' `on_observation` dispatches on the same event loop.
3. **Simpler reuse** — any other VLA ported to this benchmark can use the stock `PredictModelServer` path instead of reimplementing session state management.

The frame-history logic (`n_obs_steps=2`, `obs_step_delta=8`) can still live in the model server via a small `prev_frames: dict[session_id, ...]` keyed on `ctx.session_id`, because `chunk_size == obs_step_delta == 8` means "the previous chunk refill" is exactly "8 steps ago". No per-step state needed on the server side.

This refactor is intentionally out of scope for the initial landing: it would require a fresh 200-episode reproduction run to confirm the numerical score is unchanged, which costs ~3 hours of GPU time. The current path mirrors `SynthVLAPolicy` bit-for-bit, which is the safer starting point for "does this integration reproduce the paper at all?".

### Critical config pins (from MolmoBot paper reproduction)

The MolmoBot paper ([README](https://github.com/allenai/MolmoBot/blob/main/MolmoBot/README.md#running-the-sim-eval-for-franka)) runs eval with:

```
--eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig
--task_horizon 600
```

This config pins four values that are load-bearing:

| Parameter | Value | Why it matters |
|-----------|-------|---------------|
| `action_type` | `joint_pos` (absolute) | Model outputs absolute joint positions. Using `joint_pos_rel` (the other default) interprets the same outputs as deltas — at worst silently zeros the policy. |
| `policy_dt_ms` | `66.0` (≈15 Hz) | The other Franka eval config ships with `200.0` (5 Hz). Mismatched control rate pushes actions far outside the training distribution. |
| `clamp_gripper` | `True` | The model emits raw gripper values in roughly [0, 255]; the env expects discrete open/close. Without clamping the gripper spasms. |
| `task_horizon` | `600` | The other configs default to 500. For PickAndPlace this early-terminates a non-trivial fraction of episodes. |

An early reproduction attempt using `SynthVLAFrankaBenchmarkOriginalEvalConfig` + `task_horizon=500` scored **0/11** before these four discrepancies were identified. Switching to the paper's config immediately produced **60% on 10 episodes** and **57.0% on 200 episodes**.

### MolmoSpaces environment integration

The MolmoSpaces simulator does not expose a Gymnasium-style `env.reset()`/`env.step()` API — episodes are normally driven by `JsonEvalRunner` → `JsonEvalTaskSampler` → `BaseMujocoTask`. The harness benchmark class builds a `JsonEvalTaskSampler` per episode and calls the underlying `BaseMujocoTask.reset()`/`step()` directly, reusing MolmoSpaces's own scene loading and success judging:

```python
sampler = JsonEvalTaskSampler(exp_config, episode_spec)
task = sampler.sample_task(house_index=episode_spec.house_index)
obs, info = task.reset()
# ... run loop ...
success = task.judge_success()
```

The `exp_config` instance is patched with `EvalRuntimeParams()` (normally done by `JsonEvalRunner.patch_config`) so that `_sample_task()` works without the full runner.

### Camera name mapping

MolmoBot's reference policy uses the logical camera names `exo_camera_1` and `wrist_camera`, but the MolmoSpaces sensor suite emits them under the `FrankaDroidCameraSystem` names `droid_shoulder_light_randomization` and `wrist_camera_zed_mini`. The harness benchmark's `make_obs()` remaps both in both directions so model servers can consume the legacy names without knowing about the sensor-system prefix.

### Action wire format

The model server emits a flat 8-D action vector per step:

```
[j1, j2, j3, j4, j5, j6, j7, gripper]
```

- Elements 0–6: absolute arm joint positions (post per-step safety clamp).
- Element 7: gripper command, already clamped to `{0, 255}`.

The benchmark splits this back into MolmoSpaces's per-move-group dict (`{"arm": ..., "gripper": ...}`) in `step()`.

### Asset baking (Docker image layout)

The `allenai/molmospaces` HuggingFace dataset repo is ~13 TB when enumerated end to end (MuJoCo + USD, every scene/object/grasp source in `DATA_TYPE_TO_SOURCE_TO_VERSION`). The *actually required* subset for a Franka pick-and-place eval is ~15 GB: the `franka_droid` robot, the `thor` object catalog, the procthor-objaverse scenes that benchmark episodes reference, the matching Objaverse object bundle, DROID grasps, and the `molmospaces-bench-v2` JSON specs.

`molmo_spaces` extracts these lazily on first import, so `docker/Dockerfile.molmospaces` triggers the extraction at build time with

```dockerfile
ENV MLSPACES_ASSETS_DIR=/assets \
    MLSPACES_CACHE_DIR=/cache/molmo-spaces-resources
RUN python -c "import molmo_spaces; from molmo_spaces.evaluation.benchmark_schema import load_all_episodes"
```

The resulting layers are ~15 GB of cached tarballs under `/cache/molmo-spaces-resources` plus ~228 MB of symlinks under `/assets`. Final image is **31.4 GB**, comparable to the other heavy benchmark images (`robocasa` 35.6 GB, `robotwin` 28.6 GB) and well within the "Zero Setup" Docker pattern used by the rest of vla-eval — users `docker pull` and `docker run`, no manual asset download or volume mount required. The benchmark YAML points at the in-image path (`/assets/benchmarks/molmospaces-bench-v2/...`).

## Known Limitations

- Only the pick-and-place task on `procthor-objaverse` has been end-to-end reproduced. The benchmark class is task-agnostic (code path shared across all 8 MolmoSpaces-Bench task types), but other task/scene combinations have not been numerically validated.
- Some older benchmark JSONs in MolmoSpaces-Bench v2 (e.g. `procthor-10k/FrankaPickHardBench_20260212_200ep`) still reference the legacy `mujoco_thor.tasks.pick_task.PickTask` class path and fail to load under the current `molmo_spaces` package. Newer `procthor-objaverse` benchmarks use the correct `molmo_spaces.tasks.*` paths.
- Flow-matching inference in `SynthManipMolmoInferenceWrapper.get_action_chunk` does not accept an external `torch.Generator` by default, so runs are not bit-exact reproducible across hosts. Per-episode results can flip between runs; aggregate scores over ≥50 episodes are stable.
- `max_episodes` in `molmo_spaces.evaluation.run_evaluation` only slices the local `episodes` list for the info log; `JsonEvalRunner` reloads the full benchmark directory and ignores the subset. To run a smaller eval, write a filtered `benchmark.json` to a new directory and point at that.
- `molmo-spaces` is git-only (`molmo-spaces @ git+https://github.com/allenai/molmospaces.git`) rather than a PyPI release, so builds are sensitive to upstream main landing schema-changing commits. Pin a specific commit in the Dockerfile if long-term reproducibility matters.

## Sharding & Throughput

A single-shard 10-episode run took **~76 min** (vla-eval) / **~55 min** (native MolmoSpaces pipeline). Full 200-episode runs benefit from sharding:

- Launch two `MolmoBotModelServer` instances, one pinned to each GPU via `CUDA_VISIBLE_DEVICES`, on separate ports.
- Launch N `vla-eval run --shard-id {i} --num-shards N --server-url ws://localhost:{port}` processes; alternate shards between the two servers (even IDs → GPU 0, odd IDs → GPU 1).
- Merge results with `vla-eval merge configs/... -o merged.json`.

With N=16 shards on 2× A100 80GB the 200-episode pick-and-place run completes in approximately 3 hours. GPU utilization stays around 15–25 % because per-call inference is the bottleneck and there is no request batching; throughput scales roughly linearly with the number of GPUs rather than the number of shards above ~8.
