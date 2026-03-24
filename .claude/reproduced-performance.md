# Reproduced Performance — LIBERO Stage 1

> Evaluation date: 2026-03-24
> Harness version: 0.0.4.dev8+g3e4c63d37
> Hardware: Model servers on DGX-H100 (80GB), benchmarks on A100 nodes
> Protocol: 50 episodes/task, seed=7, num_steps_wait=10

## Summary

| Model | Reported | Reproduced | Verdict |
|---|---|---|---|
| X-VLA (0.9B) | 98.1% | **97.8%** | Reproduced |
| Pi0.5 | 96.9% | ~40% (partial) | Not reproduced — investigation needed |
| StarVLA Qwen2.5-GR00T | 95.4% | 0% | Not reproduced — action space mismatch |
| StarVLA Qwen2.5-OFT | 96.1% | 0% | Not reproduced — same issue |
| StarVLA Qwen2.5-FAST | 95.2% | 0% | Not reproduced — same issue |
| StarVLA Qwen3-OFT | 96.6% | 0% | Not reproduced — same issue |
| GR00T N1.6 (community) | 97.0% | 0% | Not reproduced — integration issue |
| OpenVLA-OFT (joint) | ~96.8% | 0-1.6% | Not reproduced — num_images_in_input misconfigured |
| OpenVLA base (LoRA) | 76.5% | — | Aborted (too slow without batch prediction) |

Only **X-VLA** was successfully reproduced. All other models require further integration
work — each model has unique observation/action requirements that must be correctly
configured in both the model server and benchmark eval config.

---

## Results

### X-VLA (0.9B) — `2toINF/X-VLA-Libero`

Shared weights (single model, all 4 suites).

| Suite | Reproduced | Reported | Delta |
|---|---|---|---|
| LIBERO-Spatial | 98.0% | 98.2% | -0.2 |
| LIBERO-Object | 98.4% | 98.6% | -0.2 |
| LIBERO-Goal | 98.4% | 97.8% | +0.6 |
| LIBERO-10 | 96.2% | 97.6% | -1.4 |
| **Average** | **97.8%** | **98.1%** | **-0.3** |

Verdict: **Reproduced** (within noise margin).

<details>
<summary>Reproduction details</summary>

**Model server** (slurm on DGX-H100-12, 1×H100 80GB):
```bash
sbatch --partition=h100 --gres=gpu:1 --mem=64G --time=6:00:00 \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml -v"
```
Server config (`configs/model_servers/xvla/libero.yaml`):
```yaml
extends: _base.yaml       # script: src/vla_eval/model_servers/xvla.py, denoising_steps: 10
args:
  model_path: "2toINF/X-VLA-Libero"
  benchmark_profile: "libero"
  domain_id: 3
  chunk_size: 30
```

**Benchmark** (50 shards, A100 node):
```bash
for i in $(seq 0 49); do
  uv run vla-eval run -c configs/eval_runs/xvla-libero.yaml \
    --shard-id $i --num-shards 50 --yes &
done
```
Key benchmark params: `absolute_action: true`, `send_wrist_image: true`, `send_state: true`,
`state_format: ee_rot6d`, `flip_wrist_image: false`.

**Server log** (`results/slurm_logs/xvla-libero_103959.err`):
```
10:22:12 INFO  Pre-loading model...
10:22:15 INFO  Loading X-VLA from 2toINF/X-VLA-Libero
10:22:21 INFO  X-VLA model loaded on cuda:0 (float32, profile=libero)
10:22:21 INFO  server listening on 0.0.0.0:8001
```

Result JSONs archived at `.claude/reproductions/xvla-libero/`.
</details>

---

### Pi0.5 — `pi05_libero` (via openpi)

Partial results (spatial + object complete, goal + 10 in progress).

| Suite | Reproduced | Reported | Delta |
|---|---|---|---|
| LIBERO-Spatial | 41.4% | 98.8% | -57.4 |
| LIBERO-Object | 39.8% | 98.2% | -58.4 |
| LIBERO-Goal | in progress | 98.0% | — |
| LIBERO-10 | in progress | 92.4% | — |

Verdict: **Not reproduced**. Consistently ~40% across suites vs reported ~97%.
Server loaded `pi05_libero` checkpoint correctly, `send_wrist_image: true` + `send_state: true`
configured. No server errors. Root cause unclear — may be action space normalization, image
preprocessing, or proprioceptive state format mismatch with openpi's expectations.

---

### StarVLA (4 variants) — 0% across all

Tested: Qwen2.5-VL-{FAST, OFT, GR00T} and Qwen3-VL-OFT, all on LIBERO-4in1 checkpoints.

All episodes ran to max_steps (220) with 0% success. Tried with and without `send_state: true`.
batch prediction was implemented and working (fast inference), but actions produced are
ineffective.

Likely cause: StarVLA's LIBERO checkpoints may require specific observation preprocessing
(image size, normalization) or action post-processing not yet implemented in the harness
integration.

**StarVLA Qwen3-PI** (`StarVLA/Qwen3-VL-PI-LIBERO-4in1`): Server crashed on startup —
DiT state_dict has 36 transformer blocks but model config builds 16. Requires compat fix.

---

### GR00T N1.6 (community) — `0xAnkitSingh/GR00T-N1.6-LIBERO`

0% success. Server started correctly with `video_keys=['image', 'wrist_image']`.
`send_wrist_image: true` configured. `predict_batch()` implemented with multi-video-key
support. Actions produced but ineffective — likely observation format or action space mismatch
with this community-trained checkpoint.

---

### OpenVLA-OFT (7B, joint) — `moojink/...-libero-spatial-object-goal-10`

LIBERO-Spatial: 1.6%, LIBERO-Object: 0.0%. Aborted.

Root cause identified: `num_images_in_input: 1` in server config, but OFT's 97.1% result
uses 3rd-person + wrist camera (2 images). Additionally, `unnorm_key` for the joint
checkpoint needs verification.

---

### OpenVLA base (7B, per-suite LoRA) — `openvla/openvla-7b-finetuned-libero-{suite}`

Aborted due to extreme slowness without `predict_batch()`. chunk_size=1 means one action
per inference call, ~2.5 min/episode. Estimated 8+ hours for all 4 suites with 30 shards.

---

## Troubleshooting Log

| Issue | Model | Resolution |
|---|---|---|
| 0% success rate (initial) | All | Each model needs specific benchmark params (`absolute_action`, `send_wrist_image`, `send_state`, `state_format`) |
| `unnorm_key` not found | OpenVLA base | Finetuned checkpoints use `libero_{suite}` not `libero_{suite}_no_noops` |
| Port conflicts on shared nodes | All | Assign unique ports per server on same node |
| `wrist_image not in obs` | GR00T | Added `send_wrist_image: true` to benchmark config |
| DiT state_dict layer mismatch | StarVLA Qwen3-PI | Checkpoint has 36 DiT blocks, model expects 16. Server crash. **Skipped.** |
| Pi0-FAST vs Pi0.5 | Pi0 | `pi0_fast_libero` has no reported numbers; switched to `pi05_libero` |
| TF CUDA JIT on H100 | OFT-joint | 30+ min startup due to TF recompiling CUDA kernels for compute 9.0 |
| `vla-eval` not in PATH after node change | All | Use `uv run vla-eval` instead of bare `vla-eval` |
| `num_images_in_input: 1` | OFT-joint | OFT 97.1% needs 2 images (3rd-person + wrist). Config had 1. |
| `max_batch_size` kwarg rejected | StarVLA | `__init__` didn't pass `**kwargs` to `super()`. Fixed. |
| Sharding covers only 1 task | OpenVLA-10 | 10 shards with 500 episodes → all go to task 0. Need ≥50 shards for 10 tasks. |

## Observations

### Only X-VLA reproduced — why?

X-VLA is the only model with a dedicated `benchmark_profile` system that maps the harness
observation format to exactly what the model expects (`absolute_action`, `ee_rot6d` state,
`flip_wrist_image: false`). All other models rely on generic observation passthrough, which
creates subtle mismatches in:
- Action space (delta vs absolute)
- Image preprocessing (flip, resize, normalization)
- Proprioceptive state format (8D vs 7D vs 20D)
- Gripper convention (binary vs continuous, sign convention)

**Recommendation**: Each model server needs a documented "benchmark profile" specifying
exactly which benchmark params are required. This should be part of the model server config,
not left to the user to figure out per model.

### Batch prediction impact on throughput

| Model | predict_batch | 2000 ep (30 shards) | Per-episode |
|---|---|---|---|
| DB-CogACT | **Yes** | ~20 min | ~0.6s |
| X-VLA | No | ~16 min (50 shards) | ~15s |
| OFT-joint | No | ~35 min | ~30s |
| StarVLA | **Yes** (new) | — (0% SR) | ~1 min/shard |
| GR00T | **Yes** (new) | — (0% SR) | — |
| OpenVLA base | No | ~3 hours | ~2.5 min |
| Pi0.5 | No (no batch API) | ~3.5 hours | ~3 min |

`predict_batch()` implemented for StarVLA and GR00T in this session. Pi0 skipped
(openpi `infer()` has no batch API). OpenVLA/OFT not yet implemented.

### Next steps

1. **Fix model-specific benchmark params**: Document required params per model server.
   Consider adding a `benchmark_params` field to model server configs.
2. **Debug Pi0.5 40%**: Investigate action normalization and observation preprocessing.
3. **Debug StarVLA 0%**: Compare observation format with StarVLA's own eval script.
4. **Fix OFT-joint**: Set `num_images_in_input: 2` and verify `unnorm_key`.
5. **Implement predict_batch for OpenVLA/OFT**: Critical for reasonable eval times.
