# Reproduced Performance — LIBERO Stage 1

> Evaluation date: 2026-03-24
> Harness version: 0.0.4.dev8+g3e4c63d37
> Hardware: Model servers on DGX-H100 (80GB), benchmarks on A100 nodes
> Protocol: 50 episodes/task, seed=7, num_steps_wait=10

## Summary

| Model | Reported | Reproduced | Verdict |
|---|---|---|---|
| X-VLA (0.9B) | 98.1% | **97.8%** | Reproduced |
| Pi0.5 | 96.9% | **96.4%** | Reproduced |
| OpenVLA-OFT (joint) | ~96.8% | **97.0%** | Reproduced |
| StarVLA (4 variants) | 95-97% | 0% | Not reproduced — absolute action + unnormalization issue |
| GR00T N1.6 (community) | 97.0% | 0% | Not reproduced — needs further debugging |
| OpenVLA base (LoRA) | 76.5% | — | Aborted (too slow without batch prediction) |

**3 models reproduced** (X-VLA, Pi0.5, OFT-spatial). Key fixes applied:
image_resolution=224 (Pi0.5), num_images_in_input=2 (OFT), gripper transform removal (StarVLA),
send_state/send_wrist_image (all).

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

| Suite | Reproduced | Reported | Delta |
|---|---|---|---|
| LIBERO-Spatial | 92.6% | 98.8% | -6.2 |
| LIBERO-Object | **98.8%** | 98.2% | +0.6 |
| LIBERO-Goal | **98.2%** | 98.0% | +0.2 |
| LIBERO-10 | **96.2%** | 92.4% | +3.8 |
| **Average** | **96.4%** | **96.85%** | **-0.4** |

Verdict: **Reproduced** (avg within 0.5%p). Spatial 6%p low — likely image flip difference.

Key fix: `image_resolution: 224` (Pi0.5 trained on 224×224, harness defaulted to 256×256).
Benchmark params: `send_wrist_image: true`, `send_state: true`.

Result JSONs archived at `.claude/reproductions/pi0-libero/`.

---

### OpenVLA-OFT (7B, joint) — `moojink/...-libero-spatial-object-goal-10`

| Suite | Reproduced | Reported | Delta |
|---|---|---|---|
| LIBERO-Spatial | **97.2%** | 97.6% | -0.4 |
| LIBERO-Object | **97.0%** | 98.4% | -1.4 |
| LIBERO-Goal | **97.8%** | 97.9% | -0.1 |
| LIBERO-10 | **95.8%** | 94.5% | +1.3 |
| **Average** | **97.0%** | **~96.8%** | **+0.2** |

Verdict: **Reproduced** (avg matches reported).

Key fixes:
1. `num_images_in_input: 2` (was 1) — fixed spatial from 1.6% → 97.2%
2. Per-suite `unnorm_key` — joint checkpoint has 4 separate keys
   (`libero_{suite}_no_noops`). Must run 4 server instances with different unnorm_keys.

Result JSONs archived at `.claude/reproductions/oft-joint/`.

---

### StarVLA (4 variants) — 0% across all

Tested: Qwen2.5-VL-{FAST, OFT, GR00T} and Qwen3-VL-OFT, all on LIBERO-4in1 checkpoints.
`predict_batch()` implemented and working. Gripper double-transform removed. `send_state: true`
and `absolute_action: true` tried.

Action debug output shows absolute EEF positions `[0.86, 0.45, 0.05, ...]` with gripper=1.0.
Values are nearly constant across steps — robot doesn't move.

Root cause analysis:
- `unnormalize_actions()` maps [-1,1] → q01/q99 range, but the resulting values may not
  match LIBERO's expected action space
- Need to compare with StarVLA's own LIBERO eval script (not found in public repo)
- May need to bypass `unnormalize_actions()` entirely for LIBERO checkpoints

**StarVLA Qwen3-PI**: Server crash — DiT state_dict has 36 blocks, model config expects 16.

---

### GR00T N1.6 (community) — `0xAnkitSingh/GR00T-N1.6-LIBERO`

0% success despite `send_wrist_image: true` + `send_state: true` + `predict_batch()`
with multi-video-key support.

Community checkpoint may require specific observation preprocessing or action post-processing
not documented. Official NVIDIA LIBERO finetuned checkpoint does not exist — this is a
third-party checkpoint with unknown evaluation protocol.

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
