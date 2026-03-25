# Stage 1 — LIBERO Reproducibility Audit

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

Result JSONs archived at `docs/reproductions/data/xvla-libero/`.
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

Result JSONs archived at `docs/reproductions/data/pi0-libero/`.

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

Result JSONs archived at `docs/reproductions/data/oft-joint/`.

---

### StarVLA (4 variants) — 0% across all

Tested: Qwen2.5-VL-{FAST, OFT, GR00T} and Qwen3-VL-OFT, all on LIBERO-4in1 checkpoints.

**Fixes applied** (all verified, none resolved 0%):
- `predict_batch()` implemented (batch inference works, fast)
- Gripper transform: `2.0 * actions[:, 6] - 1.0` (map {0,1} → {-1,+1})
- `image_resolution: 224` (official eval uses 224)
- `flip_image: false` (official eval does NOT flip)
- `send_wrist_image: true` (official eval sends 2 images: agentview + wrist)
- `send_state: true` tried (official eval collects state but doesn't pass to model)

**Official eval code comparison** (`examples/LIBERO/eval_files/`):
- Code path is identical: `predict_action()` → `unnormalize_actions()` → `env.step()`
- Action stats q01/q99 are delta-range values (correct for LIBERO delta actions)
- Model outputs near-zero normalized actions → small but nonzero deltas after unnorm

**Remaining hypothesis**: The model IS receiving different input than expected despite
all visible preprocessing matching. Possible causes:
1. **Image ordering**: model expects `[primary, wrist]` but harness sends dict values
   in undefined order. Need to verify key ordering in `obs["images"]`.
2. **editable=true not working**: server may use cached non-editable vla-eval package
   that doesn't have our fixes. Need to clear uv cache and verify.
3. **Tokenizer/processor mismatch**: model's internal image processor may expect
   different normalization than raw uint8.

**Resolution plan**:
1. Run official StarVLA eval script (`examples/LIBERO/eval_files/eval_libero.py`)
   directly → confirm it gets ~95% → then add debug logging to both scripts
   and compare observations byte-for-byte.
2. Alternative: add observation logging to harness and StarVLA's official eval,
   save first observation from each, and diff.

**StarVLA Qwen3-PI**: Server crash — DiT state_dict has 36 blocks, model config expects 16.
Requires `num_layers` compat fix in `starvla.py`.

---

### GR00T N1.6 (community) — `0xAnkitSingh/GR00T-N1.6-LIBERO`

Spatial: 1.2% (30 shards). Goal: 1 SUCCESS in quick test. Other suites ~0%.

**Fixes applied**:
- `send_wrist_image: true`, `send_state: true`
- `predict_batch()` with multi-video-key and state decomposition (flat 8D → per-key)

**Root cause identified** (from HuggingFace model card):
- `Gr00tPolicy` must be initialized with **`use_sim_policy_wrapper=True`**
- This wrapper handles observation preprocessing (image normalization, state formatting)
  and action post-processing automatically
- Our server creates `Gr00tPolicy` without this wrapper → raw obs/action mismatch

**Resolution plan**:
1. Add `use_sim_policy_wrapper: true` to `GR00TModelServer._load_model()`
2. The wrapper likely handles all obs/action conversion internally,
   making our manual state decomposition unnecessary
3. Model path may also be wrong: card shows `0xAnkitSingh/GR00T-N1.6-3B_LIBERO`
   vs our config `0xAnkitSingh/GR00T-N1.6-LIBERO` — verify correct HF repo

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

#### Immediate (Stage 1 completion)

1. **StarVLA 0% debug**: Run official eval script (`/tmp/StarVLA/examples/LIBERO/eval_files/eval_libero.py`)
   directly on H100 → confirm ~95% → add observation logging to both scripts → byte-for-byte diff.
   Key suspects: image dict key ordering, editable install not reflecting code changes.

2. **GR00T fix**: Add `use_sim_policy_wrapper=True` to `Gr00tPolicy()` in `groot.py:125`.
   Verify correct model path (`GR00T-N1.6-3B_LIBERO` vs `GR00T-N1.6-LIBERO`).
   This is likely a 1-line fix that resolves the entire 0% issue.

3. **Pi0.5 spatial gap (92.6% vs 98.8%)**: Investigate image flip — Pi0.5 official eval
   flips images (`[::-1, ::-1]`) but our harness may apply different flip behavior for
   the 224×224 path. Low priority since avg is already 96.4%.

4. **OpenVLA base**: Implement `predict_batch()` for `openvla.py` model server, then
   evaluate with 30 shards (currently ~3h without batch).

#### Infrastructure (Stage 2 prep)

5. **Benchmark profile system**: Generalize X-VLA's `benchmark_profile` pattern to all
   model servers. Each model server config should declare required benchmark params
   (`absolute_action`, `send_wrist_image`, `send_state`, `image_resolution`, `flip_image`,
   `state_format`) so the eval config is auto-generated correctly.

6. **predict_batch() for remaining servers**: OpenVLA, OFT (TensorFlow-based, harder).
   Pi0 blocked by openpi's single-observation `infer()` API.

7. **Stage 2 planning**: Expand to CALVIN, SimplerEnv, RoboTwin for top models
   (X-VLA, OFT, DB-CogACT). Requires new benchmark Docker images + model server configs.
4. **Fix OFT-joint**: Set `num_images_in_input: 2` and verify `unnorm_key`.
5. **Implement predict_batch for OpenVLA/OFT**: Critical for reasonable eval times.
