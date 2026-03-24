# Reproduced Performance — LIBERO Stage 1

> Evaluation date: 2026-03-24
> Harness version: 0.0.4.dev8+g3e4c63d37
> Hardware: Model servers on DGX-H100 (80GB), benchmarks on A100 nodes
> Protocol: 50 episodes/task, seed=7, num_steps_wait=10

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

**Merge**:
```bash
uv run vla-eval merge results/xvla-libero/LIBEROBenchmark_libero_spatial_shard*of50.json \
  -o results/xvla-libero/merged_libero_spatial.json
# repeat for libero_object, libero_goal, libero_10
```

**Server log** (`results/slurm_logs/xvla-libero_103959.err`):
```
10:22:12 INFO  Pre-loading model...
10:22:15 INFO  Loading X-VLA from 2toINF/X-VLA-Libero
10:22:21 INFO  X-VLA model loaded on cuda:0 (float32, profile=libero)
10:22:21 INFO  server listening on 0.0.0.0:8001
```
</details>

---

*Results below will be added as each model completes evaluation.*

### OpenVLA-OFT (7B, joint) — `moojink/...-libero-spatial-object-goal-10`

Status: **In progress** (40/50 shards spatial+object, 20/50 goal+10)

### OpenVLA (7B, per-suite LoRA) — `openvla/openvla-7b-finetuned-libero-{suite}`

Status: **In progress** (spatial/object/goal running, 10 server pending)

### Pi0.5 — `pi05_libero` (via openpi)

Status: **Waiting** (server ready, benchmark not yet started)

### StarVLA (Qwen2.5-FAST/OFT/GR00T, Qwen3-OFT)

Status: **Waiting** (4 servers ready on DGX-H100-11)

### GR00T N1.6 (community) — `0xAnkitSingh/GR00T-N1.6-LIBERO`

Status: **Waiting** (server ready)

---

## Troubleshooting Log

| Issue | Model | Resolution |
|---|---|---|
| 0% success rate across all models | All | Benchmark params were wrong — each model needs specific `absolute_action`, `send_wrist_image`, `send_state`, `state_format` settings |
| `unnorm_key` not found | OpenVLA base | Finetuned checkpoints use `libero_{suite}` not `libero_{suite}_no_noops` |
| Port conflicts on shared nodes | All | Assign unique ports per server on same node |
| `wrist_image not in obs` | GR00T | Added `send_wrist_image=true` to benchmark config |
| DiT state_dict layer mismatch | StarVLA Qwen3-PI | Checkpoint has 36 DiT blocks, model config expects 16. Server crash. **Skipped.** |
| Pi0-FAST vs Pi0.5 | Pi0 | `pi0_fast_libero` has no reported numbers; switched to `pi05_libero` |
| TF CUDA JIT on H100 | OFT-joint | 30+ min startup due to TF recompiling CUDA kernels for compute 9.0 |
| `vla-eval` not in PATH after node change | All | Use `uv run vla-eval` instead of bare `vla-eval` |

## Observations

### Batch prediction impact on throughput

No model server currently implements `predict_batch()`. All use single-observation `predict()`.

| Model | predict_batch | 2000 ep (30 shards) | Per-episode |
|---|---|---|---|
| DB-CogACT | **Yes** | ~20 min | ~0.6s |
| X-VLA | No | ~16 min (50 shards) | ~15s |
| OFT-joint | No | ~35 min | ~30s |
| OpenVLA base | No | ~3 hours | ~2.5 min |
| Pi0.5 | No | ~3.5 hours | ~3 min |

DB-CogACT's batch prediction gives **~5× throughput** over the next-fastest non-batch model.
This is the single highest-impact optimization for evaluation speed. Priority: implement
`predict_batch()` for OpenVLA, OFT, Pi0, and StarVLA model servers.
