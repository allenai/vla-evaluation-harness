# Reproduced Performance

Reproduction of published VLA model benchmark scores using vla-eval.

## Measurement Protocol

**Hardware:**
- Model server: DGX-H100-80GB SXM (slurm, h100 partition)
- Benchmark host: A100-Zebra (96× Xeon Gold 6336Y cores, 2× A100-80GB PCIe, 503GB RAM)

**Software:**
- Harness: vla-eval on branch `add-reproductions`
- Docker: `ghcr.io/allenai/vla-evaluation-harness/libero:latest` (rebuilt per evaluation)

**LIBERO protocol:** 4 suites (Spatial, Object, Goal, 10) × 10 tasks × 50 episodes = 2000 episodes/model.
Seed=7, num_steps_wait=10, max_steps per suite (Spatial=220, Object=280, Goal=300, 10=520).

**Verdict criteria** (binomial 95% CI for 500 episodes per suite):
- At p=0.95: CI ≈ ±1.9pp. At p=0.97: CI ≈ ±1.5pp.
- **Reproduced**: reproduced score within 95% CI of reported score.
- **Approximate**: outside CI but ≤5pp gap.
- **Not reproduced**: >5pp gap, or known systematic issue.

---

## Stage 1 — LIBERO

### Results

| Model | Spatial | Object | Goal | 10 | **Avg** | Reported | Verdict |
|-------|:-------:|:------:|:----:|:--:|:-------:|:--------:|:-------:|
| X-VLA (0.9B) | 98.0% | 98.0% | 98.0% | 94.8% | **97.2%** | 98.1% | Reproduced |
| Pi0.5 | 98.0% | 99.6% | 98.6% | 94.6% | **97.7%** | 96.9% | Reproduced |
| GR00T N1.6 | 96.6% | 98.4% | 96.8% | 87.8% | **94.9%** | 97.0% | Approximate (−2.1pp) |
| OFT (joint) | 94.0% | — | — | — | **—** | ~96.8% | Spatial only (−3.6pp) |

Each value = successful episodes / total episodes (500 per suite).
Raw result JSONs: [`data/`](data/).

### Per-model Reproduction Notes

**X-VLA** (`2toINF/X-VLA-Libero`, `benchmark_profile=libero`):
- Uses `controller_states` (from `robot.controller.ee_pos/ee_ori_mat`), NOT `states` (from `raw_obs`).
  The observation quaternion (`robot0_eef_quat`) differs from the controller rotation matrix by ~90°
  due to coordinate frame differences. X-VLA was trained on controller data; using observation data
  yields 42%. See commit `27e63c0`.
- `unflip_wrist=True`: benchmark flips all images; X-VLA was trained with unflipped wrist.
- `absolute_action=True`: X-VLA outputs absolute EE poses, not deltas.
- All params auto-negotiated via HELLO (`get_observation_params()`).

**Pi0.5** (`pi05_libero` via openpi):
- Uses `states` (from `raw_obs`), 8D `[pos3, axisangle3, gripper2]`.
- `send_wrist_image=True`, `send_state=True`. `image_resolution=224`.
- Note: `pi0_fast_libero` is a different, lower-performing model.

**GR00T N1.6** (`0xAnkitSingh/GR00T-N1.6-LIBERO`, community checkpoint):
- `invert_gripper=True`: model outputs gripper [0,1] (0=close), LIBERO expects [-1,1] (-1=open).
- `embodiment_tag=LIBERO_PANDA`, `chunk_size=16`.
- −2.1pp gap vs reported may be due to community checkpoint vs official NVIDIA finetuning.

**OFT** (`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`, joint checkpoint):
- Requires per-suite `unnorm_key` — 4 server instances or 4 sequential runs.
- `num_images_in_input=2` (3rd-person + wrist), `send_state=True`.
- TF CUDA JIT on H100 takes 30+ min at startup. Ensure server is ready before launching shards.
- Per-suite checkpoints (`moojink/openvla-7b-oft-finetuned-libero-{suite}`) show anomalous results
  (spatial 92%, goal 18%) — likely HuggingFace checkpoint issues. Use joint checkpoint instead.

### Excluded Models

| Model | Reason |
|-------|--------|
| StarVLA Q2.5-OFT/Q3-OFT/Q2.5-FAST | Supply <7 obs/s (chunk_size=1). Single shard: 4-20 hours. |
| StarVLA Q2.5-GR00T | Supply 38 obs/s but still slow. Partial result: 29.6% spatial (reported 95.4%). |
| StarVLA Qwen3-PI | state_dict mismatch: 36 vs 16 DiT transformer blocks. Server crash. |
| OpenVLA base (LoRA) | chunk_size=1, no batch prediction. ~3 hours per suite. |

### Bugs Found During Reproduction

| Bug | Impact | Fix |
|-----|--------|-----|
| `raw_obs["robot0_eef_quat"]` ≠ `robot.controller.ee_ori_mat` (~90° frame diff) | X-VLA 42% → 98% | Benchmark sends both; X-VLA reads `controller_states` |
| StarVLA gripper polarity `2x-1` inverted | Gripper open/close swapped | Changed to `1-2x` |
| GR00T missing gripper normalization/inversion | ~1% success | Added `invert_gripper` flag |
| OFT `num_images_in_input=1` (should be 2) | Missing wrist image | Fixed in `_base.yaml` |
| Pi0 default `pi0_fast_libero` (wrong model) | Wrong checkpoint loaded | Changed to `pi05_libero` |
| Smoke test `success` check broken after metrics refactor | All smoke tests failing | Fixed to read `metrics.success` |
| Shard merge dropped `server_info`, `harness_version`, `created_at` | Provenance lost | Fixed in `merge.py` |
| Port conflicts: multiple servers on same port | OFT evaluated against wrong models | Run models sequentially, verify ports |

---

## Stage 2 — Cross-benchmark

| Model | LIBERO | CALVIN | SimplerEnv | RoboTwin |
|-------|:------:|:------:|:----------:|:--------:|
| DB-CogACT | 95.2% | 4.05 avg len | 72.2% | — |

Details: [db-cogact.md](db-cogact.md)

---

## Supply — Model Server Throughput

Measured with `experiments/bench_supply.py` on H100-80GB SXM.
Command: `uv run python experiments/bench_supply.py --url ws://HOST:PORT --num-clients 4 --requests-per-client 60 --image-size 256`
Observation payload: 2× 256×256 RGB images (agentview + wrist) + 8D state.
All models at `max_batch_size=1` (no batching).

| Model | chunk_size | μ (obs/s) | Median latency |
|-------|:---------:|:---------:|:--------------:|
| X-VLA | 30 | 88.8 | 30ms |
| Pi0.5 | 10 | 84.0 | 63ms |
| GR00T N1.6 | 16 | 46.5 | 50ms |
| StarVLA Q2.5-GR00T | 1 | 38.3 | 60ms |
| OFT (joint) | 10 | 27.1 | 46ms |
| StarVLA Q2.5-OFT | 1 | 6.0 | 654ms |
| StarVLA Q3-OFT | 1 | 5.9 | 664ms |
| StarVLA Q2.5-FAST | 1 | 1.4 | 2858ms |

StarVLA/GR00T support `predict_batch()`. X-VLA/Pi0/OFT are single-predict only.

## Demand — Benchmark Observation Rate

Measured with `experiments/bench_demand.py` on A100-Zebra.
Command: `uv run python experiments/bench_demand.py --config CONFIG --shards N --episodes-per-shard 5 --gpus G --timeout 300`
Median CPU/GPU utilization during steady-state (startup transients excluded).

Full per-N sweep data: see [`../tuning-guide.md`](../tuning-guide.md).

### Bottleneck Summary

| Benchmark | Peak λ (obs/s) | Peak N | Bottleneck | 2 GPU effect | Rec. GPUs |
|-----------|:--------------:|:------:|:----------:|:------------:|:---------:|
| LIBERO | 415 | 50 | CPU (52%) | No change | 1 |
| CALVIN | 407 | 24 | CPU (93%) | No change | 1 |
| SimplerEnv | 138 | 24 | GPU (43%) | Worse (overhead) | 1 |
| RoboTwin | 4.9 | 16 | GPU (100%) | 2× improvement | 2 |

## How to Run

```bash
# 1. Build Docker
docker/build.sh libero

# 2. Start model server (slurm, one per GPU)
sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 24:00:00 \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --address 0.0.0.0:8001 -v"

# 3. Wait for server ready
curl -s --max-time 2 "http://DGX-H100-XX:8001/config"

# 4. Run benchmark (ONE MODEL AT A TIME — shard filenames collide across models)
SHARDS=10  NODE=DGX-H100-XX  MODEL=xvla
for i in $(seq 0 $((SHARDS-1))); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://${NODE}:8001 \
    --shard-id $i --num-shards $SHARDS --yes &
done
wait

# 5. Archive shards + merge
mkdir -p docs/reproductions/data/${MODEL}-libero/shards
cp results/LIBEROBenchmark_*shard*of${SHARDS}.json docs/reproductions/data/${MODEL}-libero/shards/
uv run vla-eval merge results/LIBEROBenchmark_*_shard*of${SHARDS}.json \
  -o docs/reproductions/data/${MODEL}-libero/merged.json
rm results/LIBEROBenchmark_*shard*of${SHARDS}.json

# 6. Next model — clean shards before starting
```

**Critical notes:**
- Run ONE model at a time. Merge and clean shards before the next.
- Max 50 Docker containers on benchmark host.
- Verify server port is free before launching (no stale servers on same port).
- OFT: TF JIT takes 30+ min. Confirm server ready via `curl /config` before launching shards.
- OFT joint: requires per-suite unnorm_key. Run 4 sequential passes, or 4 server instances on different ports.

## Reference

- [reported-performance.md](reported-performance.md) — Officially reported scores from papers/model cards.
- [db-cogact.md](db-cogact.md) — DB-CogACT cross-benchmark reproduction report.
- [`../tuning-guide.md`](../tuning-guide.md) — Supply/demand measurement methodology.
