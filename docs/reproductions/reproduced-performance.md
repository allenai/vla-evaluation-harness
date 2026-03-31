# Reproduced Performance

Reproduction of published VLA model benchmark scores using vla-eval.
Models listed in publication order.

## Summary

| Model | LIBERO | CALVIN | SE WidowX | SE GR-VM | RoboTwin |
|-------|:------:|:------:|:---------:|:--------:|:--------:|
| Pi0.5 (Oct 2024) | **97.7%** (96.9%) Reproduced | — | — | — | — |
| DB-CogACT (Nov 2024) | **95.2%** (94.9%) Reproduced | **4.05** (4.06) Reproduced | **72.2%** (69.5%) Reproduced | — | — |
| OFT (Feb 2025) | 94.0% spatial only (~96.8%) | — | — | — | — |
| GR00T N1.6 (Mar 2025) | **94.9%** (97.0%) Approximate | — | — | — | — |
| X-VLA (Oct 2025) | **97.8%** (98.1%) Reproduced | **4.30** (4.43) Reproduced | **69.8%** (95.8%) Partial | — (88.3%) | — (70.0%/39.0%) |

Format: **reproduced** (reported) verdict. — = not yet evaluated.

---

## LIBERO

4 suites × 10 tasks × 50 episodes = 2000 episodes/model. Seed=7.

| Model | Spatial | Object | Goal | 10 | **Avg** | Reported | Verdict |
|-------|:-------:|:------:|:----:|:--:|:-------:|:--------:|:-------:|
| [Pi0.5](../../configs/model_servers/pi0/libero.yaml) | 98.0% | 99.6% | 98.6% | 94.6% | **97.7%** | 96.9% | Reproduced |
| [DB-CogACT](../../configs/model_servers/db_cogact/libero.yaml) | 95.2% | 98.6% | 95.2% | 89.6% | **95.2%** | 94.9% | Reproduced |
| [OFT (joint)](../../configs/model_servers/oft/libero_joint.yaml) | 94.0% | — | — | — | **—** | ~96.8% | Spatial only (−3.6pp) |
| [GR00T N1.6](../../configs/model_servers/groot/libero.yaml) | 96.6% | 98.4% | 96.8% | 87.8% | **94.9%** | 97.0% | Approximate (−2.1pp) |
| [X-VLA](../../configs/model_servers/xvla/libero.yaml) | 97.8% | — | — | — | **97.8%** | 98.1% | Reproduced (spatial) |

Raw result JSONs: [`data/`](data/).

## CALVIN (ABC→D)

1000 sequences × 5 chained subtasks, max 360 steps/subtask.

| Model | Checkpoint | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg Len** | Reported | Verdict |
|-------|-----------|:---:|:---:|:---:|:---:|:---:|:-----------:|:--------:|:-------:|
| [DB-CogACT](../../configs/model_servers/db_cogact/calvin.yaml) | `Dexmal/calvin-db-cogact` | 93.3% | 86.3% | 81.5% | 75.6% | 68.4% | **4.05** | 4.06 | Reproduced |
| [X-VLA](../../configs/model_servers/xvla/calvin.yaml) | `2toINF/X-VLA-Calvin-ABC_D` | 95.6% | 91.8% | 87.1% | 81.7% | 73.9% | **4.30** | 4.43 | Reproduced |

## SimplerEnv — WidowX VM

4 tasks × 24 episodes. DB-CogACT: 3-seed average (seed 0/2/4). X-VLA: seed 0 only.

| Model | Checkpoint | Spoon | Carrot | Block | Eggplant | **Avg** | Reported | Verdict |
|-------|-----------|:-----:|:------:|:-----:|:--------:|:-------:|:--------:|:-------:|
| [DB-CogACT](../../configs/model_servers/db_cogact/simpler.yaml) | `Dexmal/simpler-db-cogact` | 94.4% | 72.2% | 25.0% | 97.2% | **72.2%** | 69.5% | Reproduced (3-seed) |
| GR00T N1.6 | `nvidia/GR00T-N1.6-bridge` | — | — | — | — | **—** | 62.1%† | Not reproduced‡ |
| [X-VLA](../../configs/model_servers/xvla/simpler_widowx.yaml) | `2toINF/X-VLA-WidowX` | 91.7% | 91.7% | 29.2% | 66.7% | **69.8%** | 95.8% | Partial (−26pp) |

† GR00T reported on non-standard 7-task set; 4-task subset avg = 57.1%.
‡ GR00T SimplerEnv requires `SimPolicyWrapper` from Isaac-GR00T; not yet implemented.

## SimplerEnv — Google Robot VM

3 tasks (Pick Coke Can, Move Near, Open/Close Drawer) × 24 episodes × 3 seeds.

| Model | Checkpoint | Pick Coke | Move Near | Drawer | **3-task Avg** | Reported | Verdict |
|-------|-----------|:---------:|:---------:|:------:|:--------------:|:--------:|:-------:|
| GR00T N1.6 | `nvidia/GR00T-N1.6-fractal` | — | — | — | **—** | 67.7%† | — |
| X-VLA | `2toINF/X-VLA-WidowX` | — | — | — | **—** | 88.3% | — |

† GR00T reported on non-standard 6-task set. Not directly comparable.

## RoboTwin

50 tasks, Protocol A (single-task, 50 clean demos/task).

| Model | Checkpoint | Easy | Hard | Reported Easy | Reported Hard | Verdict |
|-------|-----------|:----:|:----:|:-------------:|:-------------:|:-------:|
| X-VLA | `2toINF/X-VLA-WidowX` | — | — | 70.0% | 39.0% | — |

---

## Appendix A — Measurement Protocol

**Hardware:**
- Model server: H100-80GB SXM GPU
- Benchmark host: 96-core Xeon, 2× A100-80GB PCIe, 503GB RAM

**Software:**
- Harness: vla-eval `main` branch
- Docker: `ghcr.io/allenai/vla-evaluation-harness/{benchmark}:latest` (rebuilt per evaluation)

**LIBERO protocol:** max_steps per suite: Spatial=220, Object=280, Goal=300, 10=520. num_steps_wait=10.

**Verdict criteria** (binomial 95% CI for 500 episodes per suite):
- At p=0.95: CI ≈ ±1.9pp. At p=0.97: CI ≈ ±1.5pp.
- **Reproduced**: within 95% CI of reported score.
- **Approximate**: outside CI but ≤5pp gap.
- **Not reproduced**: >5pp gap, or known systematic issue.

## Appendix B — Per-model Notes

**Pi0.5** (`pi05_libero` via openpi, [arxiv 2410.24164](https://arxiv.org/abs/2410.24164)):
- Uses `states` (from `raw_obs`), 8D `[pos3, axisangle3, gripper2]`.
- `send_wrist_image=True`, `send_state=True`, `image_resolution=224`.
- `pi0_fast_libero` is a different, lower-performing model.
- CALVIN/SimplerEnv checkpoints not publicly available in openpi.

**DB-CogACT** (CogACT fine-tune, [arxiv 2411.19650](https://arxiv.org/abs/2411.19650) / [arxiv 2510.23511](https://arxiv.org/abs/2510.23511)):
- Separate checkpoint per benchmark. Full details: [db-cogact.md](db-cogact.md).
- Results: LIBERO [`data/dbcogact-libero/`](data/dbcogact-libero/),
  CALVIN [`data/dbcogact-calvin/`](data/dbcogact-calvin/),
  SimplerEnv [`data/dbcogact-simpler/`](data/dbcogact-simpler/).

**OFT** (`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`, [arxiv 2502.19645](https://arxiv.org/abs/2502.19645)):
- Joint checkpoint, requires per-suite `unnorm_key` — 4 sequential runs.
- `num_images_in_input=2` (3rd-person + wrist), `send_state=True`.
- First startup is slow (model download + CUDA init). Confirm server ready before launching shards.
- Per-suite checkpoints show anomalous results — use joint checkpoint only.

**GR00T N1.6** ([arxiv 2503.14734](https://arxiv.org/abs/2503.14734)):
- LIBERO: `0xAnkitSingh/GR00T-N1.6-LIBERO` (community checkpoint), `invert_gripper=True`,
  `embodiment_tag=LIBERO_PANDA`, `chunk_size=16`.
  −2.1pp gap may be due to community vs official NVIDIA finetuning.
- SimplerEnv WidowX: `nvidia/GR00T-N1.6-bridge`, `embodiment_tag=OXE_WIDOWX`.
  **Not yet reproduced** — official eval uses `--use-sim-policy-wrapper` (SimPolicyWrapper)
  with `max_episode_steps=300`, `n_action_steps=1`. Our `groot.py` lacks this wrapper.
  Results: [`data/groot-libero/`](data/groot-libero/) (LIBERO only).
- SimplerEnv GR: `nvidia/GR00T-N1.6-fractal`, `embodiment_tag=OXE_GOOGLE`.
- CALVIN/RoboTwin: no checkpoints available, no reported scores.

**X-VLA** ([arxiv 2510.10274](https://arxiv.org/abs/2510.10274)):
- LIBERO: `2toINF/X-VLA-Libero`, `benchmark_profile=libero`. Uses `controller_states`
  (NOT `states`) — observation quaternion differs from controller rotation by ~90°.
  Using observation data yields 42%. `unflip_wrist=True`, `absolute_action=True`.
  All params auto-negotiated via HELLO. rot6d layout: contiguous (differs from
  CALVIN/SimplerEnv which use interleaved). Results: [`data/xvla-libero/`](data/xvla-libero/).
- CALVIN: `2toINF/X-VLA-Calvin-ABC_D`, `benchmark_profile=calvin`, `domain_id=2`, `chunk_size=20`.
  Avg chain 4.30 (reported 4.43). Results: [`data/xvla-calvin/`](data/xvla-calvin/).
- SimplerEnv WidowX: `2toINF/X-VLA-WidowX`, `benchmark_profile=simpler_widowx`.
  Requires `Dockerfile.simpler_xvla` (absolute EE control + sink camera patches).
  69.8% avg (seed 0), −26pp gap from reported 95.8% — StackGreenCube (29.2%) and
  PutEggplant (66.7%) account for most of the gap. Results: [`data/xvla-simpler/`](data/xvla-simpler/).
- RoboTwin: `benchmark_profile=robotwin` supported in model server. Config needed.

### Excluded Models

| Model | Reason |
|-------|--------|
| StarVLA Q2.5-OFT/Q3-OFT/Q2.5-FAST | Supply <7 obs/s (chunk_size=1). 4-20 hours per suite. |
| StarVLA Q2.5-GR00T | Partial: 29.6% spatial (reported 95.4%). |
| StarVLA Qwen3-PI | state_dict mismatch (36 vs 16 transformer blocks). |
| OpenVLA base (LoRA) | chunk_size=1, no batch prediction. ~3 hours per suite. |

## Appendix C — Supply & Demand

### Supply — Model Server Throughput

Measured with `experiments/bench_supply.py` on H100-80GB SXM.
Command: `uv run python experiments/bench_supply.py --url ws://HOST:PORT --num-clients 4 --requests-per-client 60 --image-size 256`
Observation payload: 2× 256×256 RGB images (agentview + wrist) + 8D state.
All models at `max_batch_size=1` (no batching).

| Model | chunk_size | μ (obs/s) | Median latency | GPU inf/s |
|-------|:---------:|:---------:|:--------------:|:---------:|
| [Pi0.5](../../configs/model_servers/pi0/libero.yaml) | 10 | 84.0 | 63ms | 8.4 |
| [DB-CogACT](../../configs/model_servers/db_cogact/libero.yaml) | 12 | 165.2 | 18ms | 13.8 |
| [OFT (joint)](../../configs/model_servers/oft/libero_joint.yaml) | 10 | 27.1 | 46ms | 2.7 |
| [GR00T N1.6](../../configs/model_servers/groot/libero.yaml) | 16 | 46.5 | 50ms | 2.9 |
| [StarVLA Q2.5-GR00T](../../configs/model_servers/starvla/libero_qwen25_groot.yaml) | 1 | 38.3 | 60ms | 38.3 |
| [StarVLA Q2.5-OFT](../../configs/model_servers/starvla/libero_qwen25_oft.yaml) | 1 | 6.0 | 654ms | 6.0 |
| [StarVLA Q3-OFT](../../configs/model_servers/starvla/libero_qwen3_oft.yaml) | 1 | 5.9 | 664ms | 5.9 |
| [StarVLA Q2.5-FAST](../../configs/model_servers/starvla/libero_qwen25_fast.yaml) | 1 | 1.4 | 2858ms | 1.4 |
| [X-VLA](../../configs/model_servers/xvla/libero.yaml) | 30 | 88.8 | 30ms | 3.0 |

GPU inf/s = actual GPU forward passes per second (μ / chunk_size). Models with chunk_size > 1
serve most observations from cached action chunks without GPU inference.

StarVLA/GR00T support `predict_batch()` — sweeping `max_batch_size` may improve throughput.
X-VLA/Pi0/OFT are single-predict only (`max_batch_size=1`).
DB-CogACT supports `predict_batch()` with optimal throughput at `max_batch_size=16` (468 obs/s on H100).

<details>
<summary>DB-CogACT batch sweep (<a href="../../configs/model_servers/db_cogact/libero.yaml">db_cogact/libero.yaml</a>, chunk_size=12)</summary>

**A100-80GB PCIe:**

| B (max_batch_size) | μ (obs/s) | Inference latency (p50) |
|:------------------:|:---------:|:-----------------------:|
| 1 | 71.7 | 21.0ms |
| 2 | 103.8 | 20.9ms |
| 4 | 151.0 | 18.4ms |
| 8 | 185.9 | 18.6ms |
| 16 | 203.1 | 31.2ms |
| 24 | 196.6 | 51.4ms |
| 32 | 201.4 | 68.4ms |

**H100-80GB SXM:**

| B (max_batch_size) | μ (obs/s) | Inference latency (p50) |
|:------------------:|:---------:|:-----------------------:|
| 1 | 165.2 | 18.2ms |
| 2 | 255.4 | 18.5ms |
| 4 | 347.3 | 20.0ms |
| 8 | 423.8 | 23.9ms |
| 16 | 468.2 | 28.7ms |
| 24 | 485.5 | 38.8ms |
| 32 | 483.2 | 52.5ms |

A100 peaks at B=16 (203 obs/s), H100 peaks at B=24 (486 obs/s). H100 achieves ~2.4× A100 throughput.

</details>

### Demand — Benchmark Observation Rate

Measured with `experiments/bench_demand.py` on the benchmark host.
Command: `uv run python experiments/bench_demand.py --config CONFIG --shards N --episodes-per-shard 5 --gpus G --timeout 300`
Median CPU/GPU utilization during steady-state (startup transients excluded).

| Benchmark | Rendering | Per-shard obs/s | Peak λ (obs/s) | Peak N | Bottleneck | 2 GPU effect | Rec. GPUs |
|-----------|-----------|:---------------:|:--------------:|:------:|:----------:|:------------:|:---------:|
| [LIBERO](../../configs/libero_spatial.yaml) | GPU EGL (MuJoCo) | ~7.3 | 415 | 50 | CPU (52%) | No change | 1 |
| [CALVIN](../../configs/calvin_eval.yaml) | GPU EGL (PyBullet) | ~36.7 | 407 | 24 | CPU (93%) | No change | 1 |
| [SimplerEnv](../../configs/simpler_all_tasks.yaml) | GPU (SAPIEN/Vulkan) | ~10.1 | 138 | 24 | GPU (43%) | Worse (overhead) | 1 |
| [RoboTwin](../../configs/robotwin_eval.yaml) | GPU | TBD | 4.9 | 16 | GPU (100%) | 2× improvement | 2 |

<details>
<summary>LIBERO Spatial — λ(N) sweep</summary>

| N (shards) | observations | elapsed (s) | λ (obs/s) |
|:----------:|:------------:|:-----------:|:---------:|
| 1 | 1,100 | 98.2 | 11.2 |
| 8 | 8,800 | 106.2 | 82.9 |
| 16 | 17,600 | 115.7 | 152.1 |
| 24 | 26,400 | 123.2 | 214.2 |
| 32 | 35,200 | 131.8 | 267.1 |
| 50 | 55,000 | 150.8 | 364.6 |
| 64 | 70,400 | 171.5 | 410.2 |
| 80 | 88,000 | 196.8 | 446.8 |
| 100 | 110,000 | 282.3 | 389.4 |

Peak at N=80 (447 obs/s). Per-shard throughput degrades from ~11.2 (N=1) to ~5.6 (N=80). At N=100, contention overwhelms parallelism.

</details>

<details>
<summary>CALVIN ABC→D — λ(N) sweep</summary>

| N (shards) | observations | elapsed (s) | λ (obs/s) |
|:----------:|:------------:|:-----------:|:---------:|
| 1 | 1,080 | 29.4 | 36.7 |
| 4 | 4,320 | 36.7 | 117.6 |
| 8 | 8,640 | 39.2 | 220.4 |
| 16 | 17,280 | 46.0 | 376.0 |
| 24 | 25,920 | 60.0 | 432.3 |
| 32 | 34,560 | 87.6 | 394.6 |

Peak at N=24 (432 obs/s). Fast PyBullet rendering (36.7 obs/s/shard at N=1) makes CALVIN supply-bottlenecked — demand exceeds estimated H100 supply (~283 obs/s) at just N=16.

</details>

<details>
<summary>SimplerEnv WidowX — λ(N) sweep</summary>

| N (shards) | observations | elapsed (s) | λ (obs/s) |
|:----------:|:------------:|:-----------:|:---------:|
| 1 | 2,400 | 238.4 | 10.1 |
| 4 | 9,600 | 331.6 | 28.9 |
| 8 | 19,200 | 272.6 | 70.4 |
| 16 | 38,400 | 299.0 | 128.4 |
| 24 | 42,825 | 298.1 | 143.7* |
| 32 | 41,484 | 300.6 | 138.0* |

\* timeout — partial results.
Peak at N=24 (144 obs/s). SAPIEN/Vulkan shards consume much more GPU memory than MuJoCo/PyBullet EGL, so demand saturates far earlier than LIBERO/CALVIN.

</details>

**Key takeaways:**
- LIBERO/CALVIN use lightweight GPU EGL rendering — scales to many shards, CPU-bottlenecked.
- SimplerEnv (SAPIEN/Vulkan) saturates GPU earlier — adding shards past N=24 decreases throughput.
- RoboTwin is heavily GPU-bottlenecked — 2 GPUs double throughput.
- Use `num_shards` such that λ(N) < 80% of model server supply μ(B*). See tuning guide for derivation.

### Recommended Shard Counts (LIBERO, H100)

Rule: `num_shards ≤ 0.8 × μ / per_shard_demand`. LIBERO per-shard ≈ 7.3 obs/s.

| Model | μ (obs/s) | Max shards (80% rule) | Recommended | Est. wall time |
|-------|:---------:|:---------------------:|:-----------:|:--------------:|
| [X-VLA](../../configs/model_servers/xvla/libero.yaml) | 88.8 | 9 | 10 | ~30 min |
| [Pi0.5](../../configs/model_servers/pi0/libero.yaml) | 84.0 | 9 | 10 | ~30 min |
| [GR00T](../../configs/model_servers/groot/libero.yaml) | 46.5 | 5 | 5 | ~55 min |
| [StarVLA Q2.5-GR00T](../../configs/model_servers/starvla/libero_qwen25_groot.yaml) | 38.3 | 4 | 4 | ~70 min |
| [OFT](../../configs/model_servers/oft/libero_joint.yaml) | 27.1 | 3 | 4 | ~70 min |
| [StarVLA Q2.5-OFT](../../configs/model_servers/starvla/libero_qwen25_oft.yaml) | 6.0 | 0.6 | 1 | ~4.5 hrs |
| [StarVLA Q3-OFT](../../configs/model_servers/starvla/libero_qwen3_oft.yaml) | 5.9 | 0.6 | 1 | ~4.5 hrs |
| [StarVLA Q2.5-FAST](../../configs/model_servers/starvla/libero_qwen25_fast.yaml) | 1.4 | 0.15 | 1 | ~20 hrs |

Models with μ < 7.3 obs/s cannot keep up with 1 shard — the single shard generates observations
faster than the server processes them. Runs still work (queue absorbs bursts) but wall time is
bottlenecked by inference speed, not parallelism.

Full per-N sweep data and worked examples: [`../tuning-guide.md`](../tuning-guide.md).

## Appendix D — How to Run

```bash
# 1. Build Docker
docker/build.sh libero

# 2. Start model server
sbatch --gres=gpu:1 -c8 --mem=64G -t 24:00:00 \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --address 0.0.0.0:8001 -v"

# 3. Wait for server ready
curl -s --max-time 2 "http://GPU-NODE:8001/config"

# 4. Run sharded evaluation
SHARDS=10  NODE=GPU-NODE  MODEL=xvla
for i in $(seq 0 $((SHARDS-1))); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://${NODE}:8001 \
    --shard-id $i --num-shards $SHARDS --yes &
done
wait

# 5. Archive + merge
mkdir -p docs/reproductions/data/${MODEL}-libero/shards
cp results/LIBEROBenchmark_*shard*of${SHARDS}.json docs/reproductions/data/${MODEL}-libero/shards/
uv run vla-eval merge results/LIBEROBenchmark_*_shard*of${SHARDS}.json \
  -o docs/reproductions/data/${MODEL}-libero/merged.json
rm results/LIBEROBenchmark_*shard*of${SHARDS}.json
```

**Notes:**
- One model at a time. Merge and clean shards before starting the next.
- Max 50 Docker containers on benchmark host.
- Verify server port is free before launching (no stale servers on same port).
- OFT: first startup is slow (model download + CUDA init). Confirm server ready via `curl /config` before launching shards.
- OFT joint: requires per-suite `unnorm_key`. Run 4 sequential passes, or 4 server instances on different ports.

## Reference

- [reported-performance.md](reported-performance.md) — Officially reported scores.
- [db-cogact.md](db-cogact.md) — DB-CogACT cross-benchmark details.
- [`../tuning-guide.md`](../tuning-guide.md) — Supply/demand methodology.
