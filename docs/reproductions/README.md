# Reproduction Roadmap

Systematic verification that vla-eval reproduces published model scores.

## Stages

### Stage 1 — Breadth: many models × LIBERO

| Model | Config | Reported | Status |
|-------|--------|:--------:|--------|
| X-VLA (0.9B) | `xvla/libero.yaml` | 98.1% | Ready |
| Pi0.5 | `pi0/libero.yaml` | 96.9% | Ready |
| OpenVLA-OFT (joint) | `oft/libero_joint.yaml` | ~96.8% | Ready |
| StarVLA Q2.5-FAST | `starvla/libero_qwen25_fast.yaml` | 95.2% | Ready (slow) |
| StarVLA Q2.5-OFT | `starvla/libero_qwen25_oft.yaml` | 96.1% | Ready |
| StarVLA Q2.5-GR00T | `starvla/libero_qwen25_groot.yaml` | 95.4% | Ready |
| StarVLA Q3-OFT | `starvla/libero_qwen3_oft.yaml` | 96.6% | Ready |
| GR00T N1.6 | `groot/libero.yaml` | 97.0% | Ready |

Not included: OpenVLA base (too slow), StarVLA Qwen3-PI (state_dict mismatch).

Execution guide: [stage1-libero.md](stage1-libero.md)

### Stage 2 — Depth: key models × multiple benchmarks

| Model | LIBERO | CALVIN | SimplerEnv | RoboTwin | Report |
|-------|:------:|:------:|:----------:|:--------:|--------|
| DB-CogACT | Done | Done | Done | — | [db-cogact.md](db-cogact.md) |
| X-VLA | Stage 1 | Pending | Pending | Pending | |
| Pi0 | Stage 1 | Pending | Pending | Pending | |
| GR00T | Stage 1 | Pending | Pending | — | |

All configs under `configs/model_servers/`.

---

## Supply — Model Server Throughput (H100-80GB SXM)

Measured with `experiments/bench_supply.py`, `max_batch_size=1`, 256×256 images.

| Model | chunk_size | μ (obs/s) | Inference latency | GPU inf/s |
|-------|:---------:|:---------:|:-----------------:|:---------:|
| X-VLA | 30 | 88.8 | 30ms | 3.0 |
| Pi0.5 | 10 | 84.0 | 63ms | 8.4 |
| GR00T N1.6 | 16 | 46.5 | 50ms | 2.9 |
| StarVLA Q2.5-GR00T | 1 | 38.3 | 60ms | 38.3 |
| OFT (joint) | 10 | 27.1 | 46ms | 2.7 |
| StarVLA Q2.5-OFT | 1 | 6.0 | 654ms | 6.0 |
| StarVLA Q3-OFT | 1 | 5.9 | 664ms | 5.9 |
| StarVLA Q2.5-FAST | 1 | 1.4 | 2858ms | 1.4 |

StarVLA/GR00T support `predict_batch()` — sweeping `max_batch_size` may improve throughput.
X-VLA/Pi0/OFT are single-predict only (`max_batch_size=1`).

## Demand — Benchmark Observation Rate

Measured on **A100-Zebra** (96 CPU cores, 2× A100-80GB PCIe, 503GB RAM) with
`experiments/bench_demand.py`. Values show median CPU/GPU utilization during
steady-state (startup transients excluded).

### LIBERO Spatial (GPU EGL, MuJoCo)

| N | λ (obs/s) | CPU% | GPU% | GPU_MEM | SYS_RAM | Bottleneck |
|--:|----------:|-----:|-----:|--------:|--------:|:----------:|
| 1 | 14.9 | 3 | 4 | 0.5GB | 18GB | — |
| 8 | 108.3 | 10 | 22 | 4.3GB | 31GB | — |
| 16 | 205.8 | 19 | 29 | 8.6GB | 46GB | — |
| 32 | 370.6 | 35 | 36 | 17GB | 78GB | — |
| 50 | 415.3 | 52 | 52 | 27GB | 115GB | CPU |

2 GPU: no improvement (415→423). **CPU is the bottleneck.** MuJoCo physics is CPU-bound;
EGL rendering is lightweight. Peak λ≈415 at N=50.

### CALVIN ABC→D (GPU EGL, PyBullet)

| N | λ (obs/s) | CPU% | GPU% | GPU_MEM | SYS_RAM | Bottleneck |
|--:|----------:|-----:|-----:|--------:|--------:|:----------:|
| 1 | 29.4 | 3 | 3 | 0.3GB | 17GB | — |
| 8 | 237.5 | 12 | 1 | 2.0GB | 25GB | — |
| 16 | 395.1 | 86 | 0 | 4.0GB | 34GB | CPU |
| 24 | 406.9 | 93 | 0 | 6.0GB | 43GB | CPU |
| 32 | 380.6 | 94 | 0 | 8.0GB | 51GB | CPU (oversaturated) |

2 GPU: no improvement. **CPU is the sole bottleneck.** GPU utilization drops to 0% at
high N — PyBullet rendering is CPU-only despite EGL env vars being set. Peak λ≈407 at N=24.

### SimplerEnv WidowX (GPU, SAPIEN/Vulkan)

| N | λ (obs/s) | CPU% | GPU% | GPU_MEM | SYS_RAM | Bottleneck |
|--:|----------:|-----:|-----:|--------:|--------:|:----------:|
| 1 | 9.9 | 3 | 5 | 0.5GB | 17GB | — |
| 8 | 71.9 | 10 | 31 | 3.8GB | 24GB | — |
| 16 | 126.2 | 18 | 36 | 7.5GB | 31GB | GPU |
| 24 | 138.0 | 24 | 43 | 11GB | 39GB | GPU |

2 GPU: **worse** (138→123 at N=24). SAPIEN multi-GPU distribution adds overhead.
**Use 1 GPU.** Peak λ≈138 at N=24.

### RoboTwin 2.0 (GPU, SAPIEN)

| N | λ 1GPU | GPU% | λ 2GPU | GPU% | GPU_MEM (2GPU) |
|--:|-------:|-----:|-------:|-----:|---------------:|
| 1 | 2.1* | 69 | 2.2* | 69 | 3.6GB |
| 4 | 2.4* | 100 | 4.5* | 100 | 14GB |
| 8 | 2.5* | 100 | 4.7* | 100 | 29GB |
| 16 | 2.4* | 100 | 4.9* | 100 | 57GB |
| 24 | — | — | 4.8* | 100 | 86GB |

\* = timeout (all measurements hit 300s limit)

**GPU compute is the bottleneck.** 2 GPU gives ~2× improvement (2.4→4.9 obs/s).
Extremely slow rendering — each shard uses ~3.6GB VRAM. N=24 with 2 GPU uses 86GB
(near 160GB total). CPU is idle (3-25%). Peak λ≈4.9 at N=16 (2 GPU).

### Bottleneck Summary

| Benchmark | Bottleneck | GPU helps? | Peak λ (obs/s) | Rec. GPUs |
|-----------|:----------:|:----------:|:--------------:|:---------:|
| LIBERO | CPU | No | 415 (N=50) | 1 |
| CALVIN | CPU | No | 407 (N=24) | 1 |
| SimplerEnv | GPU | No (worse) | 138 (N=24) | 1 |
| RoboTwin | GPU | Yes (2×) | 4.9 (N=16) | 2 |

## Recommended Shard Counts (LIBERO, H100 supply)

Rule: `num_shards ≤ 0.8 × μ / per_shard_demand`. LIBERO per-shard ≈ 8.3 obs/s (N=50 avg).

| Model | μ (obs/s) | Max shards | Recommended | Est. wall time |
|-------|:---------:|:----------:|:-----------:|:--------------:|
| X-VLA | 88.8 | 8 | 10 | ~30 min |
| Pi0.5 | 84.0 | 8 | 10 | ~30 min |
| GR00T | 46.5 | 4 | 5 | ~55 min |
| StarVLA Q2.5-GR00T | 38.3 | 3 | 4 | ~70 min |
| OFT | 27.1 | 2 | 3 | ~90 min |
| StarVLA Q2.5-OFT | 6.0 | 0.6 | 1 | ~4.5 hrs |
| StarVLA Q3-OFT | 5.9 | 0.6 | 1 | ~4.5 hrs |
| StarVLA Q2.5-FAST | 1.4 | 0.1 | 1 | ~20 hrs |

## Reference Data

- [reported-performance.md](reported-performance.md) — Officially reported scores from papers/model cards.
- [../tuning-guide.md](../tuning-guide.md) — Demand/supply measurement methodology.

## Result Artifacts

Raw evaluation outputs (merged result JSONs) are archived under [`data/`](data/) after each run.
