# Reproduction Roadmap

Systematic verification that vla-eval reproduces published model scores.

## Stages

### Stage 1 — Breadth: many models × LIBERO

| Model | Config | Reported | Status |
|-------|--------|:--------:|--------|
| X-VLA (0.9B) | `xvla/libero.yaml` | 98.1% | Ready |
| Pi0.5 | `pi0/libero.yaml` | 96.9% | Ready |
| OpenVLA-OFT (joint) | `oft/libero_joint.yaml` | ~96.8% | Ready |
| StarVLA Q2.5-FAST | `starvla/libero_qwen25_fast.yaml` | 95.2% | Ready (slow — see supply) |
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

All configs under `configs/model_servers/`. Each prefixed by benchmark name (e.g. `xvla/libero.yaml`, `xvla/calvin.yaml`).

---

## Supply — Model Server Throughput (H100-80GB SXM)

Measured with `experiments/bench_supply.py`, `max_batch_size=1` (no batching), 256×256 images.

| Model | chunk_size | μ (obs/s) | Inference latency | GPU inf/s | Notes |
|-------|:---------:|:---------:|:-----------------:|:---------:|-------|
| X-VLA | 30 | 88.8 | 30ms | 3.0 | |
| Pi0.5 | 10 | 84.0 | 63ms | 8.4 | |
| GR00T N1.6 | 16 | 46.5 | 50ms | 2.9 | |
| StarVLA Q2.5-GR00T | 1 | 38.3 | 60ms | 38.3 | |
| OFT (joint) | 10 | 27.1 | 46ms | 2.7 | |
| StarVLA Q2.5-OFT | 1 | 6.0 | 654ms | 6.0 | |
| StarVLA Q3-OFT | 1 | 5.9 | 664ms | 5.9 | |
| StarVLA Q2.5-FAST | 1 | 1.4 | 2858ms | 1.4 | Autoregressive token gen |

StarVLA/GR00T support `predict_batch()` — sweeping `max_batch_size` may improve throughput.
X-VLA/Pi0/OFT are single-predict only (`max_batch_size=1`).

## Demand — Benchmark Observation Rate (A100, from tuning-guide)

| Benchmark | Rendering | Per-shard obs/s | Peak λ (N) | Notes |
|-----------|-----------|:---------------:|:----------:|-------|
| LIBERO | GPU EGL (MuJoCo) | ~7.3 | 447 (N=80) | Lightweight EGL, scales well |
| CALVIN | GPU EGL (PyBullet) | ~36.7 | 432 (N=24) | Fast physics, saturates early |
| SimplerEnv | GPU (SAPIEN/Vulkan) | ~10.1 | 144 (N=24) | Heavy GPU, contention limits scaling |
| RoboTwin | GPU | TBD | TBD | Needs measurement |

## Recommended Shard Counts (LIBERO, H100)

Rule: `num_shards ≤ 0.8 × μ / per_shard_demand`. LIBERO per-shard = ~7.3 obs/s.

| Model | μ (obs/s) | Max shards (80% rule) | Recommended | Est. wall time |
|-------|:---------:|:---------------------:|:-----------:|:--------------:|
| X-VLA | 88.8 | 9 | 10 | ~30 min |
| Pi0.5 | 84.0 | 9 | 10 | ~30 min |
| GR00T | 46.5 | 5 | 5 | ~55 min |
| StarVLA Q2.5-GR00T | 38.3 | 4 | 4 | ~70 min |
| OFT | 27.1 | 3 | 4 | ~70 min |
| StarVLA Q2.5-OFT | 6.0 | 0.6 | 1 | ~4.5 hrs |
| StarVLA Q3-OFT | 5.9 | 0.6 | 1 | ~4.5 hrs |
| StarVLA Q2.5-FAST | 1.4 | 0.15 | 1 | ~20 hrs |

Models with μ < 7.3 obs/s cannot even keep up with 1 shard — the single shard generates
observations faster than the server processes them. Runs will still work (queue absorbs bursts)
but wall time is bottlenecked by inference speed, not parallelism.

## Reference Data

- [reported-performance.md](reported-performance.md) — Officially reported scores from papers/model cards.
- [../tuning-guide.md](../tuning-guide.md) — Demand/supply measurement methodology and worked examples.

## Result Artifacts

Raw evaluation outputs (merged result JSONs) are archived under [`data/`](data/) after each run.
