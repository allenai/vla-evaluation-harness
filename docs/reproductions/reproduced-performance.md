# Reproduced Performance

> Measured with vla-eval. Model servers on DGX-H100-80GB (slurm), benchmarks on A100-Zebra (Docker).
> Protocol: 50 episodes/task, seed=7, num_steps_wait=10.

## Stage 1 — LIBERO (4 suites)

| Model | Spatial | Object | Goal | 10 | **Average** | Reported | Verdict |
|-------|:-------:|:------:|:----:|:--:|:-----------:|:--------:|:-------:|
| X-VLA (0.9B) | 98.0% | 98.0% | 98.0% | 94.8% | **97.2%** | 98.1% | Reproduced |
| Pi0.5 | 98.0% | 99.6% | 98.6% | 94.6% | **97.7%** | 96.9% | Reproduced |
| GR00T N1.6 | 96.6% | 98.4% | 96.8% | 87.8% | **94.9%** | 97.0% | Reproduced (−2pp) |
| OFT (per-suite) | 92.2% | — | — | — | **—** | ~96.8% | In progress |

Not evaluated: OpenVLA base (too slow), StarVLA variants (too slow, chunk_size=1),
StarVLA Qwen3-PI (state_dict mismatch).

## Stage 2 — Cross-benchmark

| Model | LIBERO | CALVIN | SimplerEnv | RoboTwin |
|-------|:------:|:------:|:----------:|:--------:|
| DB-CogACT | 95.2% | 4.05 avg len | 72.2% | — |

DB-CogACT details: [db-cogact.md](db-cogact.md)

---

## Supply — Model Server Throughput (H100-80GB SXM)

Measured with `experiments/bench_supply.py`, `max_batch_size=1`, 256×256 images.

| Model | chunk_size | μ (obs/s) | Inference latency |
|-------|:---------:|:---------:|:-----------------:|
| X-VLA | 30 | 88.8 | 30ms |
| Pi0.5 | 10 | 84.0 | 63ms |
| GR00T N1.6 | 16 | 46.5 | 50ms |
| StarVLA Q2.5-GR00T | 1 | 38.3 | 60ms |
| OFT (joint) | 10 | 27.1 | 46ms |
| StarVLA Q2.5-OFT | 1 | 6.0 | 654ms |
| StarVLA Q3-OFT | 1 | 5.9 | 664ms |
| StarVLA Q2.5-FAST | 1 | 1.4 | 2858ms |

## Demand — Benchmark Observation Rate (A100-Zebra)

96 CPU cores, 2× A100-80GB PCIe, 503GB RAM. Median utilization during steady-state.

| Benchmark | Peak λ (N) | Bottleneck | 2 GPU helps? |
|-----------|:----------:|:----------:|:------------:|
| LIBERO | 415 obs/s (N=50) | CPU | No |
| CALVIN | 407 obs/s (N=24) | CPU | No |
| SimplerEnv | 138 obs/s (N=24) | GPU | No (worse) |
| RoboTwin | 4.9 obs/s (N=16) | GPU | Yes (2×) |

## How to Run

```bash
# 1. Build Docker
docker/build.sh libero

# 2. Start model server (slurm)
sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 24:00:00 \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --address 0.0.0.0:8001 -v"

# 3. Run benchmark (one model at a time, merge before next)
NODE=DGX-H100-12  SHARDS=10
for i in $(seq 0 $((SHARDS-1))); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://${NODE}:8001 \
    --shard-id $i --num-shards $SHARDS --yes &
done
wait

# 4. Merge
mkdir -p docs/reproductions/data/xvla-libero
for suite in libero_spatial libero_object libero_goal libero_10; do
  uv run vla-eval merge results/LIBEROBenchmark_${suite}_shard*of${SHARDS}.json \
    -o docs/reproductions/data/xvla-libero/merged_${suite}.json
done

# 5. Clean shard files before next model
rm results/LIBEROBenchmark_*shard*.json
```

## Reference

- [reported-performance.md](reported-performance.md) — Officially reported scores from papers/model cards.
