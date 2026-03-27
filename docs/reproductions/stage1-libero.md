# Stage 1 — LIBERO Reproduction

Protocol: 4 suites × 10 tasks × 50 episodes = 2000 episodes/model, seed=7

## Models

| # | Name | Config | Port | Shards | Reported |
|---|------|--------|:----:|:------:|:--------:|
| 1 | X-VLA | `xvla/libero.yaml` | 8001 | 10 | 98.1% |
| 2 | Pi0.5 | `pi0/libero.yaml` | 8002 | 10 | 96.9% |
| 3 | GR00T N1.6 | `groot/libero.yaml` | 8003 | 5 | 97.0% |
| 4 | StarVLA Q2.5-GR00T | `starvla/libero_qwen25_groot.yaml` | 8004 | 4 | 95.4% |
| 5 | OFT-spatial | `oft/libero_spatial.yaml` | 8005 | 3 | 97.1% |
| 6 | OFT-object | `oft/libero_object.yaml` | 8006 | 3 | — |
| 7 | OFT-goal | `oft/libero_goal.yaml` | 8007 | 3 | — |
| 8 | OFT-10 | `oft/libero_10.yaml` | 8008 | 3 | — |
| 9 | StarVLA Q2.5-OFT | `starvla/libero_qwen25_oft.yaml` | — | 1 | 96.1% |
| 10 | StarVLA Q3-OFT | `starvla/libero_qwen3_oft.yaml` | — | 1 | 96.6% |
| 11 | StarVLA Q2.5-FAST | `starvla/libero_qwen25_fast.yaml` | — | 1 | 95.2% |

Not included: OpenVLA base (too slow), StarVLA Qwen3-PI (state_dict mismatch).

Shard counts from supply/demand analysis (see README.md).

## Step 1: Build Docker

```bash
docker/build.sh libero
```

## Step 2: Start model servers (slurm, max 8 at a time)

```bash
cd /mnt/harbor/users/claude/GitHub/vla-evaluation-harness-allenai

# Launch up to 8 at a time (1 GPU each, same node)
sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 24:00:00 -J s1-xvla \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --address 0.0.0.0:8001 -v"

# ... (see Models table for configs and ports)
```

Wait for servers:
```bash
NODE=DGX-H100-12  # replace with actual node
for port in 8001 8002 8003 8004; do
  curl -s --max-time 2 "http://${NODE}:${port}/config" >/dev/null && echo "✓ ${port}" || echo "✗ ${port}"
done
```

## Step 3: Run benchmarks (ONE MODEL AT A TIME)

**IMPORTANT**: Run models sequentially. Different models with the same `num_shards`
produce identical shard filenames and overwrite each other.

Max 50 Docker containers total on the benchmark host.

```bash
NODE=DGX-H100-12  # replace with actual node

# --- Model 1: X-VLA (10 shards) ---
SHARDS=10  PORT=8001  MODEL=xvla
for i in $(seq 0 $((SHARDS-1))); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://${NODE}:${PORT} \
    --shard-id $i --num-shards $SHARDS --yes &
done
wait

# Merge and archive immediately
mkdir -p docs/reproductions/data/${MODEL}-libero
for suite in libero_spatial libero_object libero_goal libero_10; do
  uv run vla-eval merge results/LIBEROBenchmark_${suite}_shard*of${SHARDS}.json \
    -o docs/reproductions/data/${MODEL}-libero/merged_${suite}.json
done
rm results/LIBEROBenchmark_*shard*of${SHARDS}.json

# --- Model 2: Pi0.5 (10 shards) ---
SHARDS=10  PORT=8002  MODEL=pi05
# ... same pattern ...

# --- Model 3: GR00T (5 shards) ---
SHARDS=5  PORT=8003  MODEL=groot
# ... same pattern ...
```

## Step 4: Check results

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('docs/reproductions/data/*/merged_*.json')):
    d = json.load(open(f))
    print(f'{f}: {d.get(\"mean_success\", 0):.1%}')
"
```

## Notes

- **One model at a time**: Merge and clean shard files before starting the next model.
- Benchmark params are auto-negotiated via HELLO. No `--param` needed.
- OFT requires per-suite server instances (different `unnorm_key` per suite).
- Slow models (StarVLA OFT/FAST, 1 shard): run last, takes 4-20 hours each.
- Cancel servers: `squeue -u $USER | grep s1- | awk '{print $1}' | xargs scancel`
