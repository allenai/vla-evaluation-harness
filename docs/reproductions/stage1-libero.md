# Stage 1 — LIBERO Reproduction

Protocol: 4 suites × 10 tasks × 50 episodes = 2000 episodes/model, seed=7

## Models

| # | Name | Config | Port | Reported |
|---|------|--------|:----:|:--------:|
| 1 | X-VLA | `configs/model_servers/xvla/libero.yaml` | 8001 | 98.1% |
| 2 | Pi0.5 | `configs/model_servers/pi0/libero.yaml` | 8002 | 96.9% |
| 3 | OFT-joint | `configs/model_servers/oft/libero_joint.yaml` | 8003 | ~96.8% |
| 4 | StarVLA Q2.5-FAST | `configs/model_servers/starvla/libero_qwen25_fast.yaml` | 8004 | 95.2% |
| 5 | StarVLA Q2.5-OFT | `configs/model_servers/starvla/libero_qwen25_oft.yaml` | 8005 | 96.1% |
| 6 | StarVLA Q2.5-GR00T | `configs/model_servers/starvla/libero_qwen25_groot.yaml` | 8006 | 95.4% |
| 7 | StarVLA Q3-OFT | `configs/model_servers/starvla/libero_qwen3_oft.yaml` | 8007 | 96.6% |
| 8 | GR00T N1.6 | `configs/model_servers/groot/libero.yaml` | 8008 | 97.0% |

Not included: OpenVLA base (too slow), StarVLA Qwen3-PI (state_dict mismatch).

## Step 1: Build Docker

```bash
docker/build.sh libero
```

## Step 2: Start model servers (slurm)

One job per model. All on h100 partition, unique ports.

```bash
cd /mnt/harbor/users/claude/GitHub/vla-evaluation-harness-allenai

# All 8 at once
sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-xvla \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --address 0.0.0.0:8001 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-pi05 \
  --wrap="uv run vla-eval serve -c configs/model_servers/pi0/libero.yaml --address 0.0.0.0:8002 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-oft \
  --wrap="uv run vla-eval serve -c configs/model_servers/oft/libero_joint.yaml --address 0.0.0.0:8003 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-sv-q25fast \
  --wrap="uv run vla-eval serve -c configs/model_servers/starvla/libero_qwen25_fast.yaml --address 0.0.0.0:8004 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-sv-q25oft \
  --wrap="uv run vla-eval serve -c configs/model_servers/starvla/libero_qwen25_oft.yaml --address 0.0.0.0:8005 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-sv-q25groot \
  --wrap="uv run vla-eval serve -c configs/model_servers/starvla/libero_qwen25_groot.yaml --address 0.0.0.0:8006 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-sv-q3oft \
  --wrap="uv run vla-eval serve -c configs/model_servers/starvla/libero_qwen3_oft.yaml --address 0.0.0.0:8007 -v"

sbatch -p h100 --gres=gpu:1 -c8 --mem=64G -t 8:00:00 -J s1-groot \
  --wrap="uv run vla-eval serve -c configs/model_servers/groot/libero.yaml --address 0.0.0.0:8008 -v"
```

Check which node they landed on:

```bash
squeue -u $USER --format="%.10i %.20j %.4t %.20R" | grep s1-
```

Wait for all servers to be ready:

```bash
NODE=DGX-H100-10  # replace with actual node
for port in 8001 8002 8003 8004 8005 8006 8007 8008; do
  curl -s --max-time 2 "http://${NODE}:${port}/config" >/dev/null && echo "✓ ${port}" || echo "✗ ${port}"
done
```

OFT may take 30+ min (TensorFlow CUDA JIT on H100).

## Step 3: Run benchmarks (local, one model at a time)

Max 50 shards. Replace `$NODE` with the slurm node from step 2.

```bash
# Template — run for each model
MODEL=xvla  PORT=8001  # change per model
for i in $(seq 0 49); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://${NODE}:${PORT} \
    --shard-id $i --num-shards 50 --yes &
done
wait
```

Monitor early results while running:

```bash
# Check first few completed shards
for f in results/LIBEROBenchmark_libero_spatial_shard*of50.json; do
  python3 -c "import json; d=json.load(open('$f')); print(d.get('mean_success',0))" 2>/dev/null
done
```

## Step 4: Merge and archive

```bash
MODEL=xvla  # change per model
mkdir -p docs/reproductions/data/${MODEL}-libero

for suite in libero_spatial libero_object libero_goal libero_10; do
  uv run vla-eval merge results/LIBEROBenchmark_${suite}_shard*of50.json \
    -o docs/reproductions/data/${MODEL}-libero/merged_${suite}.json
done

# Clean shard files after merge
rm results/LIBEROBenchmark_*shard*of50.json
```

## Step 5: Check results

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('docs/reproductions/data/*/merged_*.json')):
    d = json.load(open(f))
    print(f'{f}: {d.get(\"mean_success\", 0):.1%}')
"
```

## Notes

- Benchmark params (`send_wrist_image`, `send_state`, `absolute_action`) are auto-negotiated via HELLO. No `--param` needed.
- OFT-joint requires per-suite `unnorm_key`. Current config uses `libero_spatial_no_noops`. For full 4-suite eval, run 4 server instances with different unnorm_keys, or run suites separately.
- After cancelling model servers: `squeue -u $USER | grep s1- | awk '{print $1}' | xargs scancel`
