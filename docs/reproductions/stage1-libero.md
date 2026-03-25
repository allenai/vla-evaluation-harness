# Stage 1 — LIBERO Reproducibility Audit

> Protocol: 50 episodes/task, seed=7, num_steps_wait=10
> Hardware: Model servers on DGX-H100 (80GB), benchmarks on A100 nodes

## Status

All models pending re-evaluation after code audit and bug fixes.

| Model | Reported | Reproduced | Status |
|---|:---:|:---:|---|
| X-VLA (0.9B) | 98.1% | — | Pending |
| Pi0.5 | 96.9% | — | Pending |
| OpenVLA-OFT (joint) | ~96.8% | — | Pending |
| StarVLA (5 variants) | 95-97% | — | Pending (was 0%, fixed: wrist image) |
| GR00T N1.6 (community) | 97.0% | — | Pending (was ~1%, fixed: gripper inversion) |
| OpenVLA base (LoRA) | 76.5% | — | Low priority (slow without batch prediction) |

## Fixes Applied

### Code fixes (committed)

| Model | Bug | Fix |
|---|---|---|
| GR00T | Gripper action polarity inverted | Added `invert_gripper` flag in `groot.py` |
| StarVLA | Wrist image not sent to model | HELLO auto-negotiation sends `send_wrist_image=true` |
| Pi0 | Default config was `pi0_fast_libero` (wrong model) | Changed to `pi05_libero` |
| OFT | `num_images_in_input: 1` (should be 2) | Fixed in `_base.yaml` |

### Infrastructure improvements

- **HELLO observation negotiation**: Model servers auto-declare required benchmark params (`send_wrist_image`, `send_state`, `absolute_action`) — no manual `--param` flags needed.
- **`--server-url`**: Override server URL at CLI without per-host config files.
- **`--port` / `--host`**: Override model server port/host at CLI.
- **`--param KEY=VALUE`**: Manual benchmark param override for experimentation.

## Known Issues

- **StarVLA Qwen3-PI**: Checkpoint `StarVLA/Qwen3-VL-PI-LIBERO-4in1` has 36 DiT transformer blocks but the model code expects 16. Requires `num_layers` compatibility fix in `starvla.py`.

## How to Run

```bash
# 1. Build Docker image
docker/build.sh libero

# 2. Start model server (slurm)
sbatch --partition=h100 --gres=gpu:1 --mem=64G --time=6:00:00 \
  --wrap="cd $PWD && uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --port 8001 -v"

# 3. Run benchmark (50 shards, local)
for i in $(seq 0 49); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://DGX-H100-XX:8001 \
    --shard-id $i --num-shards 50 --yes &
done
wait

# 4. Merge results
uv run vla-eval merge -c configs/libero_all.yaml -o results/merged.json
```
