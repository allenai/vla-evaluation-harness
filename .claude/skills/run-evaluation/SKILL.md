---
name: run-evaluation
description: "Run a VLA model evaluation against a simulation benchmark. Use this skill whenever the user wants to evaluate, benchmark, test, or run a model on a sim environment — even if they say it casually like 'try OpenVLA on LIBERO' or 'get me CALVIN scores'. Covers the full workflow: serving the model, launching the benchmark, sharding for speed, merging results, and interpreting output."
---

# Run Evaluation

Evaluate a VLA model against a simulation benchmark. The harness decouples model serving (WebSocket server) from benchmark execution (Docker container), so they run as two separate processes.

## 1. Identify the config pair

Every evaluation needs **two YAML configs**:

- **Model server config** (`configs/model_servers/<model>.yaml`) — defines `script` and `args` for the model server
- **Benchmark config** (`configs/<benchmark>.yaml`) — defines `docker.image`, `benchmarks` entries, and `output_dir`

List available configs:
```bash
ls configs/model_servers/    # model servers
ls configs/*.yaml            # benchmarks
```

Not all model–benchmark pairs are compatible. The model server must produce actions in the format the benchmark expects (e.g. 7-DoF for LIBERO). Many model configs encode their target benchmark in the filename (e.g. `oft_libero.yaml`, `xvla_calvin.yaml`).

## 2. Check prerequisites

| Requirement | Check command | Notes |
|---|---|---|
| uv | `which uv` | Runs model server in isolated env |
| Docker | `docker info` | Benchmarks run inside containers |
| GPU | `nvidia-smi` | Model inference + sim rendering |
| Disk space | `df -h` | Model weights (tens of GB) + Docker images (4–10 GB each) |

Model weights download automatically on first `vla-eval serve`. Docker images are pulled on first `vla-eval run` (or pre-pull with `docker pull <image>`).

## 3. Run the evaluation (two terminals)

The model server and benchmark runner communicate over WebSocket and must run concurrently.

**Terminal 1 — start the model server:**
```bash
vla-eval serve -c configs/model_servers/<model>.yaml
```
Wait for the `"Model ready, starting server on ws://..."` log before proceeding.

**Terminal 2 — run the benchmark:**
```bash
vla-eval run -c configs/<benchmark>.yaml
```

This pulls the Docker image if needed, launches the container with `--network host`, runs all episodes, and saves results to `output_dir` (default `./results/`).

Add `-v` to either command for debug logging.

## 4. Parallel sharding

Single-shard runs can take hours. Sharding splits episodes across multiple Docker containers that all connect to the same model server.

```bash
# Example: 4-way parallel
for i in 0 1 2 3; do
  vla-eval run -c configs/<benchmark>.yaml --shard-id $i --num-shards 4 &
done
wait
```

Sharding details:
- Work items distributed **round-robin** (deterministic, reproducible)
- Each shard writes `{name}_shard{id}of{total}.json`
- GPU assigned round-robin (shard 0 → GPU 0, shard 1 → GPU 1, …)
- CPU cores partitioned evenly; `OMP_NUM_THREADS=1` per container

Override resource allocation:
```bash
vla-eval run -c config.yaml --gpus "0,1" --cpus "0-31"
```

See `docs/tuning-guide.md` for how to derive optimal `num_shards`, `max_batch_size`, and `max_wait_time`.

## 5. Merge shard results

```bash
vla-eval merge -c configs/<benchmark>.yaml -o results/merged.json
# or manually:
vla-eval merge results/*_shard*of4.json -o results/merged.json
```

Missing shards are allowed — the merged result is marked partial.

## 6. Understand results

Results are JSON in `output_dir`. Structure:
```json
{
  "benchmark": "LIBEROBenchmark",
  "overall_success_rate": 0.72,
  "tasks": [
    {
      "task": "pick_up_the_black_bowl...",
      "success_rate": 0.80,
      "avg_steps": 145.2,
      "episodes": [{"episode_id": "...", "success": true, "steps": 130, "elapsed_sec": 12.34}]
    }
  ]
}
```

Key metrics:
- **`overall_success_rate`** — primary metric (fraction of successful episodes)
- **Per-task `success_rate`** — breakdown by task
- **`avg_steps`** — efficiency (lower = better)

## 7. Advanced options

| Option | Command | Purpose |
|---|---|---|
| No Docker | `--no-docker` | Dev/debug, requires local benchmark deps |
| Dev mode | `--dev` | Bind-mounts local `src/` into container |
| Real-time mode | Set `mode: realtime` in config | For control benchmarks (Kinetix) |
| Skip Docker prompt | `--yes` | Non-interactive image pull |
| Custom overrides | Edit config YAML | `episodes_per_task`, `max_steps`, `max_tasks`, `params.seed`, `server.timeout` |

## Parallel evaluations of different models

Shard result files are named by benchmark + shard count. If two evals share the same benchmark config, shard count, and output directory, a file lock prevents silent overwrites. Use different `output_dir` values or different shard counts to avoid collisions.

## Troubleshooting

| Problem | Fix |
|---|---|
| Docker daemon not running | Start Docker (may need sysadmin on shared clusters) |
| `Connection refused` | Server not ready — wait for the "ready" log |
| `TimeoutError` | Increase `server.timeout` in config or check GPU utilization |
| OOM | Reduce batch size or use smaller checkpoint |
| Mismatched action dims | Check `unnorm_key` and `chunk_size` in model server config |
| Partial results | Server disconnected — results up to that point saved automatically |
