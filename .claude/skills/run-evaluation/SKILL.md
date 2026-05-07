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

**Docker image rebuild**: Benchmark code runs *inside* the Docker image. If you (or someone else) changed benchmark source code in `src/vla_eval/benchmarks/`, the pre-built image is stale — you must rebuild before running:
```bash
./docker/build.sh <benchmark_name>   # e.g. ./docker/build.sh libero
```
Skip the rebuild only if using `--dev` mode, which bind-mounts local `src/` into the container.

## 3. Run the evaluation (two terminals)

The model server and benchmark runner communicate over WebSocket and must run concurrently.

**Terminal 1 — start the model server:**
```bash
vla-eval serve -c configs/model_servers/<model>.yaml
```
Wait for the `"Model ready, starting server on ws://..."` log before proceeding.

**Remote serving via slurm** (when model needs GPUs on a different node):
```bash
srun --gres=gpu:1 --mem=32G --job-name=model-serve \
  bash -c "uv run vla-eval serve -c configs/model_servers/<model>.yaml" &
```
Check the allocated node with `squeue`, verify with `curl -s http://<node>:8000/config`, then use `--server-url ws://<node>:8000` for benchmark runs. Cancel with `scancel` when done.

**Terminal 2 — run the benchmark:**
```bash
vla-eval run -c configs/<benchmark>.yaml
```

When the model server is on a remote node, use `--server-url` to override:
```bash
vla-eval run -c configs/<benchmark>.yaml --server-url ws://<slurm-node>:8000
```

This pulls the Docker image if needed, launches the container with `--network host`, runs all episodes, and saves results to `output_dir` (default `./results/`).

**`--dev` mode**: If you changed code in `src/` since the Docker image was last built, add `--dev` to bind-mount local source into the container. Without it, the container runs stale code.

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
  "benchmark": "LIBEROBenchmark_libero_spatial",
  "mean_success": 0.968,
  "tasks": [
    {
      "task": "pick_up_the_black_bowl...",
      "mean_success": 0.96,
      "num_episodes": 50,
      "avg_steps": 95.2,
      "episodes": [
        {"episode_id": 0, "metrics": {"success": true}, "steps": 78, "elapsed_sec": 12.34},
        {"episode_id": 1, "metrics": {"success": false}, "steps": 220, "failure_reason": "timeout", "failure_detail": "..."}
      ]
    }
  ]
}
```

Key metrics:
- **`mean_success`** — primary metric (fraction of successful episodes, all episodes count)
- **Per-task `mean_success`** — breakdown by task
- **`avg_steps`** — efficiency (lower = better)
- **`num_errors`** — present on tasks that had episodes with `failure_reason` (connection errors, exceptions, etc.)
- **`failure_reason` / `failure_detail`** — per-episode diagnostic fields for debugging failures

## 7. Advanced options

| Option | Command | Purpose |
|---|---|---|
| No Docker | `--no-docker` | Dev/debug, requires local benchmark deps |
| Dev mode | `--dev` | Bind-mounts local `src/` into container (no rebuild needed) |
| Real-time mode | Set `mode: realtime` in config | For control benchmarks (Kinetix) |
| Skip Docker prompt | `--yes` | Non-interactive image pull |
| Custom overrides | Edit config YAML | `episodes_per_task`, `max_steps`, `max_tasks`, `params.seed`, `server.timeout` |

## Custom output directory

Override `output_dir` with `--output-dir`:
```bash
vla-eval run -c configs/benchmarks/libero/spatial.yaml --output-dir results/my-experiment/
```
Default is `./results/` (from config YAML). The CLI flag takes precedence over the config value.

## Parallel evaluations of different models

Shard result files are named by benchmark + shard count. If two evals share the same benchmark config, shard count, and output directory, a file lock prevents silent overwrites. Use different `output_dir` values or different shard counts to avoid collisions.

## Monitoring shard progress

Each shard writes a `.progress` file that updates after every episode. Use `watch` for a live dashboard:
```bash
watch -n 2 'for f in results/*.progress; do echo "$(basename $f .progress): $(cat $f)"; done; echo "---"; echo "Done: $(ls results/*shard*of*.json 2>/dev/null | wc -l) shards"'
```
Progress files are removed automatically when the shard finishes and writes its result JSON. Lock files are also cleaned up on completion.

## Troubleshooting

| Problem | Fix |
|---|---|
| Docker daemon not running | Start Docker (may need sysadmin on shared clusters) |
| `Connection refused` | Server not ready — wait for the "ready" log |
| `TimeoutError` | Increase `server.timeout` in config or check GPU utilization |
| OOM | Reduce batch size or use smaller checkpoint |
| Mismatched action dims | Check `unnorm_key` and `chunk_size` in model server config |
| Partial results | Server disconnected — results up to that point saved automatically |
