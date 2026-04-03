# Skill: run-evaluation

Run a VLA model against a simulation benchmark to get evaluation results.

## Trigger

User asks to run/execute/launch an evaluation, benchmark a model, get benchmark results, or evaluate a model (e.g. "run OpenVLA on LIBERO", "evaluate CogACT on CALVIN", "benchmark my model").

## Steps

### 1. Identify the Config Pair

Every evaluation needs **two configs**: a model server config and a benchmark config. Ask the user (if not already provided):

- **Model server config** — which model to evaluate (from `configs/model_servers/`)
- **Benchmark config** — which benchmark to run (from `configs/`)

#### Available model server configs

List with: `ls configs/model_servers/`

Each file defines a `script` (the model server Python file) and `args` (model-specific flags). Example (`configs/model_servers/openvla.yaml`):
```yaml
script: "src/vla_eval/model_servers/openvla.py"
args:
  model_path: openvla/openvla-7b
  unnorm_key: bridge_orig
  chunk_size: 1
  port: 8000
```

#### Available benchmark configs

List with: `ls configs/*.yaml`

Each file defines `server.url`, `docker.image`, `output_dir`, and one or more `benchmarks` entries. Example (`configs/libero_spatial.yaml`):
```yaml
server:
  url: "ws://localhost:8000"
docker:
  image: ghcr.io/allenai/vla-evaluation-harness/libero:latest
output_dir: "./results"
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    mode: sync
    episodes_per_task: 50
    params:
      suite: libero_spatial
      seed: 7
      num_steps_wait: 10
```

**Model ↔ Benchmark pairing**: Not all models work with all benchmarks. The model server must produce actions in the format the benchmark expects (e.g. 7-DoF for LIBERO). Check the model server config comments and `unnorm_key` for guidance. Many model server configs encode the target benchmark in their filename (e.g. `oft_libero.yaml`, `xvla_calvin.yaml`).

### 2. Check Prerequisites

Before running, verify:

1. **uv** — Required for model server launch. Check: `which uv`
2. **Docker** — Required for benchmark execution (benchmarks run inside Docker containers). Check: `docker info`
3. **GPU** — Most model servers need NVIDIA GPU with CUDA. Benchmarks also need GPU for rendering. Check: `nvidia-smi`
4. **Model weights** — Downloaded automatically on first `vla-eval serve` (from HuggingFace Hub), but can be large (7B+ parameters). Ensure sufficient disk space.
5. **Docker image** — Benchmark images are large (4–10 GB). `vla-eval run` prompts to pull if missing. Pre-pull with: `docker pull ghcr.io/allenai/vla-evaluation-harness/<benchmark>:latest`
6. **Disk space** — Model weights (tens of GB) + Docker images (4–10 GB each) + results

### 3. Run the Evaluation (Two-Terminal Workflow)

Evaluations require **two concurrent processes**: the model server and the benchmark runner. They communicate via WebSocket.

#### Terminal 1: Start the model server

```bash
vla-eval serve --config configs/model_servers/<model>.yaml
```

This runs `uv run <script> --<arg1> <val1> ...` based on the YAML config. The server:
- Downloads model weights on first run (if from HuggingFace Hub)
- Loads the model onto GPU
- Starts a WebSocket server (default: `ws://0.0.0.0:8000`)
- Logs `"Model ready, starting server on ws://..."` when ready

**Wait for the "ready" log before starting the benchmark.**

Add `-v` for verbose/debug logging: `vla-eval serve -c configs/model_servers/<model>.yaml -v`

#### Terminal 2: Run the benchmark

```bash
vla-eval run --config configs/<benchmark>.yaml
```

This:
1. Pulls the Docker image if not present (prompts for confirmation; use `--yes` to skip)
2. Launches a Docker container with `--network host` (so it can reach the model server on localhost)
3. Inside the container: instantiates the benchmark, connects to the model server via WebSocket, runs all episodes
4. Saves results to `output_dir` (default: `./results/`)

Add `-v` for verbose logging: `vla-eval run -c configs/<benchmark>.yaml -v`

### 4. Parallel Evaluation with Sharding

A single-shard LIBERO Spatial run (10 tasks × 50 episodes = 500 episodes) can take **hours**. Sharding splits work across multiple Docker containers running in parallel, each handling a disjoint subset of episodes. Actual speedup depends on the benchmark, model server throughput, and hardware — production results show **~47× for LIBERO (H100, 50 shards)** and **~16× for CALVIN (H100, 16 shards)**. See [`docs/tuning-guide.md`](../../docs/tuning-guide.md) for how to measure demand/supply curves and derive optimal `num_shards`, `max_batch_size`, and `max_wait_time` for your setup.

```bash
# Launch 4 shards (each connects to the same model server)
vla-eval run -c configs/libero_spatial.yaml --shard-id 0 --num-shards 4 &
vla-eval run -c configs/libero_spatial.yaml --shard-id 1 --num-shards 4 &
vla-eval run -c configs/libero_spatial.yaml --shard-id 2 --num-shards 4 &
vla-eval run -c configs/libero_spatial.yaml --shard-id 3 --num-shards 4 &
wait
```

Sharding details:
- Work items (task × episode pairs) are distributed **round-robin** across shards — deterministic and reproducible
- Each shard writes a deterministic output file: `{name}_shard{id}of{total}.json`
- **GPU**: automatically assigned round-robin (shard 0 → GPU 0, shard 1 → GPU 1, …)
- **CPU**: cores are partitioned evenly across shards to prevent cross-container contention
- **OpenMP/MKL**: forced single-threaded per container (`OMP_NUM_THREADS=1`) to avoid thread explosion
- The model server handles concurrent WebSocket connections (one per shard) and can batch inference across them if `predict_batch()` is implemented

#### Resource allocation flags

Override Docker resource allocation:
```bash
vla-eval run -c config.yaml --gpus "0,1" --cpus "0-31"
```

Or set in the benchmark config under `docker:`:
```yaml
docker:
  image: ghcr.io/allenai/vla-evaluation-harness/libero:latest
  gpus: "0,1"
  cpus: "0-31"
```

### 5. Merge Shard Results

After sharded runs complete, merge into a single result file:

```bash
# Auto-discover shards from config
vla-eval merge -c configs/libero_spatial.yaml -o results/libero_spatial_merged.json

# Or specify files manually
vla-eval merge results/LIBEROBenchmark_libero_spatial_shard*of4.json -o results/merged.json
```

Missing shards are allowed — the merged result is marked partial.

### 6. Understand the Results

Results are saved as JSON in `output_dir`. Filename format: `{benchmark_name}_{tag}_{timestamp}.json` (or `{name}_shard{id}of{total}.json` for sharded runs). `tag` is `sync` or `realtime` (from `EvalConfig.mode`) on success, or `partial` if the run was interrupted (e.g. server disconnect).

Structure:
```json
{
  "benchmark": "LIBEROBenchmark",
  "mode": "sync",
  "harness_version": "0.1.0",
  "overall_success_rate": 0.72,
  "tasks": [
    {
      "task": "pick_up_the_black_bowl...",
      "success_rate": 0.80,
      "avg_steps": 145.2,
      "episodes": [
        {
          "episode_id": "task_name_ep0",
          "success": true,
          "steps": 130,
          "elapsed_sec": 12.34
        }
      ]
    }
  ],
  "config": { "..." },
  "server_info": { "..." }
}
```

Key metrics:
- **`overall_success_rate`** — primary metric, fraction of successful episodes across all tasks
- **Per-task `success_rate`** — success rate for each individual task
- **`avg_steps`** — average steps taken per episode (lower = more efficient)
- **`server_info`** — model server metadata from the HELLO handshake (model name, version, etc.)

### 7. Advanced Options

#### Run without Docker (development/debugging)

```bash
vla-eval run --config configs/<benchmark>.yaml --no-docker
```

Requires all benchmark dependencies installed locally. Useful when developing a new benchmark.

#### Dev mode (mount local source into container)

```bash
vla-eval run --config configs/<benchmark>.yaml --dev
```

Bind-mounts the local `src/` directory into the container, so code changes are reflected without rebuilding the Docker image.

#### Custom benchmark config overrides

Create or modify a config YAML to adjust:
- `episodes_per_task` — number of episodes per task (default varies by benchmark)
- `max_steps` — step limit per episode (omit to use benchmark default)
- `max_tasks` — cap on number of tasks to evaluate (useful for quick tests)
- `params.seed` — random seed for reproducibility
- `server.url` — model server address (default: `ws://localhost:8000`)
- `server.timeout` — seconds to wait for each action response (default: 30)

#### Real-time mode (for real-time control benchmarks)

Some benchmarks support real-time evaluation where the model must respond within a fixed time budget:
```yaml
benchmarks:
  - benchmark: "vla_eval.benchmarks.kinetix.benchmark:KinetixBenchmark"
    mode: realtime
    hz: 10.0
    hold_policy: repeat_last
    paced: true
```

- `hz` — target control frequency (default: 10.0)
- `hold_policy` — what to do when action is late: `"repeat_last"` (default) or `"zero"`
- `paced` — whether to pace to real-time (default: true)

### Quick Reference

```bash
# Full workflow: evaluate OpenVLA on LIBERO Spatial
# Terminal 1:
vla-eval serve -c configs/model_servers/openvla.yaml

# Terminal 2 (after server is ready):
vla-eval run -c configs/libero_spatial.yaml

# Sharded (4-way parallel):
for i in 0 1 2 3; do
  vla-eval run -c configs/libero_spatial.yaml --shard-id $i --num-shards 4 &
done
wait
vla-eval merge -c configs/libero_spatial.yaml -o results/libero_spatial.json

# Smoke test (no real model, validates the pipeline):
vla-eval test --all
```

### Parallel evaluations of different models

Shard result files are named by benchmark + shard ID (e.g.
`LIBEROBenchmark_libero_spatial_shard0of10.json`). If two evals use the
same benchmark config, shard count, and output directory, they will
collide. The orchestrator prevents this with a file lock — the second
eval will **fail immediately** with `FileExistsError` rather than
silently overwriting results.

If you hit this error, either:
- Use **different output directories** (modify `output_dir` in the config), or
- Use **different shard counts** (e.g. `--num-shards 10` vs `--num-shards 8`).

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `Docker daemon is not running` | Start Docker (requires admin privileges — ask your sysadmin on shared clusters) |
| `'uv' not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` (user-local, no sudo needed) |
| `Connection refused` on model server | Server not ready yet — wait for the "ready" log |
| `TimeoutError` during episodes | Increase `server.timeout` in config (default 30s) or check GPU utilization |
| Model server OOM | Reduce batch size or use a smaller model checkpoint |
| Docker image pull fails | Check network; or pre-pull: `docker pull <image>` |
| Partial results saved | Server disconnected mid-run — results up to that point are saved automatically |
| Mismatched action dimensions | Model server action output doesn't match benchmark expectation — check `unnorm_key` and `chunk_size` |
