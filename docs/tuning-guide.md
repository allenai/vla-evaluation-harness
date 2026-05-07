# Tuning Guide: num_shards, max_batch_size, max_wait_time

## Overview

Sharding + batching has three parameters:

| Parameter | Where | What it controls |
|-----------|-------|-----------------|
| `num_shards` | CLI `--num-shards` | Number of parallel environment processes |
| `max_batch_size` | `BatchPredictModelServer` | Max observations per GPU forward pass |
| `max_wait_time` | `BatchPredictModelServer` | Seconds to wait before dispatching a partial batch |

These interact: the optimal value of each depends on the others. Two benchmark scripts measure **demand** (environment side) and **supply** (model server side) independently.

## Resource Allocation

By default, `vla-eval` assumes the benchmark host is fully dedicated to the evaluation — all CPUs and GPUs are available for shard containers. If you share the machine with other workloads, use `--cpus` and `--gpus` to restrict resources (see below).

When running sharded evaluations, each Docker container receives isolated CPU and GPU resources automatically via [`docker_resources.py`](../src/vla_eval/docker_resources.py):

| Resource | Non-sharded | Sharded (N containers) |
|----------|-------------|------------------------|
| **GPU** | `--gpus all` | Round-robin across available devices |
| **CPU** | All host CPUs | Partitioned evenly (e.g. 48 cores / 8 shards = 6 cores each) |
| **Threads** | Default | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |

Thread pinning (`OMP_NUM_THREADS=1`) prevents cross-container contention — without it, each shard spawns one OpenMP thread per visible core, causing massive context-switch overhead (e.g. 8 shards × 48 threads = 384 threads on 48 cores).

Override via CLI flags:

```bash
vla-eval run -c configs/benchmarks/libero/spatial.yaml \
    --num-shards 16 \
    --gpus 0,1 \        # only use GPU 0 and 1 for rendering
    --cpus 0-31          # only use cores 0-31
```

Or in the benchmark config YAML under `docker:`:

```yaml
docker:
  gpus: "0,1"
  cpus: "0-31"
```

## Step 1: Measure Demand — λ(N)

> "With N shards, how many inference requests/sec does the environment generate?"

```bash
uv run python experiments/bench_demand.py \
    --config configs/benchmarks/libero/spatial.yaml \
    --chunk-size 12 \
    --shards 1,2,4,8
```

Starts an instant-response server and launches N real Docker shards against it. Counts `on_observation()` calls to measure actual observation rate (including chunk-buffered steps).

**What to look for**: λ(N) grows linearly at first, then flattens at high N due to host contention.

## Step 2: Measure Supply — μ(B)

> "How many requests/sec can the model server process under saturation?"

```bash
# Server must already be running
# Sweep max_batch_size at runtime (no restart needed)
uv run python experiments/bench_supply.py \
    --url ws://GPU-NODE:8000 \
    --sweep-batch-sizes 1,2,4,8,16 \
    --num-clients 32 \
    --requests-per-client 50 \
    --image-size 256
```

Floods the running model server with saturating clients while sweeping `max_batch_size` via the server's HTTP control plane (`GET /config?max_batch_size=B`). Use `--image-size` to match your benchmark's observation size. Use enough `--num-clients` to keep the server saturated at all batch sizes.

If the server uses chunk buffering (chunk_size > 1), only 1 out of every chunk_size requests triggers GPU inference. Set `--requests-per-client` high enough that each client triggers many inferences (e.g. ≥ chunk_size × 15). Too few requests yields noisy, unreliable throughput numbers.

**What to look for**: μ grows with B but eventually plateaus (or even degrades due to padding overhead). The optimal `max_batch_size` is at the knee — not necessarily the maximum.

## Step 3: Combine the Curves

Read the tables from Step 1 and Step 2.

**Rule of thumb: always keep λ(N) < μ(B\*).** If demand exceeds supply, the server's request queue grows without bound — latency climbs over time, clients eventually hit their response timeout, and you get connection drops, lost episodes, and cascading failures. The correct approach is to maximize demand (more shards = faster wall-clock) while staying strictly below the supply ceiling.

1. **`max_batch_size = B*`**: Find the optimal B that maximizes μ — pick B at the knee where μ(B) starts to plateau. Larger B does not always help (padding overhead, memory pressure). This sets the supply ceiling.

2. **`num_shards = N*`**: Maximize λ while keeping λ(N) < ~80% of μ(B*) — pick the largest N that stays well below the supply ceiling.
   - A ~20% margin absorbs variance in per-step timing, batch fill fluctuations, and GC pauses. Without margin, λ ≈ μ puts you right at the edge where transient spikes cause queue buildup.
   - More shards = faster wall-clock, but you must not cross μ(B*).
   - Also watch for host contention: beyond a certain N, per-shard throughput degrades even if the aggregate λ stays below μ.

3. **`max_wait_time`**: Minimize batch collection latency at the chosen operating point.
   - Formula: `max_wait_time ≈ B* / predict_rate`, where `predict_rate = λ(N*) / chunk_size`
   - When chunk_size = 1 (every observation triggers a predict), this simplifies to `B* / λ(N*)`.
   - This is a starting point. Slightly higher values improve batch fill rate; slightly lower values reduce per-request latency.
   - For `PredictModelServer` (no batching), this parameter does not apply.

## Quick Reference

```
max_batch_size  = optimal B at the knee of μ(B)
num_shards      = max N where λ(N) < 0.8 × μ(max_batch_size)
predict_rate    = λ(num_shards) / chunk_size        # chunk_size=1 if no chunking
max_wait_time   ≈ max_batch_size / predict_rate
```

## Launching a Model Server

The `vla-eval serve` command launches a model server from a YAML config:

```bash
vla-eval serve -c configs/model_servers/dexbotic_cogact_libero.yaml -v
```

The config YAML specifies the server script and its arguments:

```yaml
script: "src/vla_eval/model_servers/dexbotic/cogact.py"
args:
  model_path: Dexmal/libero-db-cogact
  chunk_size_map: '{"libero_spatial": 12, "libero_object": 16, ...}'
  port: 8000
  max_batch_size: 16
  max_wait_time: 0.05
```

Under the hood, `vla-eval serve` resolves `script` to an absolute path, converts `args` into CLI flags (`--key value`; bools become bare `--flag`), and runs `uv run <script> <flags>`. The script's inline metadata (`# /// script`) declares its own dependencies, so `uv` creates an isolated environment automatically — no manual install needed.

To run on a specific GPU, set `CUDA_VISIBLE_DEVICES` before the command.

Server configs live in `configs/model_servers/`. The `--sweep-batch-sizes` flag in `bench_supply.py` changes `max_batch_size` at runtime via the server's HTTP control plane (`GET /config?max_batch_size=B`), so you can sweep batch sizes without restarting.

## Auxiliary Scripts

Before running the demand/supply benchmarks, you may want to measure the basic timing properties of your setup:

- **`experiments/measure_sim_delay.py`** — Instant server that records per-step timestamps. Run alongside a single Docker shard to measure per-step simulation time. Useful for sanity-checking demand curve results.

- **`experiments/measure_inference_delay.py`** — Sends realistic observations to a running model server and measures per-request latency. Useful for sanity-checking supply curve results and separating cold-start from warm inference.

## Worked Example: DB-CogACT

DB-CogACT (dexbotic fine-tuned CogACT 7B) is evaluated across three benchmarks with different rendering backends and chunk sizes. The model server runs on a dedicated GPU node, while benchmark Docker shards run on a separate host. The supply curve is shared (same model architecture), but demand characteristics differ significantly depending on the rendering backend.

| Benchmark | Rendering | chunk_size | Work items | Image size |
|-----------|-----------|:----------:|:----------:|:----------:|
| LIBERO Spatial | GPU EGL (MuJoCo) | 12 | 500 (10 tasks × 50 ep) | 256×256 |
| CALVIN ABC→D | GPU EGL (PyBullet) | 7 | 1000 sequences × 5 subtasks | 200×200 |
| SimplerEnv | GPU (SAPIEN/Vulkan) | 5 | 96 (4 tasks × 24 ep) | 224×224 |

### Supply — μ(B)

The supply curve below was measured with the LIBERO checkpoint (chunk_size=12). **chunk_size directly affects supply**: with chunk_size=C, only 1 out of every C observations triggers GPU inference — the rest are served from the cached action chunk. Smaller chunk_size means more frequent GPU inference, so lower obs/s throughput.

```bash
# Start server:
vla-eval serve -c configs/model_servers/dexbotic_cogact_libero.yaml -v

# Sweep batch sizes (server must be running):
uv run python experiments/bench_supply.py \
    --url ws://localhost:8000 \
    --sweep-batch-sizes 1,2,4,8,16,24,32 \
    --num-clients 64 \
    --requests-per-client 200 \
    --image-size 256
```

**A100-80GB (PCIe)** — chunk_size=12:

| B (max_batch_size) | μ (obs/s) | Inference latency (p50) |
|:------------------:|:---------:|:-----------------------:|
| 1                  | 71.7      | 21.0ms                  |
| 2                  | 103.8     | 20.9ms                  |
| 4                  | 151.0     | 18.4ms                  |
| 8                  | 185.9     | 18.6ms                  |
| 16                 | 203.1     | 31.2ms                  |
| 24                 | 196.6     | 51.4ms                  |
| 32                 | 201.4     | 68.4ms                  |

**H100-80GB SXM** — chunk_size=12:

| B (max_batch_size) | μ (obs/s) | Inference latency (p50) |
|:------------------:|:---------:|:-----------------------:|
| 1                  | 165.2     | 18.2ms                  |
| 2                  | 255.4     | 18.5ms                  |
| 4                  | 347.3     | 20.0ms                  |
| 8                  | 423.8     | 23.9ms                  |
| 16                 | 468.2     | 28.7ms                  |
| 24                 | 485.5     | 38.8ms                  |
| 32                 | 483.2     | 52.5ms                  |

A100 peaks at B=16 (203.1 obs/s), H100 peaks at B=24 (485.5 obs/s). The pipelined dispatch loop overlaps batch collection with GPU inference, so fast GPUs (H100) no longer idle waiting for the next batch. H100 achieves ~2.4× the throughput of A100, consistent with its higher memory bandwidth and compute.

**Estimating supply for other chunk sizes**: The GPU inference rate (inf/s) is roughly constant across chunk sizes since model architecture is identical. At optimal B: A100 ≈ 203.1/12 ≈ 16.9 inf/s, H100 ≈ 485.5/12 ≈ 40.5 inf/s. For a different chunk_size C, estimated supply is approximately:

| chunk_size | A100 μ\* (est.) | H100 μ\* (est.) |
|:----------:|:---------------:|:---------------:|
| 12 (LIBERO) | 203 obs/s | 486 obs/s |
| 7 (CALVIN) | ~118 obs/s | ~283 obs/s |
| 5 (SimplerEnv) | ~85 obs/s | ~202 obs/s |

These are linear estimates (μ ≈ inf/s × C). Actual supply may differ — run `bench_supply.py` with each checkpoint for precise numbers.

### LIBERO Spatial (chunk_size=12, GPU EGL rendering)

#### Demand — λ(N)

```bash
uv run python experiments/bench_demand.py \
    --config configs/benchmarks/libero/spatial.yaml \
    --shards 1,8,16,24,32,50,64,80,100 \
    --episodes-per-shard 5 \
    --gpu 0
```

| N (shards) | observations | elapsed (s) | λ (obs/s) |
|:----------:|:------------:|:-----------:|:---------:|
| 1          | 1,100        | 98.2        | 11.2      |
| 8          | 8,800        | 106.2       | 82.9      |
| 16         | 17,600       | 115.7       | 152.1     |
| 24         | 26,400       | 123.2       | 214.2     |
| 32         | 35,200       | 131.8       | 267.1     |
| 50         | 55,000       | 150.8       | 364.6     |
| 64         | 70,400       | 171.5       | 410.2     |
| 80         | 88,000       | 196.8       | 446.8     |
| 100        | 110,000      | 282.3       | 389.4     |

λ(N) scales nearly linearly through N=80, peaking at 446.8 obs/s. Per-shard throughput gradually decreases from ~11.2 obs/s/shard (N=1) to ~7.3 (N=50), then degrades more sharply at N=64 (~6.4) and N=80 (~5.6) from host-level CPU/Docker overhead. At N=100, aggregate throughput drops to 389.4 obs/s — contention overwhelms parallelism. **Peak demand: N=80, λ≈447 obs/s.**

#### Derivation

|                    | A100-80GB         | H100-80GB         |
|--------------------|:-----------------:|:-----------------:|
| **max_batch_size** | 16                | 24                |
| **μ(B\*)**         | 203.1 obs/s       | 485.5 obs/s       |
| **num_shards**     | 20                | 50                |
| **max_wait_time**  | 1.31s             | 0.79s             |

1. **`max_batch_size`**: A100 peaks at B=16, H100 at B=24.

2. **`num_shards`**: Per-shard demand is ~7.3 obs/s. With ~20% margin: N ≤ 0.8 × μ(B*) / 7.3 → A100: 0.8×203.1/7.3 ≈ 22, H100: 0.8×485.5/7.3 ≈ 53. Rounded down to **20** and **50** so that LIBERO Spatial's 500 work items divide evenly — 25 episodes/shard (A100) and 10 episodes/shard (H100).

3. **`max_wait_time`**: predict_rate = λ(N) / chunk_size. A100: 146/12 ≈ 12.2 → 16/12.2 ≈ 1.31s. H100: 365/12 ≈ 30.4 → 24/30.4 ≈ 0.79s.

**Production result (H100, 50 shards)**: All 4 LIBERO suites (2000 episodes) completed in **~18 min** wall-clock with `max_batch_size=16`, `max_wait_time=0.05` — a ~47× throughput gain over sequential execution. See `docs/reproductions/db-cogact.md` (LIBERO section) for full results.

**Scaling note**: H100 supply (485.5 obs/s) exceeds single-host demand peak (446.8 obs/s at N=80), so one H100 can saturate a full host of Docker shards. A100 (203.1 obs/s) requires ~2 replicas to match peak demand.

### CALVIN ABC→D (chunk_size=7, GPU EGL rendering)

CALVIN uses PyBullet with GPU EGL rendering (like LIBERO). Physics is CPU but image rendering uses GPU via EGL. Each sequence chains 5 subtasks with up to 360 steps each. PyBullet steps are faster than MuJoCo, giving CALVIN a much higher per-shard observation rate (~36.7 obs/s vs LIBERO's ~11.2 obs/s).

#### Demand — λ(N)

```bash
uv run python experiments/bench_demand.py \
    --config configs/benchmarks/calvin/eval.yaml \
    --shards 1,4,8,16,24,32 \
    --episodes-per-shard 3 \
    --timeout 300
```

| N (shards) | observations | elapsed (s) | λ (obs/s) |
|:----------:|:------------:|:-----------:|:---------:|
| 1          | 1,080        | 29.4        | 36.7      |
| 4          | 4,320        | 36.7        | 117.6     |
| 8          | 8,640        | 39.2        | 220.4     |
| 16         | 17,280       | 46.0        | 376.0     |
| 24         | 25,920       | 60.0        | 432.3     |
| 32         | 34,560       | 87.6        | 394.6     |

λ(N) scales well through N=24, peaking at 432.3 obs/s. Per-shard throughput decreases from ~36.7 obs/s/shard (N=1) to ~23.5 (N=16) to ~18.0 (N=24) to ~12.3 (N=32). At N=32, aggregate throughput drops — CPU/Docker overhead overwhelms parallelism. **Peak demand: N=24, λ≈432 obs/s.**

CALVIN's high per-shard obs rate means demand exceeds the estimated H100 supply (~283 obs/s) at just N=16. This makes CALVIN **supply-bottlenecked** — unlike LIBERO and SimplerEnv where demand is the bottleneck.

#### Derivation

|                    | H100-80GB         |
|--------------------|:-----------------:|
| **max_batch_size** | 24                |
| **μ(B\*) (est.)**  | ~283 obs/s        |
| **num_shards**     | 16                |
| **max_wait_time**  | 0.42s             |

1. **`max_batch_size`**: H100 peaks at B=24 at chunk_size=12. Optimal B may differ at chunk_size=7 — run `bench_supply.py` with the CALVIN checkpoint to confirm.

2. **`num_shards`**: Per-shard demand is ~23.5 obs/s at N=16. With ~20% margin: N ≤ 0.8 × 283 / 23.5 ≈ 9.6. However, CALVIN is supply-bottlenecked — even at N=16, demand (376 obs/s) exceeds estimated supply (~283 obs/s). In practice, the server's batch queue absorbs the excess demand without failures because chunk buffering reduces actual predict calls to ~54/s (376/7). Use **16** shards so that 1000 sequences divide into ~62–63 per shard.

3. **`max_wait_time`**: predict_rate = λ(N) / chunk_size. At N=16 (supply-limited, effective λ ≈ 283): 283 / 7 ≈ 40.4 → 24 / 40.4 ≈ 0.59s. In production, `max_wait_time=0.05s` was used — batches fill almost instantly because demand far exceeds supply, so the wait timer rarely fires.

**Production result (H100, 16 shards)**: All 1000 sequences completed in **~33 min** wall-clock with `max_batch_size=16`, `max_wait_time=0.05` — a ~16× throughput gain over sequential (513.9 min). See `docs/reproductions/db-cogact.md` (CALVIN section) for full results (Avg Len 4.051, reproducing reference 4.063).

**Supply-bottleneck note**: CALVIN's fast PyBullet rendering generates demand faster than the model server can process. The server's internal queue absorbs bursts, but sustained overload would increase latency. To increase throughput: (a) use a faster GPU, (b) reduce num_shards to stay within supply, or (c) run multiple model server replicas behind a load balancer.

### SimplerEnv WidowX Bridge (chunk_size=5, GPU rendering)

SimplerEnv uses SAPIEN/Vulkan GPU rendering on the benchmark host. Multiple shards compete for GPU memory and compute — like LIBERO/CALVIN (EGL) but with heavier GPU usage per shard.

#### Demand — λ(N)

```bash
uv run python experiments/bench_demand.py \
    --config configs/benchmarks/simpler/widowx_vm.yaml \
    --shards 1,4,8,16,24,32 \
    --episodes-per-shard 5 \
    --gpu 0 \
    --timeout 300
```

| N (shards) | observations | elapsed (s) | λ (obs/s) |
|:----------:|:------------:|:-----------:|:---------:|
| 1          | 2,400        | 238.4       | 10.1      |
| 4          | 9,600        | 331.6       | 28.9      |
| 8          | 19,200       | 272.6       | 70.4      |
| 16         | 38,400       | 299.0       | 128.4     |
| 24         | 42,825       | 298.1       | 143.7 *   |
| 32         | 41,484       | 300.6       | 138.0 *   |

\* timeout — partial results

λ(N) scales sub-linearly from the start because multiple SAPIEN rendering shards contend for GPU resources on the benchmark host. Per-shard throughput drops from ~10.1 obs/s/shard (N=1) to ~7.2 (N=8) to ~6.0 (N=24). At N=32, aggregate throughput actually decreases — GPU contention between rendering shards overwhelms parallelism. **Peak demand: N=24, λ≈144 obs/s.** Far lower than LIBERO's 447 obs/s peak because SAPIEN shards consume significantly more GPU memory per shard than MuJoCo EGL.

#### Derivation

|                    | H100-80GB         |
|--------------------|:-----------------:|
| **max_batch_size** | 24                |
| **μ(B\*) (est.)**  | ~202 obs/s        |
| **num_shards**     | 16                |
| **max_wait_time**  | 0.93s             |

1. **`max_batch_size`**: H100 peaks at B=24 at chunk_size=12. Optimal B may differ at chunk_size=5 — run `bench_supply.py` with the SimplerEnv checkpoint to confirm.

2. **`num_shards`**: Per-shard demand is ~7.2 obs/s at moderate N. With ~20% margin: N ≤ 0.8 × 202 / 7.2 ≈ 22. However, SimplerEnv's GPU rendering saturates demand at N=24 (λ≈144 obs/s) — adding more shards decreases throughput. Rounded down to **16** so that 96 work items (4 tasks × 24 episodes) divide evenly — 6 episodes/shard. λ(16) ≈ 128.4 obs/s, below the ~202 estimated supply ceiling.

3. **`max_wait_time`**: predict_rate = λ(N) / chunk_size = 128.4 / 5 ≈ 25.7 → 24 / 25.7 ≈ 0.93s.

**GPU contention note**: SimplerEnv rendering shards share GPU resources on the benchmark host (model inference runs on a separate GPU node). Demand saturates much earlier than LIBERO/CALVIN because SAPIEN shards consume more GPU memory per shard than MuJoCo/PyBullet EGL. To increase demand headroom, spread shards across multiple GPUs on the benchmark host (e.g. `--gpus 0,1`).

### Cross-Benchmark Comparison

|                    | LIBERO Spatial (A100) | LIBERO Spatial (H100) | CALVIN (H100) | SimplerEnv (H100) |
|--------------------|:---------------------:|:---------------------:|:-------------:|:-----------------:|
| **Rendering**      | GPU EGL               | GPU EGL               | GPU EGL       | GPU (SAPIEN)      |
| **chunk_size**     | 12                    | 12                    | 7             | 5                 |
| **max_batch_size** | 16                    | 24                    | 16            | 24                |
| **num_shards**     | 20                    | 50                    | 16            | 16                |
| **max_wait_time**  | 1.31s                 | 0.79s                 | 0.05s         | 0.93s             |
| **λ(N\*) obs/s**   | ~146                  | ~365                  | ~376          | ~128              |
| **μ(B\*) obs/s**   | 203.1                 | 485.5                 | ~283 (est.)   | ~202 (est.)       |
| **Headroom**       | 28%                   | 25%                   | −33% (supply-bottlenecked) | 37%   |

Key takeaways:
- **Lightweight GPU rendering** (LIBERO, CALVIN via EGL) scales to many more shards — MuJoCo/PyBullet EGL uses minimal GPU memory per shard.
- **GPU-rendered benchmarks** (SimplerEnv) saturate much earlier because rendering shards contend for GPU resources on the benchmark host. Spread shards across multiple GPUs to increase demand headroom.
- **Supply-bottlenecked benchmarks** (CALVIN): Fast per-shard obs rate + small chunk_size can push demand above supply. The server queue absorbs bursts, but sustained overload increases latency. Use fewer shards, a faster GPU, or model server replicas.
- **chunk_size** directly affects both supply and max_wait_time: larger chunks (LIBERO=12) mean fewer GPU inferences per observation → higher supply ceiling but slower batch fill → longer max_wait_time. Smaller chunks (SimplerEnv=5) → lower supply ceiling but faster batch fill → shorter max_wait_time.
- **Work item count** constrains num_shards: always round down to a divisor of total work items to avoid shard imbalance.
