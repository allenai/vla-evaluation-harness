# Architecture

Status markers: ✅ implemented, 🚧 partial, 🔜 planned.

## Component Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Orchestrator (CLI)                         │
│  Reads config → creates benchmark + runner → manages episodes      │
└──────┬───────────────────────────────────┬──────────────────────────┘
       │                                   │
       ▼                                   ▼
┌──────────────┐  run_episode()   ┌─────────────────┐
│  Benchmark   │◄────────────────►│  EpisodeRunner   │
│  (env logic) │                  │  (episode loop)  │
└──────────────┘                  └────────┬─────────┘
                                           │ act() / send_observation()
                                           ▼
                                  ┌─────────────────┐     WebSocket
                                  │   Connection     │◄──────────────►  Model Server
                                  │   (client lib)   │    (msgpack)      (your model)
                                  └─────────────────┘
```

| Component | Role | Status |
|-----------|------|--------|
| **Protocol** | WebSocket + msgpack serialization, message schema, numpy codec | ✅ |
| **Model Server** | Runs model inference. User implements `predict()`. | ✅ |
| **Connection** | Client library for benchmark→server communication. Framework-provided. | ✅ |
| **Benchmark** | Environment interface ABC (reset, step, make_obs, is_done). User implements per benchmark. | ✅ |
| **EpisodeRunner** | Episode execution strategy. Combines Benchmark + Connection. | ✅ Sync, ✅ Async (Sim2Live) |
| **Orchestrator** | Coordinates evaluation: config parsing, benchmark creation, episode iteration, result saving. | ✅ |
| **ResultCollector** | Aggregates episode→task→benchmark metrics. JSON output + summary table. | ✅ |
| **Import Resolution** | Resolves `"module:Class"` import strings from config to actual classes. | ✅ |

## Environment Isolation

Each benchmark runs in a Docker container (with optional GPU access for rendering). The model server runs outside, communicating via WebSocket. This eliminates dependency conflicts between benchmarks.

```
Host machine                          Docker container (optional GPU)
┌──────────────────┐                  ┌──────────────────────┐
│  Model Server    │◄── WebSocket ──► │  Benchmark + CLI     │
│  (inference)     │                  │  (LIBERO/SIMPLER/…)  │
└──────────────────┘                  └──────────────────────┘
```

Each integrated benchmark has a corresponding Dockerfile under `docker/`.

## Episode Execution Flow (Sync)

```
Orchestrator          SyncEpisodeRunner        Connection         Model Server
    │                       │                      │                    │
    │── run_episode() ────► │                      │                    │
    │                       │── start_episode() ──►│── episode_start ──►│
    │                       │                      │                    │
    │                       │   ┌─ step loop ──────────────────────┐    │
    │                       │   │ obs_dict = make_obs(raw_obs)     │    │
    │                       │   │ action = conn.act(obs_dict)      │    │
    │                       │   │         ──observation──►         │    │
    │                       │   │         ◄────action────          │    │
    │                       │   │ result = benchmark.step(action)  │    │
    │                       │   │ if is_done(result): break        │    │
    │                       │   └──────────────────────────────────┘    │
    │                       │                      │                    │
    │                       │── end_episode() ────►│── episode_end ───►│
    │◄─ episode_result ─────│                      │                    │
```

## Communication Protocol

**WebSocket + msgpack** over a single connection.

Every message is a msgpack binary frame:

```python
{
    "type": "observation" | "action" | "episode_start" | "episode_end" | "error",
    "payload": dict[str, Any],   # benchmark-specific, no fixed schema
    "seq": int,                  # monotonically increasing sequence number
    "timestamp": float,          # wall-clock Unix epoch
}
```

### numpy serialization

numpy arrays are encoded inline within msgpack:

```python
{"__ndarray__": True, "data": bytes, "dtype": str, "shape": tuple}
```

**Security**: `Void`, `Object`, and `Complex` dtypes are rejected on deserialization to prevent arbitrary code execution.

### Payload convention

Observations and actions are `dict[str, Any]` — the framework imposes **no fixed schema**. Each benchmark defines its own observation structure via `make_obs()`. The recommended (not enforced) convention:

| Field | Type | Description |
|-------|------|-------------|
| `images` | `dict[str, np.ndarray]` | Camera name → HWC uint8 image |
| `states` | `np.ndarray` | Proprioception |
| `task_description` | `str` | Natural language instruction |

## Model Server Hierarchy

```
ModelServer (ABC)                    ← Advanced: async on_observation()
├── PredictModelServer               ← Most models: blocking predict()
└── BatchPredictModelServer          ← Batched inference across sessions
```

**`PredictModelServer`** wraps `predict()` in `run_in_executor` and manages action chunk buffers automatically. Supports `chunk_size`, `action_ensemble` ("newest", "average", "ema", callable), and `ema_alpha`.

The server runner (`server/runner.py`) wraps any `ModelServer` into a WebSocket server with `serve(model_server, host, port)`.

## Configuration

Flat YAML. A single file contains all evaluation parameters.

```yaml
server:
  url: "ws://localhost:8000"
output_dir: "./results"
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    mode: sync                    # "sync" or "realtime"
    episodes_per_task: 50
    max_steps: 300                # optional, falls back to benchmark.get_metadata()
    params:                       # passed to Benchmark constructor as **kwargs
      suite: libero_spatial
      seed: 7
```

The full config is saved alongside results for reproducibility.

**`vla-eval validate --config ...`** checks that all `benchmark` import paths resolve to valid `Benchmark` subclasses.

## Import Resolution

Benchmarks are referenced by full import path (`"module.path:ClassName"`) in config files. No registry or short-name mapping — explicit is better than implicit.

```python
from vla_eval.registry import resolve_import_string

cls = resolve_import_string("vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark")
```

Import resolution uses `lazyregistry.ImportString` under the hood. Heavy dependencies (robosuite, MuJoCo, SAPIEN) are only imported when the benchmark class is actually instantiated.

## Result Collection

Results are aggregated hierarchically: **episode → task → benchmark**.

```python
# Structured output (TypedDict)
BenchmarkResult:
    benchmark: str
    mode: str                        # "sync" | "realtime"
    harness_version: str
    tasks: list[TaskResult]          # per-task success rate, avg steps
    overall_success_rate: float
    config: dict                     # config snapshot for reproducibility
```

Results are saved as JSON and printed as a human-readable summary table.

## Error Handling

Failures are isolated at the **episode level**. One episode crashing does not abort the evaluation.

| Scenario | Behavior |
|----------|----------|
| Action timeout | Episode marked as failed, next episode proceeds |
| Server crash (WebSocket disconnect) | Episode failed, reconnection attempted |
| Environment exception (`step()`/`reset()`) | Episode failed, environment re-initialized |
| Protocol error (bad message) | Episode failed, error logged |

All failures are recorded in structured results with `failure_reason`.

## Planned Features

- **Video / trajectory recording** — Per-episode video via `Benchmark.render()` and trajectory logs in msgpack format.
- **Reference score comparison** — Regression testing against known model+benchmark scores.

## Design Background

The design philosophy and detailed design outline were written before implementation:

- [Design Philosophy](design-philosophy.md) — Core principles: freshness, convenience, layered abstraction, quality, reproducibility, openness.
- [RFCs](rfcs/README.md) — Detailed component specifications, protocol design, and technical decisions.

These documents reflect the intended direction. The implementation follows them closely.
