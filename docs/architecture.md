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
| **Model Server** | Runs model inference. User implements `predict()` or `predict_batch()`. | ✅ |
| **Connection** | Client library for benchmark→server communication. Framework-provided. | ✅ |
| **Benchmark** | Async environment interface; StepBenchmark supplies synchronous reset/step helpers; VectorStepBenchmark steps `num_envs` environments together. | ✅ |
| **EpisodeRunner** | Episode execution strategy. Combines Benchmark + Connection. | ✅ Sync, ✅ Live, ✅ Vector |
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

## Server Startup & Readiness

Model servers load all weights inside `__init__` (the framework starts listening only after `__init__` returns), so the standard readiness probe is:

```bash
curl -fsS http://host:port/health   # → HTTP 200 once the server is up
```

TCP listen, HTTP routes (`/health`, `/config`), and the WebSocket HELLO handshake all become live at the same moment, when `__init__` completes. An `__init__` that throws (OOM, bad config, missing checkpoint) kills the process before anything starts listening, so a failing connect is itself a fail-fast signal.

Per episode, the server answers `EPISODE_START` with `EPISODE_READY` once `on_episode_start` returned (or `ERROR` if it failed); the runner waits for it (at most `server.timeout`) before its clock starts and the first observation goes out. Both HELLOs advertise `capabilities.episode_ready`; with an older peer nothing is sent or awaited. A live episode applies the benchmark's hold action until the first action arrives.

## Episode Execution Flow (Sync)

```
Orchestrator          SyncEpisodeRunner        Connection         Model Server
    │                       │                      │                    │
    │── run_episode() ────► │                      │                    │
    │                       │── start_episode() ──►│── episode_start ──►│
    │                       │                      │◄── episode_ready ──│
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

## Vectorized Benchmarks

A `VectorStepBenchmark` steps `num_envs` environments together; each step is one `VECTOR_STEP` round trip whose
observation columns the server infers as one batch, a row per slot. Sync mode only.
See [RFC-0009](rfcs/0009-vector-benchmarks.md).

```
Orchestrator ── VectorEpisodeRunner ── Connection ──► Model Server
                 │ reset(slots, ...)   VECTOR_STEP     (one row per slot,
                 │ step(slots, acts) ◄─VECTOR_RESULT   predict_batch(obs, rows))
                 ▼
          VectorStepBenchmark (num_envs environments)
```

## Communication Protocol

**WebSocket + msgpack** over a single connection.

Every message is a msgpack binary frame:

```python
{
    "type": "observation" | "action" | "episode_start" | "episode_ready" | "episode_end" | "error" | ...,
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

Observations and actions are `dict[str, Any]`; the framework imposes **no fixed schema**. Each benchmark defines its own observation structure via `make_obs()`. The recommended (not enforced) convention:

| Field | Type | Description |
|-------|------|-------------|
| `images` | `dict[str, np.ndarray]` | Camera name → HWC uint8 image |
| `states` | `np.ndarray` | Proprioception |
| `task_description` | `str` | Natural language instruction |

## Model Server

```
ModelServer (ABC)            ← async callbacks: on_episode_start / on_observation / on_episode_end,
│                              answered with send_action / send_error; fail_episode ends one as failed
├── PredictModelServer       ← subclass this: predict(obs, row) or predict_batch(obs, rows), batched across
│                              rows and connections (PredictionEngine); hooks reset_episodes / finish_episodes
└── DaemonModelServer        ← run(ep) per episode (EpisodeDaemons): the loop decides when to infer, reply or
                               publish, and advances its clock (SimClock when sync, WallClock when live)
```

Execution policies are loops on these primitives, not core classes: continuous inference is a few lines of
`run` (see `DaemonModelServer`'s docstring); RTC streaming is [an example](../examples/live_servers/rtc.py).

Every running episode is a **row** of the model server (a connection's episode, or a vector slot); `row.id` keys
per-episode state. Hooks and options: see [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-model-server).

One call into the model runs at a time, on one model thread. The harness ends every episode the server started
with `EPISODE_END {status, failure_reason}`; the server ends those a client dropped as `aborted`.

Live episodes may get server-initiated messages, each negotiated in HELLO (`capabilities: {"publish": true,
"episode_error": true}`) and bound to the episode's `EPISODE_START` seq: `PUBLISH {action, index, source_seq}`
(applied latest-wins with replies; a stale index drops) and `EPISODE_ERROR {error, failure_reason: "model_act"}`
(the client ends the episode `model_act`). A server that `requires` a capability refuses an old client's start
(`ERROR` instead of `EPISODE_READY`, so the start fails `model_start`) and closes its socket. A `DaemonModelServer`
sends `EPISODE_READY` once `reset_episodes` returned and the episode's `run` is scheduled. `EPISODE_START` may carry
`control_hz` (the live runner's `hz`, else `BenchmarkCommon.control_hz`): a `DaemonModelServer` loop's `ep.sim_dt`.

The server runner (`model_servers/serve.py`) wraps any `ModelServer` into a WebSocket server with `serve(model_server, host, port)`.

## Configuration

Flat YAML. A single file contains all evaluation parameters.

```yaml
server:
  url: "ws://localhost:8000"
output_dir: "./results"
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    mode: sync                    # "sync" or "live"
    episodes_per_task: 50
    max_steps: 300                # optional, falls back to benchmark.get_metadata()
    params:                       # passed to Benchmark constructor as **kwargs
      suite: libero_spatial
      seed: 7
```

The full config is saved alongside results for reproducibility.

**`vla-eval test --validate`** checks that all `benchmark` import paths resolve to valid `Benchmark` subclasses.

## Import Resolution

Benchmarks are referenced by full import path (`"module.path:ClassName"`) in config files. No registry or short-name mapping; explicit is better than implicit.

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
    mode: str                        # "sync" | "live"
    harness_version: str
    tasks: list[TaskResult]          # per-task success rate, avg steps
    mean_success: float
    config: dict                     # config snapshot for reproducibility
```

Results and step fields are stored in one SQLite DB per evaluation. Export produces episode JSONL and aggregate JSON, selecting the last committed attempt per task/episode. Shard completion records preserve partial-run status. Optional episode videos are saved separately as MP4.

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

- **Reference score comparison**: Regression testing against known model+benchmark scores.

## Design Background

The design philosophy and detailed design outline were written before implementation:

- [Design Philosophy](design-philosophy.md): Core principles: freshness, convenience, layered abstraction, quality, reproducibility, openness.
- [RFCs](rfcs/README.md): Detailed component specifications, protocol design, and technical decisions.

These documents reflect the intended direction. The implementation follows them closely.
