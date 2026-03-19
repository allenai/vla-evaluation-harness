# RFC-0004: Benchmark and Episode Execution

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2025-02-22
- **Requires:** RFC-0001
- **Superseded-By:** —

## Summary

This RFC defines the benchmark environment interface (`Benchmark` ABC), the episode execution strategies (`EpisodeRunner`), environment isolation via Docker, component registration, orchestration, configuration, error handling, result collection, and logging. It is the largest single design surface in the framework.

The key architectural insight (from RFC-0001): separating *what* the environment does (`Benchmark`) from *how* episodes are executed (`EpisodeRunner`) allows the same benchmark implementation to work in both synchronous and real-time evaluation modes without modification.

## Benchmark ABC

Each benchmark has a unique environment interface (observation structure, action space, reset/step API, success criteria). To add a new benchmark, implement the `Benchmark` ABC — this is the only implementation contract on the benchmark side.

```python
@dataclass
class StepResult:
    obs: Any              # raw environment observation
    reward: float
    done: bool
    info: dict[str, Any]

class Benchmark(ABC):
    def get_tasks(self) -> list[dict[str, Any]]: ...
    async def start_episode(self, task: dict[str, Any]) -> None: ...
    async def apply_action(self, action: dict[str, Any]) -> None: ...
    async def get_observation(self) -> dict[str, Any]: ...
    async def is_done(self) -> bool: ...
    async def get_result(self) -> EpisodeResult: ...
    def get_metadata(self) -> dict[str, Any]: ...  # optional — benchmark defaults
    def render(self) -> np.ndarray | None: ...  # optional — for video recording
```

Six abstract methods form the minimum contract. `get_metadata()` and `render()` are optional overrides. The benchmark implementor focuses only on environment-specific logic — episode loops, communication, and timing are handled by `EpisodeRunner`. Environment handles are stored internally (`self._env`) rather than passed through callers.

## EpisodeRunner

Defines *how* episodes are executed. Combines a `Benchmark` (environment) with a `Connection` (communication) to drive episodes.

```python
class EpisodeRunner(ABC):
    async def run_episode(self, benchmark: Benchmark, task: dict, conn: Connection) -> dict: ...
```

### SyncEpisodeRunner

Waits for model inference to complete before stepping. Environment time is coupled to inference speed. Used for most simulation benchmark evaluations.

```python
class SyncEpisodeRunner(EpisodeRunner):
    max_steps: int = 300

    async def run_episode(self, benchmark, task, conn):
        await benchmark.start_episode(task)
        obs_dict = await benchmark.get_observation()
        await conn.start_episode({"task": task})
        for step in range(self.max_steps):
            action = await conn.act(obs_dict)          # blocks until inference completes
            await benchmark.apply_action(action)
            if await benchmark.is_done(): break
            obs_dict = await benchmark.get_observation()
        result = await benchmark.get_result()
        await conn.end_episode(result)
        return result
```

### AsyncEpisodeRunner (→ RFC-0001)

Ties the simulation clock to wall-clock time — the environment keeps advancing whether or not the model has returned an action. If inference hasn't completed, a hold policy is applied (e.g., repeat last action). Used for real-time evaluation.

Key mechanisms:
- **Wall-clock stepping**: `asyncio.sleep(1/hz)` keeps the environment advancing independent of agent inference.
- **Action buffer**: `on_action` callback stores arriving actions; each step reads the latest.
- **Hold policy**: `repeat_last` (default), `zero`, or custom callable when buffer is empty.
- **Non-blocking observation**: `conn.send_observation()` sends each step's observation without waiting.

### Custom EpisodeRunner

For non-step-based environments (real robot hardware, BEHAVIOR-1K WebSocket servers), implement `EpisodeRunner` directly. The custom runner manages its own loop, communication, and timing inside `run_episode()`. The `Benchmark` may only provide `get_tasks()` in this case.

## Environment Isolation (Docker)

Each benchmark's simulation environment runs in a Docker container (with GPU access when needed for rendering), eliminating dependency conflicts between benchmarks (e.g., robosuite vs SAPIEN).

- **Image naming**: `vla-eval/{benchmark_name}:{version}` (e.g., `vla-eval/libero:0.1.0`)
- **Connection library included**: benchmark images include the `vla-eval` connection library
- **Port 8080**: exposed for the orchestrator control channel
- **Model server**: runs outside the container, communicates via WebSocket — unaffected by isolation

### Container Lifecycle

```
Orchestrator                    Docker Container
    |--- docker run ---------------→ |  (start container)
    |                                |  (initialize environment)
    |←-- /healthz 200 OK ---------- |  (ready signal)
    |--- WebSocket connect --------→ |  (connect to model server)
    |    [episode loop]              |
    |--- episode_start -----------→ |
    |←-- observation/action ------→ |
    |--- episode_end -------------→ |
    |    ...                         |
    |--- SIGTERM -----------------→ |  (graceful shutdown)
    |    (10s wait)                   |
    |--- SIGKILL -----------------→ |  (force kill if needed)
```

## Import Resolution

Benchmarks are referenced by full import path (`"module.path:ClassName"`) in config files — no registry or short-name mapping.

```python
from vla_eval.registry import resolve_import_string
cls = resolve_import_string("vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark")
```

This is explicit, requires no registration step, and supports external packages (just use their import path in config).

## Orchestrator

Coordinates the full evaluation flow:

1. **Start container**: launch the benchmark's Docker container, manage ports and volumes
2. **Health check**: poll `/healthz` until ready (timeout 120s, then skip to next benchmark)
3. **Connect**: WebSocket to model server with exponential backoff (max 5 retries)
4. **Select runner**: `SyncEpisodeRunner` or `AsyncEpisodeRunner` based on config `mode`
5. **Run episodes**: iterate `benchmark.get_tasks()`, call `runner.run_episode()` for each
6. **Collect results**: pass episode metrics to `ResultCollector`
7. **Shutdown**: SIGTERM → 10s wait → SIGKILL, generate final report

**Execution model**: single session by default (one model server, one benchmark client). Benchmarks run sequentially. The interface supports multi-session via `session_id` in `SessionContext` without changes.

## Configuration

Flat YAML — single file contains all parameters. No Hydra-style composition.

```yaml
server:
  url: "ws://localhost:8000"
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    mode: sync
    tasks: ["libero_spatial", "libero_object"]
    episodes_per_task: 50
    max_steps: 300          # omit to use Benchmark.get_metadata() default
    params:                 # benchmark-specific — passed to constructor
      headless: true
      seed: 42
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    mode: realtime
    tasks: ["libero_spatial"]
    realtime:
      hz: 10.0
      hold_policy: repeat_last
```

The full config is saved alongside results for reproducibility. `vla-eval test --validate` checks that all `benchmark` import paths resolve to valid `Benchmark` subclasses.

## Error Handling

Failures are isolated per episode — one episode failure does not abort the evaluation.

| Scenario | Behavior |
|----------|----------|
| Action timeout | Episode failed, next episode proceeds |
| Server crash | Episode failed, reconnect attempted; if reconnect fails, skip benchmark |
| Environment exception | Episode failed, environment re-initialized |
| Protocol error | Episode failed, error logged |

All failures are recorded in structured logs and included in the final report with failure reasons.

## Result Collection

Hierarchical aggregation: episode → task → benchmark.

```python
class EpisodeResult(TypedDict):
    task: str; episode_id: str; success: bool; steps: int
    elapsed_sec: float; failure_reason: str | None; extra: dict[str, Any]

class TaskResult(TypedDict):
    task: str; episodes: list[EpisodeResult]; success_rate: float; avg_steps: float

class BenchmarkResult(TypedDict):
    benchmark: str; mode: str; tasks: list[TaskResult]
    overall_success_rate: float; config: dict[str, Any]
```

Output: structured JSON (`BenchmarkResult`) + human-readable summary table. Inference latency (`p50`, `p95`, `p99`) computed from protocol message timestamps. Reference scores (YAML-managed) enable regression testing in CI.

## Logging

- **Execution logs**: episode start/end, connection state, errors — JSON Lines format
- **Trajectory recording** (optional): per-episode observation/action sequences in msgpack. Each step: `{"step", "obs", "action", "reward", "done", "timestamp"}`
- **Video recording** (optional): if `Benchmark.render()` is implemented, `EpisodeRunner` collects frames each step
- **Metadata**: config file, Docker image version, seed — auto-recorded with results

## Implementation Status

- ✅ Benchmark ABC (`benchmarks/base.py`), StepResult dataclass
- ✅ SyncEpisodeRunner (`runners/sync_runner.py`)
- ✅ AsyncEpisodeRunner (`runners/async_runner.py`)
- ✅ Orchestrator (`orchestrator.py`)
- ✅ Import resolution via `resolve_import_string()` (`registry.py`)
- ✅ YAML configuration loading
- ✅ ResultCollector with hierarchical aggregation (`results/collector.py`)
- ✅ Error isolation (per-episode try/except in orchestrator)
- ✅ LIBERO benchmark (score-verified: Spatial, Object)
- ✅ SIMPLER benchmark (score-verified: WidowX 4 tasks)
- ✅ Docker isolation (LIBERO, CALVIN, SimplerEnv, ManiSkill2, Kinetix images)
- ✅ Container lifecycle management
- 🔜 Reference score comparison
- 🔜 Video and trajectory recording



