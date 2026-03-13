# RFC-0001: Real-time Evaluation as Architectural Foundation

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2025-02-22
- **Requires:** —
- **Superseded-By:** —

## Summary

This project exists because **no existing VLA evaluation framework measures what actually matters for deployment: whether a model can control a robot in real time.** Sync-only evaluation freezes the world while the model thinks — real robots cannot do this. Real-time evaluation, where the simulation clock is tied to wall-clock time and the environment keeps advancing whether or not the model has returned an action, is therefore not a feature but the core architectural requirement. Every major design decision in vla-evaluation-harness — client-server separation, async model interfaces, the Benchmark/EpisodeRunner split — traces back to this requirement.

## Motivation

Standard evaluation frameworks (e.g. `model.predict(obs)` inside a `gym.step()` loop) are **synchronous**: the environment blocks until the model returns an action. This has three consequences:

1. **Time stops during inference.** A model taking 500ms per action gets the same score as one taking 5ms, even though the slow model would fail on a real robot where objects keep moving.
2. **Sync scores are misleading.** A model scoring 90% success in sync evaluation might score 60% under real-time constraints because half its actions arrive too late. Without measuring this gap, researchers cannot assess deployment readiness.
3. **gymnasium's `step()` makes real-time structurally impossible.** The blocking `action = model(obs); obs, reward, done, info = env.step(action)` loop couples environment time to inference time. You cannot simply "add a timer" — the architecture must decouple them from the start.

These are not hypothetical problems. VLA models vary in inference latency from 5ms (small diffusion policies) to 500ms+ (large autoregressive models). This 100× range means real-time performance is a first-class evaluation axis, not an afterthought.

## What is Real-time Evaluation?

Real-time evaluation **ties the simulation clock to wall-clock time** — the environment keeps advancing whether or not the model has returned an action:

- If inference is still running when a step occurs, a **hold policy** fills the gap (repeat last action, output zeros, or a custom callable).
- The same `Benchmark` implementation works with both sync and real-time runners — the benchmark defines *what* to evaluate, not *how fast*.

This is exactly what happens on a real robot: the control loop keeps running, and the policy must keep up or the robot acts on stale commands.

## Architectural Consequences

Real-time evaluation requires decoupling inference from environment stepping. This single requirement drives four architectural decisions:

### Client-Server Separation (→ RFC-0002)

The model and environment must run in **separate processes** (typically separate machines). This is not just for deployment convenience — it is necessary so that the environment loop and inference loop can advance on independent clocks. A single-process `model.predict()` call inside `env.step()` makes decoupling impossible.

### Model Server is Async by Default (→ RFC-0003)

The base `ModelServer` interface is `async` because in real-time mode, observations arrive while inference is still running. The server must be able to receive new observations, decide which to process, and manage its own inference pipeline — none of which is possible with a blocking `predict(obs) → action` signature. (A blocking `PredictModelServer` convenience class is provided for the common case, but it builds on the async foundation.)

### Benchmark ≠ EpisodeRunner (→ RFC-0004)

`Benchmark` defines the environment interface (reset, step, observation format, success criteria). `EpisodeRunner` defines the execution strategy (sync vs. real-time, timing, hold policies). This separation exists so that a single benchmark implementation (e.g. `LIBEROBenchmark`) works with both `SyncEpisodeRunner` and `AsyncEpisodeRunner` without modification. If the benchmark owned the execution loop, every benchmark would need to implement real-time logic.

### Connection Dual API

The `Connection` class exposes two calling conventions on the same WebSocket:

```python
class Connection:
    # Sync mode — blocks until action arrives
    async def act(self, obs: dict[str, Any]) -> dict[str, Any]: ...

    # Real-time mode — non-blocking send + action callback
    async def send_observation(self, obs: dict[str, Any]) -> None: ...
    def on_action(self, callback: Callable[[dict[str, Any]], None]) -> None: ...
```

`act()` is request-response: send observation, await action. `send_observation()` + `on_action()` is fire-and-forget: observations flow out, actions arrive asynchronously via callback. The EpisodeRunner chooses which API to use.

## AsyncEpisodeRunner Design

```python
class AsyncEpisodeRunner(EpisodeRunner):
    hz: float = 10.0                    # environment step frequency
    hold_policy: str = "repeat_last"    # what to do when no new action
    max_steps: int = 300

    async def run_episode(self, benchmark, task, conn):
        env, obs_dict = benchmark.reset(task)
        await conn.start_episode({"task": task, "mode": "realtime"})

        action_buffer = ActionBuffer(hold_policy=self.hold_policy)
        conn.on_action(lambda a: action_buffer.update(a))
        await conn.send_observation(obs_dict)

        for step in range(self.max_steps):
            action = action_buffer.get()            # latest action or hold policy
            result = benchmark.step(env, action)
            if benchmark.is_done(result): break
            obs_dict = benchmark.make_obs(result.obs, task)
            await conn.send_observation(obs_dict)
            await asyncio.sleep(1 / self.hz)        # wall-clock pacing

        await conn.end_episode(benchmark.get_result(result))
        return benchmark.get_result(result)
```

**Key mechanics:**

1. **Fixed Hz via `asyncio.sleep(1/hz)`** — the environment advances on a wall-clock schedule. Model inference runs concurrently and does not block stepping.
2. **ActionBuffer** stores the latest action received from `on_action`. Each step reads from the buffer, not from a blocking call.
3. **Hold policy** determines behavior when the buffer has no new action: `repeat_last` replays the previous action, `zero` outputs a zero action, or a custom callable computes a fallback.
4. **Non-blocking observation sending** — `send_observation()` returns immediately. The model server may skip observations it cannot process in time.

## Real-time Metrics

Real-time evaluation introduces metrics that sync evaluation cannot capture:

| Metric | Description |
|--------|-------------|
| **RT Success Rate** | Task success rate under real-time (fixed Hz) conditions |
| **RT Degradation** | Success rate drop from sync to real-time (quantifies the "deployment gap") |
| **Effective Control Frequency** | Rate at which the agent produces genuinely new actions |
| **Stale Action Ratio** | Fraction of environment steps using a repeated (hold-policy) action |
| **Latency Distribution** | Inference latency percentiles: p50, p95, p99 |

**RT Degradation** is the headline metric. A model with 90% sync / 85% real-time has 5% degradation — likely deployment-ready. A model with 90% sync / 40% real-time has 50% degradation — fast inference or architectural changes are needed regardless of the sync score.

## Implementation Status

- ✅ `SyncEpisodeRunner` — fully implemented and tested across LIBERO, CALVIN, SimplerEnv
- ✅ Client-server architecture — WebSocket + msgpack protocol operational
- ✅ Protocol timestamps — `seq` and `timestamp` fields on every message
- ✅ `Connection.send_observation()` / `on_action()` / `start_listener()` / `stop_listener()`
- ✅ `AsyncEpisodeRunner` — fully implemented (`runners/async_runner.py`)
- ✅ `ActionBuffer` + hold policies (`repeat_last`, `zero`, callable) — (`runners/action_buffer.py`)
- ✅ Real-time metrics collection — effective control Hz, step timing, stale action ratio
- ✅ `Clock` abstraction for pace control (`runners/clock.py`)



