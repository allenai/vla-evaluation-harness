# RFC-0009: Vectorized Benchmarks

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2026-10-03
- **Requires:** RFC-0004, RFC-0006, RFC-0007
- **Superseded-By:** —

## Summary

`VectorStepBenchmark` steps `num_envs` environments in one process; `VectorEpisodeRunner` runs up to
`num_envs` episodes on it over one connection, one `VECTOR_STEP` round trip per step.  Observations and
actions travel as columns (one array or list per key, a row per slot), so harness cost per step is nearly flat
in `num_envs`.  GPU-parallel simulators (MuJoCo Warp / mjlab, ManiSkill3, Isaac Lab) need this: one
environment per process (RFC-0006 shards) leaves the device idle.

## Benchmark contract

```python
Batch = dict[str, Any]  # a column per key: an array whose first axis is the row, or a list; row i is slots[i]

class VectorStepBenchmark(BenchmarkCommon):  # BenchmarkCommon: tasks, specs, metadata, cleanup (shared with Benchmark)
    partial_reset: ClassVar[bool] = False  # True: a freed slot restarts while others run; else waves of num_envs
    def __init__(self, num_envs: int = 1): ...
    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch: ...
    def step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]: ...  # observations, done
    def result(self, slot: int) -> EpisodeResult: ...  # the episode that just ended in slot
```

- `reset` starts `tasks[i]` in `slots[i]` without disturbing running slots; slots not in `slots` stay inert.
- An outcome depends only on its task, never on slot, `num_envs` or neighbours: seed per slot, no native auto-reset.
- Lists carry what does not stack: images as `{camera: HWC array}`, strings, ragged actions.
- Benchmark calls run on the orchestrator's thread (Isaac Kit, GL/Vulkan and CUDA state are thread-bound).
- Sync mode only; running episodes share one task, so an adapter builds one scene at a time.

## Wire contract

```
VECTOR_STEP   {"ends": [[slot, episode_end]], "starts": [[slot, episode_start]], "slots": [slot, ...], "observations": Batch}
VECTOR_RESULT {"actions": Batch, "errors": [[slot, detail]]}  # action rows: the slots without an error, in order
```

- The server applies ends, then starts, then observations; restarting a busy slot ends it first.
- `ends` carry `EPISODE_END` payloads with `status` and `failure_reason`; a final step may carry ends only.
- HELLO advertises `capabilities.vector_step` (`ModelServer.supports_vector_step`); the orchestrator refuses a
  server without it.  Other messages are unchanged.

## Server contract (rows)

- Every running episode is a row, a vector slot or a single-episode connection alike: `serve` calls
  `on_episode_start(rows)`, `on_observation(obs, rows)`, `on_episode_end(rows, results)`; the server answers
  with `send_action(rows, actions)` / `send_error(rows, exc)`.
- `rows.ids` are stable while an episode runs and reused after it ends; a reply through a stale row is dropped.
  `Row = (id, step, is_first, episode)`; `episode.chunk_size` may be set per episode in `reset_episodes`.
- `PredictModelServer`: override `predict(obs, row)` or `predict_batch(obs, rows)` (columns in, columns out),
  plus `reset_episodes` / `finish_episodes`; knobs `chunk_size`, `max_batch_size`, `max_wait_time`, `max_rows`.
- The framework slices by `max_batch_size`, serves chunks from per-row buffers, and isolates a row's failure.
  A vector step never waits `max_wait_time`; single-episode requests coalesce in the same dispatcher.
- All hooks run on one model thread.  `LiveModelServer` (continuous inference) has `supports_vector_step = False`.

## Failures

| Failure | Episodes that end | Then |
|---------|-------------------|------|
| `reset` raises | those starting (`env_start`) and all running | next items |
| `step` raises | all running (`env_step`; `timeout` for `TimeoutError`) | next items |
| `result(slot)` raises | that slot's (`env_step`) | next items |
| a slot's error in `VECTOR_RESULT` | that slot's (`model_act`) | continue |
| round trip fails (closed, timeout, malformed reply) | all running, after a reconnect | new sessions |
| reconnect fails | all held | partial result; items stay queued for a rerun |

`RecordingStore.claim(..., exclude=...)` never re-hands a shard's running items; an item finishes in the
queue only after its episode is recorded.  `--requeue-unhealthy` also releases in-flight items.

## Measured

Harness cost per step (zero-cost env and model, 64-float state, server in its own process):

| `num_envs` | connection per slot | per-slot rows, one message | columns, `predict` | columns, `predict_batch` |
|---:|---:|---:|---:|---:|
| 1 | 2.2 ms | 1.7 ms | 3.0 ms | 2.6 ms |
| 64 | 77 ms | 7.1 ms | 3.5 ms | 2.6 ms |
| 512 | 533 ms | 32 ms | 9.7 ms | 4.0 ms |
| 4500 | — | 188 ms | 33 ms | 13 ms |

Columns leave ~2-4 ms fixed (thread hops) plus ~2 µs per environment.  PNG costs ~10 ms per 224×224 frame on
the event loop; `server.image_format: raw` sends images lossless and unencoded.

End to end (A100, hold-style server):

| Benchmark | Before | After |
|-----------|-------:|------:|
| Shortstop (mjlab), 32 worlds × 430 steps, 3 cameras | 231 s (connection per slot) | 72 s (columns, `raw`), outcomes identical |
| OCBench (MuJoCo Warp), ms/step at 500 / 4500 worlds | own loop 31.9 / 71.2 | 36.1 / 83.2 (+7 to +17 %) |
| RoboDojo (Isaac Sim 5.1), stack_blocks | 1.03 env-steps/s | 10.6 (8 envs, `raw`) |

- RoboDojo: with `num_envs > 1` wrist frames render ~79 % black (upstream reproduces it; untested on a GPU with
  the RT cores Isaac Sim 5.1 requires), so its vector scores are not comparable to `num_envs=1` yet.
- RoboTwin 2.0: no adapter; SAPIEN 3 serialises ray tracing and OIDN on one queue, so envs never overlap.

## Alternatives (rejected)

- One connection per slot: harness cost grows ~1 ms per environment.
- Per-slot rows in one message: 188 ms a step at 4500 environments, all per-slot Python.
- Per-slot benchmark views behind a barrier: one connection per slot plus barrier and timeout handling.
- Threads around `StepBenchmark`: cannot share one batched simulator call.

## Limitations

- Lockstep: each step waits for the slowest slot; without `partial_reset` early finishers idle until the wave ends.
- With `partial_reset`, `reset` blocks every running slot, and chunked slots fall out of phase (smaller batches).
- `elapsed_sec` includes other slots' resets and inference; not comparable with the one-at-a-time runner.
- `predict`-only servers pay ~5-20 µs per environment-step for the per-row split; `predict_batch` removes it.
- `Benchmark` and `VectorStepBenchmark` are siblings, so the orchestrator keeps two loops.
