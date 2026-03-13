# RFC-0007: BatchPredictModelServer Implementation

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2026-02-23
- **Requires:** RFC-0003, RFC-0006
- **Superseded-By:** —

## Summary

Implementation design for `BatchPredictModelServer`, whose external interface is defined in RFC-0003. This RFC covers the internal queuing, dispatch loop, and per-session chunk buffer management that coalesce N concurrent observations into batched GPU inference.

## Problem

With episode sharding (RFC-0006), N shard processes connect to a single model server via WebSocket. Each observation triggers an independent `predict()` call, serialized on the GPU. For a model like Pi0 with 50-step action chunks, the GPU is idle most of the time while chunks drain. Batching N observations into a single `predict_batch()` call gives up to N× throughput improvement.

## Design

### Internal `_BatchQueue`

A simple async queue holding `(obs, ctx, asyncio.Future)` tuples. Each `on_observation` call that needs inference enqueues a tuple and awaits the future. The future is resolved by the dispatch loop after `predict_batch()` returns.

### Background `_dispatch_loop`

A single `asyncio.Task` coroutine that runs for the lifetime of the server:

```python
async def _dispatch_loop(self):
    while True:
        # Wait for at least one item
        item = await self._queue.get()
        batch = [item]

        # Collect more items up to max_batch_size or max_wait_time
        deadline = asyncio.get_event_loop().time() + self.max_wait_time
        while len(batch) < self.max_batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # Dispatch batch to executor
        obs_batch = [b[0] for b in batch]
        ctx_batch = [b[1] for b in batch]
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, self.predict_batch, obs_batch, ctx_batch
            )
            for (_, _, fut), result in zip(batch, results):
                fut.set_result(result)
        except Exception as exc:
            for _, _, fut in batch:
                fut.set_exception(exc)
```

Key properties:
- Blocks on the first item (zero CPU when idle).
- After the first item arrives, collects up to `max_batch_size` with `max_wait_time` as the deadline.
- `predict_batch()` runs in the default executor (thread pool), so it doesn't block the event loop.
- On exception, all futures in the batch receive the error.

### `on_observation` Flow

```python
async def on_observation(self, obs, ctx):
    # 1. Check chunk buffer — serve cached action if available
    if self.chunk_size > 1:
        buf = self._chunk_buffers.get(ctx.session_id)
        if buf and not buf.empty:
            await ctx.send_action({"actions": buf.pop()})
            return

    # 2. Lazily start dispatch loop
    if self._dispatch_task is None:
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    # 3. Enqueue and await
    fut = asyncio.get_running_loop().create_future()
    await self._queue.put((obs, ctx, fut))
    result = await fut

    # 4. Handle chunk buffer (identical to PredictModelServer)
    actions = result.get("actions")
    if self.chunk_size > 1 and actions is not None and np.asarray(actions).ndim > 1:
        buf = self._get_or_create_buffer(ctx.session_id)
        buf.push_chunk(np.asarray(actions))
        await ctx.send_action({"actions": buf.pop()})
    else:
        await ctx.send_action(result)
```

The dispatch loop starts lazily on the first `on_observation` call, not at construction time, so the server can be instantiated before the event loop is running.

### Per-Session ActionChunkBuffer

Identical to `PredictModelServer`: a `dict[str, ActionChunkBuffer]` keyed by `session_id`. Buffers are created on first use and cleared on `on_episode_start`. The `ActionChunkBuffer` and `get_ensemble_fn` from `model_servers/chunking.py` are reused directly.

## Interaction with Sharding

```
Shard 0 ──WebSocket──┐
Shard 1 ──WebSocket──┤                    ┌──────────────┐
Shard 2 ──WebSocket──┼── on_observation ──►  _BatchQueue  ──► _dispatch_loop ──► predict_batch()
Shard 3 ──WebSocket──┘                    └──────────────┘         │
                                                                   ▼
                                              futures resolved ◄── results
```

N shard processes produce N concurrent WebSocket connections. The server's async handler calls `on_observation` for each, which enqueues into the shared `_BatchQueue`. The dispatch loop coalesces up to `max_batch_size` observations and calls `predict_batch()` once.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_batch_size` | `1` | Upper bound on batch size. Set to N (number of shards) for full utilization. |
| `max_wait_time` | `0.01` | Seconds to wait for a full batch before dispatching a partial one. Latency/throughput tradeoff. |
| `chunk_size` | `1` | Action chunk length. Same semantics as `PredictModelServer`. |
| `action_ensemble` | `"newest"` | Ensemble strategy for overlapping chunks. |

## File Location

`src/vla_eval/model_servers/predict.py` (merged into `PredictModelServer`; no separate `batch.py` file).

## Testing Strategy

- Mock `predict_batch` with a controlled delay (`time.sleep`).
- Verify that N concurrent `on_observation` calls are coalesced into a single `predict_batch` call with N items.
- Verify `max_wait_time` triggers partial batch dispatch.
- Verify chunk buffer serves cached actions without enqueuing.
- Verify exception in `predict_batch` propagates to all awaiting callers.



