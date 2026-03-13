# RFC-0003: Model Server Hierarchy

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2025-02-22
- **Requires:** RFC-0001
- **Superseded-By:** —

## Summary

Three-level class hierarchy for model servers: an async base (`ModelServer`), a blocking convenience layer (`PredictModelServer`), and a planned batched variant (`BatchPredictModelServer`). The hierarchy lets most researchers implement a single `predict()` method while preserving full async control for advanced use cases.

## Hierarchy

```
ModelServer (ABC, async)
├── PredictModelServer (blocking predict, action chunking)
└── BatchPredictModelServer (dynamic batching)
```

## ModelServer (Base — Async)

```python
class ModelServer(ABC):
    @abstractmethod
    async def on_observation(self, obs: dict[str, Any], ctx: SessionContext) -> None:
        """Run inference and call ctx.send_action()."""

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Reset model state. Optional override."""

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        """Episode cleanup. Optional override."""
```

**Why async?** Real-time evaluation requires observations to arrive during inference. The server must control which observations to process and which to skip. Streaming observation dispatch, async batch inference, and other async patterns use this interface directly.

Use `ModelServer` only when you need explicit control over the async flow.

## PredictModelServer (Blocking Convenience)

```python
class PredictModelServer(ModelServer):
    chunk_size: int = 1
    action_ensemble: str | Callable = "newest"  # "newest" | "average" | "ema" | callable
    ema_alpha: float = 0.5

    @abstractmethod
    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        """Blocking inference. Return {"actions": np.array(...)}.
        Shape (action_dim,) for chunk_size=1, (chunk_size, action_dim) otherwise."""
```

The framework wraps `predict()` in `asyncio.run_in_executor` — the researcher never touches async. When `chunk_size > 1`, the framework manages a per-session `ActionChunkBuffer`: it pops buffered actions on each observation and only calls `predict()` when the buffer is exhausted (or every step for `"average"` ensemble).

## Action Chunking

Action chunking is entirely the server's responsibility. The benchmark client always receives a single action per step.

| Strategy | Behavior |
|----------|----------|
| `"newest"` | New chunk replaces old. Re-infer when exhausted. |
| `"average"` | Infer every step. Uniform temporal ensemble over overlapping actions. |
| `"ema"` | Exponential moving average with `ema_alpha`. Newer chunk weighted higher. |
| `callable` | User function: `(old_actions, new_actions) → ensembled`. |

Callable example — cosine-similarity adaptive ensemble (used in starVLA, dexbotic):

```python
def adaptive_ensemble(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    cos_sim = np.dot(old.flatten(), new.flatten()) / (
        np.linalg.norm(old) * np.linalg.norm(new) + 1e-8
    )
    weight = 0.5 * (1 + cos_sim)
    return weight * old + (1 - weight) * new

class MyCogACTServer(PredictModelServer):
    chunk_size = 16
    action_ensemble = adaptive_ensemble
```

## BatchPredictModelServer

For multi-session evaluation where GPU utilization benefits from batching.

```python
class BatchPredictModelServer(ModelServer):
    max_batch_size: int = 1
    max_wait_time: float = 0.01   # seconds to wait before dispatching partial batch
    chunk_size: int = 1
    action_ensemble: str = "newest"

    @abstractmethod
    def predict_batch(
        self,
        obs_batch: list[dict[str, Any]],
        ctx_batch: list[SessionContext],
    ) -> list[dict[str, Any]]:
        """Batched blocking inference. len(result) == len(obs_batch)."""
```

The framework queues observations across sessions and dispatches `predict_batch()` when `max_batch_size` is reached or `max_wait_time` elapses — whichever comes first. Per-session chunk buffers are managed identically to `PredictModelServer`.

## SessionContext

Framework-created context passed to every callback. Separates what the researcher implements from what the framework provides.

```python
class SessionContext:
    async def send_action(self, action: dict[str, Any]) -> None: ...
    @property
    def session_id(self) -> str: ...
    @property
    def episode_id(self) -> str: ...
    @property
    def mode(self) -> Literal["sync", "realtime"]: ...
    @property
    def step(self) -> int: ...
    @property
    def is_first(self) -> bool: ...
```

In `PredictModelServer`, `send_action` is called automatically from the return value of `predict()`. In raw `ModelServer`, the user calls `ctx.send_action()` explicitly inside `on_observation`.

## Examples

| Model | Class | Config |
|-------|-------|--------|
| OpenVLA | `PredictModelServer` | `chunk_size=1` |
| CogACT | `PredictModelServer` | `chunk_size=16`, `action_ensemble="newest"` |
| CogACT (adaptive) | `PredictModelServer` | `chunk_size=16`, `action_ensemble=adaptive_ensemble` |
| Pi0 | `PredictModelServer` | `chunk_size=50`, `action_ensemble="newest"` |
| Pi0 (batch) | `BatchPredictModelServer` | `max_batch_size=4`, `chunk_size=50` |

## When to Use Which

- **`PredictModelServer`** — default choice. Covers single-step and chunk models in both sync and real-time evaluation.
- **`BatchPredictModelServer`** — multiple environments evaluated concurrently to maximize GPU utilization.
- **`ModelServer`** — only when you need direct async control (custom observation filtering, streaming inference, etc.).

## Implementation Status

- ✅ `ModelServer` ABC (`model_servers/base.py`)
- ✅ `PredictModelServer` with action chunking (`model_servers/predict.py`, `model_servers/chunking.py`)
- ✅ `SessionContext` (`model_servers/base.py`)
- ✅ Server runner — WebSocket wrapper (`model_servers/serve.py`)
- ✅ CogACT reference implementation (`model_servers/dexbotic/cogact.py`)
- ✅ Batched inference via `PredictModelServer` (`max_batch_size > 1`)
- ✅ Real-time observation dispatch (`continuous_inference` mode)

