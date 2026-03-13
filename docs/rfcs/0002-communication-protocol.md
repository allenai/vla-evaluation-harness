# RFC-0002: Communication Protocol

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2025-02-22
- **Requires:** RFC-0001
- **Superseded-By:** —

## Summary

Defines the wire protocol between benchmark clients and model servers: transport, message framing, serialization, and the client-side `Connection` API. The protocol must support both synchronous (lock-step obs→action) and real-time (streaming, non-blocking) evaluation modes over a single connection.

## Transport

- **WebSocket + msgpack** async communication.
- **Single WebSocket connection** (`ws://server:port`). Message `type` field multiplexes logical channels — no need for multiple connections or HTTP endpoints.
- **Serialization**: msgpack with a custom numpy codec (see below). Arbitrary `dict[str, Any]` payloads including numpy arrays are serialized transparently.
- **Image encoding**: Configurable per deployment — raw numpy (default), JPEG, or PNG.
- **Compression**: Default `None`. On LAN the compression CPU overhead exceeds network savings. WAN deployments can enable compression via configuration.

## Message Schema

Every message is a single msgpack binary WebSocket frame:

```python
{
    "type": "observation" | "action" | "episode_start" | "episode_end" | "error",
    "payload": dict[str, Any],
    "seq": int,
    "timestamp": float,
}
```

| Field | Description |
|-------|-------------|
| `type` | Discriminator that determines how the payload is interpreted. |
| `payload` | Arbitrary dict. Contents are benchmark-specific (see Payload Convention). |
| `seq` | Monotonically increasing sequence number. In sync mode the server's action `seq` must match the client's observation `seq`, enforcing 1:1 pairing. In real-time mode it is used for ordering and debugging. |
| `timestamp` | Wall-clock Unix epoch (float seconds) at message creation. Used for latency measurement and real-time metrics (p50/p95/p99 inference latency, stale-action ratio). |

Implemented in `protocol/messages.py` as the `Message` dataclass with `pack_message` / `unpack_message` helpers.

## numpy Serialization

numpy arrays are encoded inline as msgpack dicts by the custom codec in `protocol/numpy_codec.py`:

```python
{"__ndarray__": True, "data": bytes, "dtype": str, "shape": tuple}
```

- `data`: raw bytes from `array.tobytes()`.
- `dtype`: numpy dtype string (e.g. `"float32"`, `"uint8"`).
- `shape`: integer tuple.

**Security**: On deserialization the codec rejects `Void`, `Object`, and `Complex` dtypes — these can trigger arbitrary code execution via pickle. Only `bool`, `int*`, `uint*`, and `float*` kinds are allowed.

Scalar numpy types (`np.integer`, `np.floating`, `np.bool_`) are converted to Python builtins during encoding so they survive msgpack round-trips.

## Payload Convention

Observations and actions are `dict[str, Any]` — the framework imposes **no fixed schema**. Each benchmark defines its own structure via `Benchmark.make_obs()`.

The recommended (not enforced) convention, based on the [RLinf](../proposal/03_REFERENCE_SURVEY/rlinf.md) field set adopted by starVLA / InternVLA-M1 / dexbotic:

| Field | Type | Description |
|-------|------|-------------|
| `images` | `dict[str, np.ndarray]` | Camera name → HWC uint8 image. e.g. `{"main": ..., "wrist": ...}` |
| `states` | `np.ndarray` | Proprioception (joint angles, end-effector pose, etc.) |
| `task_description` | `str` | Natural-language task instruction |

Following the convention enables a single model server implementation to work across multiple benchmarks without per-benchmark key mapping. Not following it is fine — the protocol layer only requires `dict[str, Any]`.


## Connection Client API

The framework provides `Connection` — a client library used by benchmarks and episode runners to communicate with the model server. No user-side ABC implementation is needed.

```python
conn = vla_eval.connect("ws://model-server:8000")
```

```python
class Connection:
    # Episode lifecycle
    async def start_episode(self, config: dict[str, Any]) -> None: ...
    async def end_episode(self, result: dict[str, Any]) -> None: ...

    # Sync mode — send observation, await action
    async def act(self, obs: dict[str, Any]) -> dict[str, Any]: ...

    # Real-time mode — non-blocking send + action callback
    async def send_observation(self, obs: dict[str, Any]) -> None: ...
    def on_action(self, callback: Callable[[dict[str, Any]], None]) -> None: ...
```

Key design points:

- **`act()` uses `asyncio.Future`** internally — it awaits the server's action response without busy-polling.
- **Sync mode**: `act()` sends an observation and returns the matching action (matched by `seq`).
- **Real-time mode**: `send_observation()` fires and forgets; incoming actions are dispatched to the registered `on_action` callback. The `AsyncEpisodeRunner` uses this with an `ActionBuffer` and a hold policy.
- **Sequence numbering** is managed internally; callers never set `seq` manually.
- **Context manager** support (`async with Connection(...) as conn`) for automatic connect/close.

Implemented in `connection.py`.

## Implementation Status

| Component | Module | Status |
|-----------|--------|--------|
| Message types & (de)serialization | `protocol/messages.py` | ✅ Implemented |
| numpy codec with dtype validation | `protocol/numpy_codec.py` | ✅ Implemented |
| Connection client (sync + realtime API) | `connection.py` | ✅ Implemented |
| Server-side WebSocket handler | `model_servers/serve.py` | ✅ Implemented |
| Image encoding (JPEG/PNG/raw) | `protocol/image_codec.py` | ✅ Implemented |
| WAN compression option | — | 🔜 Not yet implemented |



