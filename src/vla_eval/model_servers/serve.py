"""WebSocket server runner for ModelServer instances.

Each episode is a model-server row: ``EPISODE_START``..``EPISODE_END`` of a connection, or a vector slot's
``starts``..``ends``.  An ``OBSERVATION`` (``VECTOR_STEP``) is answered once all its rows are; a live episode's
replies go out as they come.  Failures send an ``ERROR`` (per slot) and keep the connection; a disconnect ends its
episodes (``status: aborted``).

HTTP control plane:
    ``GET /health`` returns ``200 {"status": "ok"}`` once ``__init__`` has
    finished — use this as the readiness probe.
    ``GET /config`` returns the current server configuration as JSON.
    ``GET /config?max_batch_size=8`` updates whitelisted parameters and
    returns the applied values.  This allows tools like ``bench_supply.py``
    to sweep ``max_batch_size`` without restarting the server.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import partial
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlparse

import anyio
import numpy as np
from anyio.abc import TaskGroup, TaskStatus
from anyio.to_thread import run_sync as _run_in_thread
import websockets

# Import the implementation directly rather than through the top-level ``websockets.serve``
# alias.  That alias resolves to ``websockets.legacy.server`` on <=13.x and to this module
# from 14.0 on, and the two call ``process_request`` with different arguments: the legacy
# one passes ``(path, Headers)``, this one passes ``(connection, Request)``.  Naming the
# module keeps one calling convention on every version the project supports.
from websockets.asyncio.server import serve as ws_serve

from vla_eval import columns
from vla_eval.model_servers.base import ModelServer, _Pending
from vla_eval.protocol.messages import Message, MessageType, make_hello_payload, pack_message, unpack_message

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound work (msgpack/base64 decoding).
# Default anyio limit (40) is too low when 50+ shards send observations concurrently.
_DECODE_LIMITER = anyio.CapacityLimiter(max(128, (os.cpu_count() or 8) * 16))

# Attributes that can be changed at runtime via GET /config?key=value.
_CONFIG_WHITELIST: dict[str, type] = {
    "max_batch_size": int,
    "max_wait_time": float,
}

# ---------------------------------------------------------------------------
# Backpressure monitoring — shared across all connections
# ---------------------------------------------------------------------------
_inflight: int = 0
_BACKPRESSURE_CHECK_INTERVAL: float = 5.0  # seconds between checks
_BACKPRESSURE_COOLDOWN: float = 30.0  # minimum seconds between warnings


_RESTARTED = {"status": "aborted", "failure_reason": None}  # started over without EPISODE_END
_GONE = {"status": "aborted", "failure_reason": "connection_closed"}


async def _handle_connection(ws: Any, model_server: ModelServer, tg: TaskGroup) -> None:
    """Handle one connection's messages in order, so an episode has at most one observation in flight."""
    global _inflight
    session_id = str(uuid.uuid4())
    row: int | None = None  # the connection's own episode
    live = False  # its replies go out as they come (EPISODE_START mode "live")
    slots = np.full(0, -1, np.int64)  # a vector benchmark's slot -> row (-1: none)
    closed = False
    waiting: _Pending | None = None

    async def watch() -> None:  # a disconnect wakes the observation waiting for replies
        nonlocal closed
        await ws.wait_closed()
        closed = True
        if waiting is not None:
            waiting.done.set()

    async def wait(pending: _Pending) -> bool:
        """Whether every row of ``pending`` was answered (else the connection closed)."""
        nonlocal waiting
        if pending.left and not closed:
            waiting = pending
            try:
                await pending.done.wait()
            finally:
                waiting = None
        return not pending.left

    async def end(result: dict[str, Any]) -> None:
        nonlocal row
        if row is not None:
            ended, row = row, None
            await model_server._end_episodes([ended], [result])

    async def reply(msg_type: MessageType, payload: dict[str, Any], seq: int) -> None:
        await ws.send(pack_message(Message(type=msg_type, payload=payload, seq=seq)))

    def forwarder(seq: int) -> Any:
        async def forward(action: dict[str, Any] | None, error: Exception | None) -> None:
            if error is None:
                try:
                    data = pack_message(Message(type=MessageType.ACTION, payload=action or {}, seq=seq))
                except Exception as exc:
                    error = exc
                else:
                    try:
                        await ws.send(data)
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    return
            logger.error("Error in act session=%s: %s: %s", session_id[:8], type(error).__name__, error)
            detail = f"observation_failed: {type(error).__name__}: {error}"
            await _try_reply(reply, MessageType.ERROR, {"error": detail}, seq)

        return forward

    tg.start_soon(watch)
    logger.info("Client connected: session=%s", session_id)
    _msg_count = 0
    try:
        async for raw_data in ws:
            _msg_count += 1
            msg = await _run_in_thread(partial(unpack_message, raw_data), limiter=_DECODE_LIMITER)

            if msg.type == MessageType.HELLO:
                extra: dict[str, Any] = {}
                obs_params = model_server.get_observation_params()
                if obs_params:
                    extra["observation_params"] = obs_params
                try:
                    action_spec = model_server.get_action_spec()
                    obs_spec = model_server.get_observation_spec()
                    if action_spec:
                        extra["action_spec"] = {k: v.to_dict() for k, v in action_spec.items()}
                    if obs_spec:
                        extra["observation_spec"] = {k: v.to_dict() for k, v in obs_spec.items()}
                except NotImplementedError:
                    pass
                reply_payload = make_hello_payload(
                    model_server=type(model_server).__name__,
                    capabilities={"vector_step": model_server.supports_vector_step},
                    **extra,
                )
                await reply(MessageType.HELLO, reply_payload, msg.seq)
                logger.info(
                    "HELLO session=%s client=%s server=%s",
                    session_id[:8],
                    msg.payload.get("harness_version"),
                    reply_payload["harness_version"],
                )

            elif msg.type == MessageType.EPISODE_START:
                await end(_RESTARTED)  # the client may start over without ending
                (out,) = await model_server._start_episodes([(msg.payload, session_id)])
                if isinstance(out, Exception):
                    detail = f"episode_start_failed: {type(out).__name__}: {out}"
                    await _try_reply(reply, MessageType.ERROR, {"error": detail}, msg.seq)
                    continue
                row, live = out, msg.payload.get("mode") == "live"
                logger.info("EPISODE_START session=%s row=%d", session_id[:8], row)

            elif msg.type == MessageType.OBSERVATION:
                if row is None:  # an observation without EPISODE_START: an episode of its own
                    (out,) = await model_server._start_episodes([({}, session_id)])
                    if isinstance(out, Exception):
                        await _try_reply(reply, MessageType.ERROR, {"error": f"observation_failed: {out}"}, msg.seq)
                        continue
                    row, live = out, False
                assert row is not None  # set above; the nested end() defeats type narrowing
                obs, ids = columns.stack([msg.payload]), [row]
                _inflight += 1
                try:
                    if live:  # each reply is sent on as it comes
                        await model_server._observe(obs, ids, _Pending(1, forward=forwarder(msg.seq)))
                        continue
                    pending = _Pending(1)
                    await model_server._observe(obs, ids, pending)
                    if not await wait(pending):  # closed: still read what arrived, e.g. its EPISODE_END
                        continue
                finally:
                    _inflight -= 1
                if pending.errors:
                    exc = pending.errors[0]
                    logger.error("Error in act session=%s: %s: %s", session_id[:8], type(exc).__name__, exc)
                    detail = f"observation_failed: {type(exc).__name__}: {exc}"
                    await _try_reply(reply, MessageType.ERROR, {"error": detail}, msg.seq)
                    continue
                try:
                    await reply(MessageType.ACTION, columns.split(pending.result(), 1)[0], msg.seq)
                except websockets.exceptions.ConnectionClosed:
                    continue

            elif msg.type == MessageType.VECTOR_STEP:
                if not model_server.supports_vector_step:
                    detail = f"vector_step_unsupported: {type(model_server).__name__}"
                    await _try_reply(reply, MessageType.ERROR, {"error": detail}, msg.seq)
                    continue
                _inflight += 1
                try:
                    result, slots = await _vector_step(msg.payload, slots, session_id, model_server, wait)
                finally:
                    _inflight -= 1
                if result is None:
                    continue
                try:
                    await reply(MessageType.VECTOR_RESULT, result, msg.seq)
                except websockets.exceptions.ConnectionClosed:
                    continue

            elif msg.type == MessageType.EPISODE_END:
                logger.info("EPISODE_END session=%s", session_id[:8])
                await end(msg.payload)

            elif msg.type == MessageType.ERROR:
                logger.error("Client error: %s", msg.payload)

        logger.info("Client disconnected: session=%s msgs=%d", session_id[:8], _msg_count)
    except websockets.exceptions.ConnectionClosed as exc:
        close_code = exc.rcvd.code if exc.rcvd else None
        close_reason = exc.rcvd.reason if exc.rcvd else None
        logger.info(
            "Client disconnected: session=%s code=%s reason=%s msgs=%d",
            session_id[:8],
            close_code,
            close_reason,
            _msg_count,
        )
    except Exception:
        logger.exception("Error handling session=%s msgs=%d", session_id[:8], _msg_count)
    finally:
        open_rows = [int(r) for r in slots if r >= 0]
        with anyio.CancelScope(shield=True):
            await end(_GONE)
            if open_rows:
                await model_server._end_episodes(open_rows, [_GONE] * len(open_rows))


async def _try_reply(reply: Any, msg_type: MessageType, payload: dict[str, Any], seq: int) -> None:
    try:
        await reply(msg_type, payload, seq)
    except Exception:
        pass


async def _vector_step(
    payload: dict[str, Any],
    slots: np.ndarray,
    session_id: str,
    model_server: ModelServer,
    wait: Callable[[_Pending], Awaitable[bool]],
) -> tuple[dict[str, Any] | None, np.ndarray]:
    """One VECTOR_STEP (ends, then starts, then observations); ``slots[s]``: slot ``s``'s row, -1 if none.
    Returns the VECTOR_RESULT payload (actions of the slots that did not fail, in order; ``None`` if the
    connection closed meanwhile) and the new ``slots``."""
    errors: dict[int, str] = {}
    starts = list(dict(payload.get("starts", [])).items())  # one per slot
    ends = {**{s: _RESTARTED for s, _ in starts}, **dict(payload.get("ends", []))}  # restarting a busy slot ends it
    ends = {s: r for s, r in ends.items() if s < len(slots) and slots[s] >= 0}
    if ends:
        await model_server._end_episodes([int(slots[s]) for s in ends], list(ends.values()))
        for s in ends:
            slots[s] = -1
    if starts:
        top = max(s for s, _ in starts) + 1
        if top > len(slots):
            slots = np.concatenate([slots, np.full(top - len(slots), -1, np.int64)])
        for s, out in zip(
            [s for s, _ in starts],
            await model_server._start_episodes([(config, f"{session_id}:{s}") for s, config in starts]),
        ):
            if isinstance(out, Exception):
                errors[s] = f"episode_start_failed: {type(out).__name__}: {out}"
            else:
                slots[s] = out
    order = np.asarray(payload.get("slots", []), np.int64)
    obs: dict[str, Any] = payload.get("observations", {})
    known = (order >= 0) & (order < len(slots))
    rows = np.where(known, slots[np.where(known, order, 0)] if len(slots) else -1, -1)
    if (rows < 0).any():
        keep = np.flatnonzero(rows >= 0)
        for s in order[rows < 0].tolist():
            errors.setdefault(s, f"observation_failed: slot {s} has no running episode")
        obs, order, rows = columns.take(obs, keep), order[keep], rows[keep]
    actions: dict[str, Any] = {}
    if len(rows):
        pending = _Pending(len(rows), vector=True)
        await model_server._observe(obs, rows, pending)
        if not await wait(pending):
            return None, slots
        actions = pending.result()
        for i, exc in pending.errors.items():
            errors[int(order[i])] = f"observation_failed: {type(exc).__name__}: {exc}"
    return {"actions": actions, "errors": [[s, e] for s, e in errors.items()]}, slots


def _make_process_request(model_server: ModelServer) -> Any:
    """Create a ``process_request`` callback that serves the HTTP control plane.

    When the request path matches one of the supported routes, the callback
    returns an HTTP response instead of proceeding with the WebSocket handshake:

    - ``GET /health`` — returns ``200 {"status": "ok"}``. Reaches this point
      only after ``__init__`` returned, so it is a true readiness signal.
    - ``GET /config`` — returns current whitelisted attribute values as JSON.
    - ``GET /config?max_batch_size=8`` — updates the attribute(s) and returns
      the applied values.

    Unknown keys are ignored (logged as warning).  Type conversion errors
    return 422.
    """

    def process_request(connection: Any, request: Any) -> Any:
        parsed = urlparse(request.path)
        if parsed.path == "/health":
            resp = connection.respond(HTTPStatus.OK, '{"status": "ok"}\n')
            # respond() defaults Content-Type to text/plain; replace with JSON.
            del resp.headers["Content-Type"]
            resp.headers["Content-Type"] = "application/json"
            return resp
        if parsed.path != "/config":
            return None  # proceed with WebSocket handshake

        params = parse_qs(parsed.query)
        applied: dict[str, Any] = {}
        errors: list[str] = []

        for key, values in params.items():
            if key not in _CONFIG_WHITELIST:
                logger.warning("GET /config: unknown key %r ignored", key)
                errors.append(f"unknown key: {key}")
                continue
            if not hasattr(model_server, key):
                errors.append(f"server has no attribute: {key}")
                continue
            cast = _CONFIG_WHITELIST[key]
            try:
                value = cast(values[-1])  # last value wins
                if not (value > 0 if cast is int else math.isfinite(value) and value >= 0):
                    raise ValueError(f"{value} is out of range")
            except (ValueError, TypeError) as exc:
                errors.append(f"bad value for {key}: {exc}")
                continue
            setattr(model_server, key, value)
            applied[key] = value

        if errors and not applied:
            body = json.dumps({"errors": errors})
            return connection.respond(HTTPStatus.UNPROCESSABLE_ENTITY, body + "\n")

        # Build response: applied changes + current values
        current = {}
        for key in _CONFIG_WHITELIST:
            if hasattr(model_server, key):
                current[key] = getattr(model_server, key)
        body = json.dumps({"applied": applied, "config": current})
        if errors:
            body = json.dumps({"applied": applied, "config": current, "errors": errors})
        logger.info("GET /config applied=%s current=%s", applied, current)
        return connection.respond(HTTPStatus.OK, body + "\n")

    return process_request


async def _backpressure_monitor(threshold: int) -> None:
    """Periodically warn if in-flight observation count is high."""
    last_warning = 0.0
    while True:
        await anyio.sleep(_BACKPRESSURE_CHECK_INTERVAL)
        if _inflight >= threshold:
            now = time.monotonic()
            if now - last_warning >= _BACKPRESSURE_COOLDOWN:
                last_warning = now
                logger.warning(
                    "Backpressure detected — %d observations in-flight for inference. "
                    "Model server throughput may be insufficient for the current shard count.",
                    _inflight,
                )


async def serve_async(
    model_server: ModelServer,
    host: str = "0.0.0.0",
    port: int = 8000,
    backpressure_threshold: int = 4,
    *,
    on_shutdown: Callable[[], Any] | None = None,
    task_status: TaskStatus[int] = anyio.TASK_STATUS_IGNORED,
) -> None:
    """Start a WebSocket server wrapping the given ModelServer. Reports the bound port via *task_status*;
    calls *on_shutdown* when cancelled, before waiting for the calls in flight."""
    logger.info("Starting model server on ws://%s:%d", host, port)
    logger.info("HTTP config endpoint at http://%s:%d/config", host, port)

    process_request = _make_process_request(model_server)
    async with anyio.create_task_group() as tg:

        async def handler(ws: Any) -> None:
            await _handle_connection(ws, model_server, tg)

        await model_server.on_serve_start(tg)
        tg.start_soon(_backpressure_monitor, backpressure_threshold)
        async with ws_serve(
            handler,
            host,
            port,
            process_request=process_request,
            compression=None,  # disable deflate; unnecessary for binary payloads and costly under high concurrency
            max_size=None,  # observations with images can exceed the 1MB default
            ping_interval=None,  # disable keepalive pings; JIT warmup can hold the GIL for 20s+
        ) as server:
            task_status.started(server.sockets[0].getsockname()[1])
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                if on_shutdown is not None:
                    on_shutdown()
                raise


def serve(
    model_server: ModelServer,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start a WebSocket server (blocking). Entry point for model server scripts.

    Ctrl-C finishes the call in flight and the open episodes' cleanup; a second Ctrl-C exits at once."""

    def arm() -> None:
        logger.info("Shutting down after the calls in flight; Ctrl-C again to exit now")
        signal.signal(signal.SIGINT, lambda *_: os._exit(130))

    previous = signal.getsignal(signal.SIGINT)
    try:
        anyio.run(partial(serve_async, model_server, host, port, on_shutdown=arm))
    finally:
        signal.signal(signal.SIGINT, previous)
        model_server._model_thread.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------
# run_server: auto-argparse entrypoint for model server scripts
# ---------------------------------------------------------------------------


def _parse_address(address: str, default_host: str = "0.0.0.0", default_port: int = 8000) -> tuple[str, int]:
    """Parse ``host:port`` string. Raises ``ValueError`` on bad port."""
    parts = address.rsplit(":", 1)
    host = parts[0] or default_host
    port = default_port
    if len(parts) == 2:
        try:
            port = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid port in address {address!r}: {parts[1]!r} is not a number") from None
    return host, port


def run_server(server_cls: type[ModelServer]) -> None:
    """Standard entrypoint for model server scripts.

    ``--config <yaml>`` reads the model-server yaml directly via
    ``ActionConfigFile``; the yaml's ``args:`` sub-dict maps to
    ``server_cls.__init__`` kwargs. CLI overrides use ``--args.<key>=<value>``.
    """
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(description=f"{server_cls.__name__} model server")
    parser.add_argument("--config", action=ActionConfigFile, help="Path(s) to YAML config")
    parser.add_argument("--address", default=None, help="Bind address as host:port (e.g. 0.0.0.0:8001)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (prefer --address)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (prefer --address)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    # ``script`` is host-side yaml metadata; absorb here so the same yaml loads.
    parser.add_argument("--script", default=None, help=argparse.SUPPRESS)
    parser.add_class_arguments(server_cls, nested_key="args", fail_untyped=False, sub_configs=True)

    cfg = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    host, port = cfg.host, cfg.port
    if cfg.address:
        host, port = _parse_address(cfg.address, host, port)

    server = server_cls(**cfg.args.as_dict())

    logger.info("Starting server on ws://%s:%d", host, port)
    serve(server, host=host, port=port)
