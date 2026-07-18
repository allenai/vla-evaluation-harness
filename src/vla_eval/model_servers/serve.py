"""WebSocket server runner for ModelServer instances.

Session lifecycle:
    1. Each WebSocket connection gets a unique ``session_id`` (UUID) that
       persists across episodes within that connection.
    2. On ``EPISODE_START``, a new ``SessionContext`` is created (new
       ``episode_id``, step counter reset to 0), but ``session_id`` is reused.
    3. On ``OBSERVATION``, ``on_observation()`` is called, then the step
       counter increments.  Inside ``predict()``, ``ctx.step`` reflects the
       count *before* the current observation.
    4. On ``EPISODE_END``, ``on_episode_end()`` is called.

Error handling:
    Exceptions in ``on_observation()`` send an ``ERROR`` message to the client
    and log the traceback, but do **not** close the connection.

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
import os
import time
import uuid
from functools import partial
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlparse

import anyio
from anyio.to_thread import run_sync as _run_in_thread
import websockets

from vla_eval.model_servers.base import ModelServer, SessionContext
from vla_eval.types import Action
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


async def _handle_connection(
    ws: Any,
    model_server: ModelServer,
) -> None:
    """Handle a single WebSocket client connection."""
    global _inflight
    session_id = str(uuid.uuid4())
    episode_id = ""
    ctx = SessionContext(session_id=session_id, episode_id="", mode="sync")
    # Track the seq of the observation currently being processed so that
    # send_action echoes it back, matching the protocol contract.
    _current_obs_seq: list[int] = [0]  # mutable container for closure

    async def send_action(action: Action) -> None:
        msg = Message(type=MessageType.ACTION, payload=action, seq=_current_obs_seq[0])
        await ws.send(pack_message(msg))

    ctx._send_action_fn = send_action

    logger.info("Client connected: session=%s", session_id)
    _msg_count = 0
    in_episode = False
    try:
        async for raw_data in ws:
            _msg_count += 1
            msg = await _run_in_thread(partial(unpack_message, raw_data), limiter=_DECODE_LIMITER)

            if msg.type == MessageType.HELLO:
                obs_params = model_server.get_observation_params()
                extra: dict[str, Any] = {}
                if obs_params:
                    extra["observation_params"] = obs_params
                model_metadata = model_server.get_metadata()
                if model_metadata:
                    extra["model_metadata"] = model_metadata
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
                    capabilities={},
                    **extra,
                )
                reply = Message(type=MessageType.HELLO, payload=reply_payload, seq=msg.seq)
                await ws.send(pack_message(reply))
                logger.info(
                    "HELLO session=%s client=%s server=%s",
                    session_id[:8],
                    msg.payload.get("harness_version"),
                    reply_payload["harness_version"],
                )
                continue

            elif msg.type == MessageType.EPISODE_START:
                # `recording` carries (sid, eid, eval_id, db_path) so model-server-side
                # code can open a StepRecorder against the same SQLite the harness uses.
                rec = msg.payload.get("recording") or {}
                effective_sid = rec.get("sid") or session_id
                episode_id = rec.get("eid") or str(uuid.uuid4())
                ctx = SessionContext(
                    session_id=effective_sid,
                    episode_id=episode_id,
                    mode="sync",
                    eval_id=rec.get("eval_id") or "",
                    recording_db_path=rec.get("db_path") or "",
                )
                ctx._send_action_fn = send_action
                logger.info("EPISODE_START session=%s episode=%s", effective_sid[:8], episode_id[:8])
                try:
                    await model_server.on_episode_start(msg.payload, ctx)
                    in_episode = True
                except Exception as exc:
                    logger.exception("Error in on_episode_start session=%s", session_id[:8])
                    error_detail = f"episode_start_failed: {type(exc).__name__}: {exc}"
                    error_msg = Message(type=MessageType.ERROR, payload={"error": error_detail}, seq=msg.seq)
                    try:
                        await ws.send(pack_message(error_msg))
                    except Exception:
                        pass

            elif msg.type == MessageType.OBSERVATION:
                _current_obs_seq[0] = msg.seq
                _inflight += 1
                try:
                    await model_server.on_observation(msg.payload, ctx)
                except websockets.exceptions.ConnectionClosed:
                    logger.info(
                        "Connection closed during on_observation session=%s step=%d",
                        session_id[:8],
                        ctx.step,
                    )
                    return  # connection gone, exit handler
                except Exception as exc:
                    logger.exception(
                        "Error in on_observation session=%s step=%d",
                        session_id[:8],
                        ctx.step,
                    )
                    error_detail = f"observation_failed: {type(exc).__name__}: {exc}"
                    error_msg = Message(type=MessageType.ERROR, payload={"error": error_detail}, seq=msg.seq)
                    try:
                        await ws.send(pack_message(error_msg))
                    except Exception:
                        pass
                    continue
                finally:
                    _inflight -= 1
                ctx._increment_step()

            elif msg.type == MessageType.EPISODE_END:
                logger.info("EPISODE_END session=%s", session_id[:8])
                try:
                    await model_server.on_episode_end(msg.payload, ctx)
                except Exception:
                    logger.exception("Error in on_episode_end session=%s", session_id[:8])
                in_episode = False

            elif msg.type == MessageType.ERROR:
                logger.error("Client error: %s", msg.payload)

        # Loop exited normally (websockets v16+ exits iterator on close)
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
        if in_episode:
            try:
                await model_server.on_episode_end({}, ctx)
            except Exception:
                logger.exception("Error in cleanup on_episode_end session=%s", session_id[:8])


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
) -> None:
    """Start a WebSocket server wrapping the given ModelServer."""
    logger.info("Starting model server on ws://%s:%d", host, port)
    logger.info("HTTP config endpoint at http://%s:%d/config", host, port)

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, model_server)

    process_request = _make_process_request(model_server)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_backpressure_monitor, backpressure_threshold)
        async with websockets.serve(
            handler,
            host,
            port,
            process_request=process_request,
            compression=None,  # disable deflate; unnecessary for binary payloads and costly under high concurrency
            max_size=None,  # observations with images can exceed the 1MB default
            ping_interval=None,  # disable keepalive pings; JIT warmup can hold the GIL for 20s+
        ):
            await anyio.sleep_forever()


def serve(
    model_server: ModelServer,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start a WebSocket server (blocking). Entry point for model server scripts."""
    anyio.run(serve_async, model_server, host, port)


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
