"""Connection client: communicates with ModelServer over WebSocket."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import anyio
import websockets
from websockets.protocol import State as WebSocketState

from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class Connection:
    """WebSocket client for communicating with a ModelServer.

    Handles observation→action exchange, episode lifecycle messages,
    and automatic reconnection with exponential backoff.

    The class is organised in two tiers:

    **Core primitives** — the irreducible operations everything else builds on:

    - `connect` / `close` — WebSocket lifecycle.
    - `send(msg_type, payload)` — send any message (auto-reconnects, auto-seq).
    - `recv(timeout=...)` — receive the next message.

    **Convenience methods** — thin compositions of the primitives above:

    - `start_episode`, `end_episode`, `send_observation` — one-liner sends.
    - `act` — request/response (send + recv + seq validation).
    - `on_action`, `start_listener`, `stop_listener` — callback-based receive.
    - `reconnect` — close + connect with backoff.

    Timeouts and retries:
        - ``timeout`` (default 30s): max wait for each ``recv()`` call.
        - ``max_retries`` (default 5): reconnection attempts before raising
          ``ConnectionError``.
        - ``backoff_base`` (default 2.0): wait ``backoff_base ** attempt``
          seconds between retries (2s, 4s, 8s, 16s, 32s ≈ 62s total).

    Can be used as an async context manager::

        async with Connection("ws://localhost:8000") as conn:
            await conn.start_episode(task_info)
            action = await conn.act(obs_dict)
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._ws: Any = None
        self._seq: int = 0
        self._action_callback: Callable[[Action], None] | None = None
        self._listener_task: asyncio.Task[None] | None = None

    @property
    def is_connected(self) -> bool:
        """Return True if the WebSocket is open."""
        return self._ws is not None and self._ws.state is WebSocketState.OPEN

    async def connect(self) -> None:
        """Establish WebSocket connection with retry and backoff."""
        await self._connect_with_backoff()

    async def close(self) -> None:
        """Close WebSocket connection and stop listener if running."""
        await self.stop_listener()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass  # already closed
            self._ws = None

    async def send(self, msg_type: MessageType, payload: dict[str, Any]) -> int:
        """Send a message of any type.  Returns the assigned sequence number.

        Auto-reconnects if the connection was lost.
        Sequence numbers are managed internally — callers never set them.
        """
        seq = self._next_seq()
        msg = Message(type=msg_type, payload=payload, seq=seq)
        await self._ensure_connected()
        await self._ws.send(pack_message(msg))
        return seq

    async def recv(self, *, timeout: float | None = None) -> Message:
        """Receive the next message from the server.

        Args:
            timeout: Max seconds to wait, or ``None`` to block indefinitely.

        Raises:
            RuntimeError: If not connected.
            TimeoutError: If *timeout* expires.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        with anyio.fail_after(timeout):
            data = await self._ws.recv()
        return unpack_message(data)

    # ── Convenience methods (all delegate to send / recv) ────────────

    async def reconnect(self) -> None:
        """Close existing connection (if any) and reconnect with backoff.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        await self.close()
        await self._connect_with_backoff()

    async def start_episode(self, config: dict[str, Any]) -> None:
        """Signal episode start to the server."""
        await self.send(MessageType.EPISODE_START, config)

    async def end_episode(self, result: dict[str, Any]) -> None:
        """Signal episode end to the server."""
        await self.send(MessageType.EPISODE_END, result)

    async def act(self, obs: Observation) -> Action:
        """Send observation and wait for action response (sync mode).

        Validates that the response ``seq`` matches the request ``seq``
        to detect message ordering issues.
        """
        seq = await self.send(MessageType.OBSERVATION, obs)
        response = await self.recv(timeout=self.timeout)
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Server error: {response.payload}")
        if response.seq != seq:
            logger.warning("Seq mismatch: sent %d, received %d. Messages may be out of order.", seq, response.seq)
        return response.payload

    async def send_observation(self, obs: Observation) -> None:
        """Send observation without waiting (realtime mode)."""
        await self.send(MessageType.OBSERVATION, obs)

    def on_action(self, callback: Callable[[Action], None]) -> None:
        """Register callback for incoming actions (realtime mode)."""
        self._action_callback = callback

    async def start_listener(self) -> None:
        """Start a background task that reads messages and dispatches to on_action.

        Required for realtime mode where actions arrive asynchronously.
        Call ``stop_listener()`` to cancel.
        """
        if self._listener_task is not None and not self._listener_task.done():
            return
        self._listener_task = asyncio.create_task(self._listener_loop())
        self._listener_task.add_done_callback(self._on_listener_done)

    async def stop_listener(self) -> None:
        """Stop the background listener task."""
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, anyio.get_cancelled_exc_class()):
                pass
            self._listener_task = None

    def _on_listener_done(self, task: asyncio.Task[None]) -> None:
        """Log unexpected listener loop termination."""
        if not task.cancelled() and task.exception() is not None:
            logger.error("Listener loop crashed", exc_info=task.exception())

    # ── Private helpers ──────────────────────────────────────────────

    async def _connect_with_backoff(self) -> None:
        """Try to connect with exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                with anyio.fail_after(self.timeout):
                    self._ws = await websockets.connect(
                        self.url,
                        compression=None,
                        max_size=None,
                        ping_interval=None,  # server may block GIL during JIT warmup
                    )
                logger.info("Connected to %s (attempt %d/%d)", self.url, attempt, self.max_retries)
                return
            except anyio.get_cancelled_exc_class():
                raise
            except Exception as e:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Server unreachable at {self.url} after {self.max_retries} retries") from e
                wait = self.backoff_base**attempt
                logger.warning(
                    "Reconnect attempt %d/%d failed (%s), retrying in %.1fs...",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    wait,
                )
                await anyio.sleep(wait)

    async def _ensure_connected(self) -> None:
        """Reconnect if the current WebSocket is closed."""
        if self.is_connected:
            return
        logger.warning("Connection lost, attempting reconnect...")
        await self._connect_with_backoff()

    async def _listener_loop(self) -> None:
        """Background loop: read messages and dispatch to on_action callback.

        Only used in realtime mode (AsyncEpisodeRunner).  Sync mode uses
        ``act()`` (send + recv) directly and never starts the listener.

        On connection loss the listener exits instead of reconnecting.
        Reconnecting here would create a new server-side session without
        re-sending EPISODE_START, causing the model server to receive
        observations on an uninitialised session.  Letting the listener
        exit causes the episode to fail cleanly, which the orchestrator
        already handles via episode-level error isolation.
        """
        cancelled = anyio.get_cancelled_exc_class()
        _recv_count = 0
        try:
            while True:
                try:
                    msg = await self.recv()
                    _recv_count += 1
                    if msg.type == MessageType.ACTION and self._action_callback is not None:
                        self._action_callback(msg.payload)
                        if _recv_count <= 3:
                            logger.info("Listener received ACTION #%d (seq=%s)", _recv_count, msg.seq)
                    elif msg.type == MessageType.ERROR:
                        logger.error("Server error in listener: %s", msg.payload)
                    else:
                        logger.debug("Listener received msg type=%s #%d", msg.type, _recv_count)
                except cancelled:
                    raise
                except (ConnectionError, websockets.exceptions.ConnectionClosed):
                    logger.warning(
                        "Connection lost in listener loop, exiting listener (received %d msgs)", _recv_count
                    )
                    break
                except Exception:
                    logger.exception("Error processing message in listener loop, continuing")
        except cancelled:
            logger.debug("Listener cancelled after receiving %d msgs", _recv_count)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def __aenter__(self) -> Connection:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
