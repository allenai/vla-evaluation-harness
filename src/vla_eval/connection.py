"""Connection client: communicates with ModelServer over WebSocket."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import websockets
from websockets.protocol import State as WebSocketState

from vla_eval.protocol.messages import Message, MessageType, make_hello_payload, pack_message, unpack_message
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

CAPABILITIES = {"episode_ready": True, "publish": True, "episode_error": True}  # what this client handles (HELLO)


class ServerEpisodeError(RuntimeError):
    """The server failed the running episode (``EPISODE_ERROR``); its ``failure_reason`` is ``model_act``."""


class Connection:
    """WebSocket client for communicating with a ModelServer.

    Handles observation→action exchange, episode lifecycle messages,
    and automatic reconnection with exponential backoff.

    The class is organised in two tiers:

    **Core primitives** — the irreducible operations everything else builds on:

    - `connect` / `close` — WebSocket lifecycle.
    - `send(msg_type, payload)` — send any message (auto-seq; EPISODE_START auto-reconnects).
    - `recv(timeout=...)` — receive the next message.

    **Convenience methods** — thin compositions of the primitives above:

    - `start_episode` — send + await ``EPISODE_READY`` (when the server advertises it).
    - `end_episode`, `send_observation` — one-liner sends.
    - `act` — request/response (send + recv + seq validation).
    - `on_action`, `on_publish`, `listening()` — callback-based receive (live mode).
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
        self._episode_seq: int = 0  # the last EPISODE_START's: replies before it belong to an earlier episode
        self._action_seq: int = 0  # the live listener's last applied ACTION: an older one is stale
        self._action_callback: Callable[[Action], None] | None = None
        self._publish_callback: Callable[[Action], None] | None = None
        self._episode_active = False  # between EPISODE_START and EPISODE_END: server messages for it count
        self._publish_index = -1  # the episode's last applied PUBLISH
        self._benchmark: str | None = None
        self.server_info: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        """Return True if the WebSocket is open."""
        return self._ws is not None and self._ws.state is WebSocketState.OPEN

    async def connect(self, *, benchmark: str | None = None) -> None:
        """Establish WebSocket connection with retry, backoff, and HELLO handshake."""
        self._benchmark = benchmark
        await self._connect_with_backoff()
        await self._hello_handshake()

    async def _hello_handshake(self) -> None:
        """Exchange HELLO messages with the server."""
        extra: dict[str, Any] = {"benchmark": self._benchmark} if self._benchmark else {}
        payload = make_hello_payload(capabilities=dict(CAPABILITIES), **extra)
        await self.send(MessageType.HELLO, payload)
        reply = await self.recv(timeout=self.timeout)
        if reply.type != MessageType.HELLO:
            raise RuntimeError(f"Expected HELLO reply, got {reply.type}")
        self.server_info = reply.payload

    async def close(self) -> None:
        """Close WebSocket connection and stop listener if running."""
        self._episode_active = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass  # already closed
            self._ws = None

    async def send(self, msg_type: MessageType, payload: dict[str, Any]) -> int:
        """Send a message of any type.  Returns the assigned sequence number.

        Only EPISODE_START reconnects a lost connection: mid-episode a new server
        session would lack EPISODE_START, so the dead socket raises ``ConnectionClosed``.
        Sequence numbers are managed internally — callers never set them.
        """
        if msg_type == MessageType.EPISODE_START:
            await self._ensure_connected()
        elif self._ws is None:
            raise RuntimeError("Not connected")
        seq = self._next_seq()
        if msg_type == MessageType.EPISODE_START:  # before the send can suspend: replies may race it
            self._episode_seq, self._episode_active, self._publish_index = seq, True, -1
        elif msg_type == MessageType.EPISODE_END:
            self._episode_active = False
        try:
            await self._ws.send(pack_message(Message(type=msg_type, payload=payload, seq=seq)))
        except BaseException:
            if msg_type == MessageType.EPISODE_START and self._episode_seq == seq:
                self._episode_active = False
            raise
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
        """Close existing connection (if any) and reconnect with backoff + HELLO.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        await self.close()
        await self._connect_with_backoff()
        await self._hello_handshake()

    async def start_episode(self, config: dict[str, Any]) -> None:
        """Send ``EPISODE_START``; a server advertising ``episode_ready`` is awaited (``timeout``) until it started.
        Call it outside ``listening()``, which would read the same socket.

        Raises:
            RuntimeError: The server failed the start (``ERROR``).
            TimeoutError: No ``EPISODE_READY`` within ``timeout``.
        """
        seq = await self.send(MessageType.EPISODE_START, config)
        if not self.server_info.get("capabilities", {}).get("episode_ready"):
            return  # an older server: a failed start surfaces at the first reply
        try:
            with anyio.fail_after(self.timeout):
                reply = await self.recv()
                while reply.seq < seq:  # a late reply of an earlier episode
                    reply = await self.recv()
        except TimeoutError:
            raise TimeoutError(f"no EPISODE_READY within {self.timeout}s") from None
        if reply.type == MessageType.ERROR:  # e.g. a refused start (``requires``): no episode to end
            self._episode_active = False
            raise RuntimeError(f"Server error: {reply.payload}")
        if reply.type != MessageType.EPISODE_READY or reply.seq != seq:
            raise RuntimeError(f"Expected EPISODE_READY for seq {seq}, got {reply.type.value} seq {reply.seq}")

    async def end_episode(self, result: dict[str, Any]) -> None:
        """Signal episode end to the server."""
        await self.send(MessageType.EPISODE_END, result)

    async def act(self, obs: Observation) -> Action:
        """Send observation and wait for action response (sync mode).

        Validates that the response ``seq`` matches the request ``seq``
        to detect message ordering issues.
        """
        seq = await self.send(MessageType.OBSERVATION, obs)
        with anyio.fail_after(self.timeout):
            response = await self.recv()
            while response.seq < self._episode_seq:  # a late reply of an earlier episode
                logger.debug("Dropped %s seq %d from before EPISODE_START", response.type.value, response.seq)
                response = await self.recv()
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Server error: {response.payload}")
        if response.seq != seq:
            logger.warning("Seq mismatch: sent %d, received %d. Messages may be out of order.", seq, response.seq)
        return response.payload

    async def exchange(self, step: dict[str, Any]) -> dict[str, Any]:
        """Send a ``VECTOR_STEP``, return the ``VECTOR_RESULT`` payload.  Never reconnects: the server's episodes
        die with the socket, so the runner must know."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        seq = self._next_seq()
        await self._ws.send(pack_message(Message(type=MessageType.VECTOR_STEP, payload=step, seq=seq)))
        reply = await self.recv(timeout=self.timeout)
        if reply.type == MessageType.ERROR and reply.seq == seq:
            raise RuntimeError(f"Server error: {reply.payload.get('error')}")
        if reply.type != MessageType.VECTOR_RESULT or reply.seq != seq:
            raise RuntimeError(f"Expected VECTOR_RESULT for seq {seq}, got {reply.type.value} seq {reply.seq}")
        return reply.payload

    async def send_observation(self, obs: Observation) -> None:
        """Send observation without waiting (live mode)."""
        await self.send(MessageType.OBSERVATION, obs)

    def on_action(self, callback: Callable[[Action], None]) -> None:
        """Register callback for incoming actions (live mode)."""
        self._action_callback = callback

    def on_publish(self, callback: Callable[[Action], None]) -> None:
        """Register callback for actions the server publishes unasked (live mode, ``PUBLISH``)."""
        self._publish_callback = callback

    @asynccontextmanager
    async def listening(self) -> AsyncIterator[None]:
        """Run the listener for the block (live mode): ACTION to ``on_action``, PUBLISH to ``on_publish``.  A listener
        error (a dropped connection, ``ServerEpisodeError``) cancels the block and is re-raised."""
        error: Exception | None = None

        async def listen() -> None:
            nonlocal error
            try:
                await self._listener_loop()
            except Exception as exc:
                error = exc
                tg.cancel_scope.cancel()

        async with anyio.create_task_group() as tg:
            tg.start_soon(listen)
            try:
                yield
            except Exception as exc:
                error = exc  # re-raised outside the group so callers see the bare type
            tg.cancel_scope.cancel()
        if error is not None:
            raise error

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
        await self._hello_handshake()

    async def _listener_loop(self) -> None:
        """Background loop: read messages and dispatch to on_action callback.

        Only used in live mode (LiveEpisodeRunner).  Sync mode uses
        ``act()`` (send + recv) directly and never starts the listener.

        On connection loss (or any recv error) the listener raises instead of reconnecting: a new
        server-side session would lack EPISODE_START.  ``listening()`` re-raises
        it so the orchestrator fails the episode via episode-level error isolation.
        """
        cancelled = anyio.get_cancelled_exc_class()
        _recv_count = 0
        try:
            while True:
                # A recv error (dropped or misused socket) is unrecoverable and may not await, so retrying
                # would spin the event loop; end the listener instead.
                try:
                    msg = await self.recv()
                except Exception as exc:
                    logger.warning("Listener recv failed after %d msgs: %r", _recv_count, exc)
                    raise
                _recv_count += 1
                try:
                    if msg.seq < self._episode_seq or (msg.type == MessageType.ACTION and msg.seq <= self._action_seq):
                        logger.debug("Dropped stale %s seq %d", msg.type.value, msg.seq)  # earlier episode or action
                    elif msg.type in (MessageType.PUBLISH, MessageType.EPISODE_ERROR) and not (
                        self._episode_active and msg.seq == self._episode_seq
                    ):
                        logger.debug("Dropped %s seq %d of no running episode", msg.type.value, msg.seq)
                    elif msg.type == MessageType.EPISODE_ERROR:
                        raise ServerEpisodeError(f"Server failed the episode: {msg.payload.get('error')}")
                    elif msg.type == MessageType.PUBLISH:
                        index = msg.payload.get("index")
                        if not isinstance(index, int) or index <= self._publish_index:
                            logger.debug("Dropped stale PUBLISH index %r", index)
                        else:
                            self._publish_index = index
                            if self._publish_callback is not None:
                                self._publish_callback(msg.payload["action"])
                    elif msg.type == MessageType.ACTION and self._action_callback is not None:
                        self._action_seq = msg.seq
                        self._action_callback(msg.payload)
                        if _recv_count <= 3:
                            logger.info("Listener received ACTION #%d (seq=%s)", _recv_count, msg.seq)
                    elif msg.type == MessageType.ERROR:
                        logger.error("Server error in listener: %s", msg.payload)
                    else:
                        logger.debug("Listener received msg type=%s #%d", msg.type, _recv_count)
                except ServerEpisodeError:  # terminal for the episode; the runner ends it
                    raise
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
