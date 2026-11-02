"""The callback contract (send_action/send_error over Rows snapshots) and the servers built on it."""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import anyio
import numpy as np
from anyio.lowlevel import checkpoint
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import ModelServer, Rows
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.protocol.messages import MessageType

from tests.conftest import LogServer, LoopLogServer, start_server, stop_server

LIVE = {"task": {"name": "t"}, "mode": "live"}


def _value(action: dict[str, Any]) -> float:
    return float(np.asarray(action["actions"]).reshape(-1)[0])


def _act(rows: Rows, values: Any) -> dict[str, Any]:
    return {"actions": np.asarray(values, np.float32).reshape(len(rows), 1) * np.ones((1, 7), np.float32)}


class ReplayServer(ModelServer):
    """A bare ModelServer that keeps every Rows it got, to replay stale replies with them."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[np.ndarray, Rows]] = []
        self.ended: list[dict[str, Any]] = []
        self.start_error: Exception | None = None
        self.mode = "echo"

    async def on_episode_start(self, rows: Rows) -> None:
        try:
            await self.send_action(rows, _act(rows, [0.0] * len(rows)))
        except RuntimeError as exc:
            self.start_error = exc

    async def on_observation(self, obs: dict[str, Any], rows: Rows) -> None:
        values = np.asarray(obs["value"], np.float32)
        if self.seen and self.mode == "replay":  # the previous observation's rows answer again, first
            old_values, old_rows = self.seen[-1]
            await self.send_action(old_rows, _act(old_rows, old_values + 100))
        self.seen.append((values, rows))
        if self.mode == "repeat":  # row 0 listed twice in one reply: it answers row 0 once, never row 1
            await self.send_action(rows.take([0, 0]), _act(rows.take([0, 0]), [values[0], -1.0]))
            assert rows._pending is not None and not rows._pending.done.is_set()
            await self.send_action(rows.take([1]), _act(rows.take([1]), values[1:]))
            return
        if self.mode == "partial":
            await self.send_action(rows.take([0]), _act(rows.take([0]), values[:1]))
            raise ValueError("the rest failed")
        if self.mode != "silent":
            await self.send_action(rows, _act(rows, values))
            await self.send_action(rows, _act(rows, values + 1000))  # a duplicate: dropped

    async def on_episode_end(self, rows: Rows, results: list[dict[str, Any]]) -> None:
        self.ended += results


@pytest.mark.anyio
async def test_a_duplicate_or_delayed_reply_never_answers_another_observation(free_port):
    server = ReplayServer()
    server.mode = "replay"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            assert [_value(await conn.act({"value": v})) for v in (1.0, 2.0, 3.0)] == [1.0, 2.0, 3.0]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_reply_through_a_reused_row_is_dropped(free_port):
    server = ReplayServer()
    server.mode = "replay"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0], [[0, {}]], 1.0))
            reply = await conn.exchange(_vstep([0], [[0, {}]], 2.0))  # slot 0 restarts on the same row id
            assert server.seen[0][1].ids.tolist() == server.seen[1][1].ids.tolist() == [0]
            assert _value({"actions": reply["actions"]["actions"][0]}) == 2.0 and reply["errors"] == []
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_reply_without_an_observation_is_rejected(free_port):
    server = ReplayServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            assert _value(await conn.act({"value": 4.0})) == 4.0
        assert isinstance(server.start_error, RuntimeError)
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_rows_answered_before_an_exception_keep_their_actions(free_port):
    server = ReplayServer()
    server.mode = "partial"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0, 1, 2], [[s, {}] for s in range(3)], [1.0, 2.0, 3.0]))
        assert [_value({"actions": a}) for a in reply["actions"]["actions"]] == [1.0]
        assert sorted(s for s, _ in reply["errors"]) == [1, 2]
        assert all("ValueError: the rest failed" in e for _, e in reply["errors"])
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_row_listed_twice_in_a_reply_is_answered_once(free_port):
    server = ReplayServer()
    server.mode = "repeat"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0, 1], [[0, {}], [1, {}]], [1.0, 2.0]))
        assert [_value({"actions": a}) for a in reply["actions"]["actions"]] == [1.0, 2.0] and reply["errors"] == []
    finally:
        await stop_server(task)


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        (None, {"status": "aborted", "failure_reason": "connection_closed"}),
        ({"status": "error", "failure_reason": "timeout"}, {"status": "error", "failure_reason": "timeout"}),
    ],
)
@pytest.mark.anyio
async def test_a_disconnect_ends_an_episode_waiting_for_a_reply(free_port, end, expected):
    """With an EPISODE_END sent before closing (the harness timed out), its status reaches the server."""
    server = ReplayServer()
    server.mode = "silent"
    task = await start_server(server, free_port)
    try:
        conn = Connection(f"ws://127.0.0.1:{free_port}", timeout=0.2)
        await conn.connect()
        await conn.start_episode({"task": {"name": "t"}})
        with pytest.raises(TimeoutError):
            await conn.act({"value": 1.0})
        if end is not None:
            await conn.end_episode(end)
        await conn.close()
        with anyio.fail_after(3):
            while not server.ended:
                await anyio.sleep(0.02)
        assert server.ended == [expected]
    finally:
        await stop_server(task)


def _vstep(slots: list[int], starts: list, values: Any) -> dict[str, Any]:
    v = np.broadcast_to(np.asarray(values, np.float64), (len(slots),)).copy()
    return {"ends": [], "starts": starts, "slots": slots, "observations": {"value": v}}


# ---------------------------------------------------------------------------
# PredictModelServer
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_failed_observation_still_counts_as_a_step(free_port):
    server = LogServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            with pytest.raises(RuntimeError, match="observation_failed"):
                await conn.act({"value": -1.0})
            await conn.act({"value": 2.0})
        assert server.predicted == [(-1.0, 0), (2.0, 1)]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_vector_end_sent_before_a_disconnect_still_counts(free_port):
    """A step still inferring when the client closes; the ends it sent after it keep their status."""
    server = LogServer(delay=0.3)
    task = await start_server(server, free_port)
    try:
        conn = Connection(f"ws://127.0.0.1:{free_port}")
        await conn.connect()
        await conn.send(MessageType.VECTOR_STEP, _vstep([0], [[0, {}]], 1.0))
        end = {"status": "error", "failure_reason": "timeout"}
        await conn.send(MessageType.VECTOR_STEP, {"ends": [[0, end]], "starts": [], "slots": [], "observations": {}})
        await conn.close()
        with anyio.fail_after(3):
            while not server.results:
                await anyio.sleep(0.02)
        assert server.results == [end]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_max_wait_time_counts_from_the_oldest_requests_arrival(free_port):
    """B arrives while A infers; once A is done, B has waited max_wait_time already and is inferred at once."""
    server = LogServer(delay=0.4, max_batch_size=4, max_wait_time=0.3)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as a, Connection(f"ws://127.0.0.1:{free_port}") as b:
            await a.start_episode({"task": {"name": "t"}})
            await b.start_episode({"task": {"name": "t"}})
            await a.send_observation({"value": 1.0})
            await anyio.sleep(0.35)  # A waits 0.3 s, then infers until 0.7 s
            sent = time.monotonic()
            await b.act({"value": 2.0})
            await a.recv()
        started = {c.value: c.t for c in server.calls}
        assert started[2.0] - sent < 0.5  # 0.35 s behind A; 0.65 s if the wait started over
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_cancelled_model_call_never_runs_or_returns_only_after_it_ran():
    server = ReplayServer()
    ran: list[str] = []

    def slow(tag: str) -> None:
        time.sleep(0.2)
        ran.append(tag)

    with anyio.CancelScope() as scope:
        scope.cancel()
        await server.run_on_model_thread(slow, "cancelled before")
    with anyio.move_on_after(0.05):
        await server.run_on_model_thread(slow, "cancelled during")
    assert ran == ["cancelled during"]  # the await outlived the call
    await anyio.sleep(0.3)
    assert ran == ["cancelled during"]


# ---------------------------------------------------------------------------
# Cancellation and failure paths: every row released once, every request completed once
# ---------------------------------------------------------------------------


class HangingStartServer(ReplayServer):
    max_rows = 1

    def __init__(self) -> None:
        super().__init__()
        self.hang = True

    async def on_episode_start(self, rows: Rows) -> None:
        if self.hang:
            await anyio.sleep_forever()


@pytest.mark.anyio
async def test_a_start_cancelled_in_on_episode_start_frees_its_row():
    server = HangingStartServer()
    with anyio.move_on_after(0.05):
        await server._start_episodes([({}, "s")])
    assert server._running() == 0
    server.hang = False
    assert await server._start_episodes([({}, "s")]) == [0]


@pytest.mark.anyio
async def test_a_cancelled_episode_end_still_finishes_the_episode():
    server = LogServer()
    (rid,) = await server._start_episodes([({"task": {}}, "s")])
    assert isinstance(rid, int)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await server._end_episodes([rid], [{}])
    assert server.ended == [[rid]] and server._running() == 0


async def _requests(server: PredictModelServer, obs: list[dict[str, Any]]) -> list[Any]:
    from vla_eval.model_servers.engine import _Request

    ids = await server._start_episodes([({}, "s")] * len(obs))
    requests = []
    for o, i in zip(obs, ids):
        assert isinstance(i, int)
        requests.append(_Request({k: np.asarray([v]) for k, v in o.items()}, server._rows([i]), wait=False))
    return requests


@pytest.mark.anyio
async def test_a_cancelled_dispatch_fails_its_requests():
    server = LogServer()
    (req,) = await _requests(server, [{"value": 1.0}])
    with anyio.CancelScope() as scope:
        scope.cancel()
        await server._engine._run([req])
    assert req.done.is_set() and set(req.errors) == {0} and "stopped" in str(req.errors[0])


@pytest.mark.anyio
async def test_a_dispatch_cancelled_one_request_at_a_time_completes_every_request():
    server = LogServer(delay=0.2)
    first, second = await _requests(server, [{"value": 1.0}, {"other": 2.0}])  # different keys: one at a time
    with anyio.move_on_after(0.05):
        await server._engine._run([first, second])
    assert first.done.is_set() and not first.errors and _value(first.actions) == 1.0
    assert second.done.is_set() and set(second.errors) == {0}


@pytest.mark.anyio
async def test_a_row_ending_before_the_model_call_is_not_inferred():
    server = LogServer()
    (req,) = await _requests(server, [{"value": 1.0}])
    async with anyio.create_task_group() as tg:
        tg.start_soon(server._engine._run, [req])
        await checkpoint()  # the dispatcher runs to its first checkpoint
        server._gen[0] += 1
    assert server.predicted == [] and set(req.errors) == {0}


# ---------------------------------------------------------------------------
# An episode loop on the latest observation (continuous inference)
# ---------------------------------------------------------------------------


async def _live(server: ModelServer, port: int, values: list[float], gap: float = 0.0) -> list[Any]:
    """Send ``values`` in one live episode (``gap`` apart), then collect the replies until they stop."""
    task = await start_server(server, port)
    try:
        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            assert conn.server_info["capabilities"]["vector_step"] is False
            await conn.start_episode(LIVE)
            seqs = []
            for v in values:
                seqs.append(await conn.send(MessageType.OBSERVATION, {"value": v}))
                await anyio.sleep(gap)
            got = []
            while True:
                try:
                    m = await conn.recv(timeout=0.5)
                except Exception:
                    break
                got.append((m.type, seqs.index(m.seq), m.payload))
            await conn.end_episode({})
        return got
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_live_coalesces_to_the_latest_observation_and_counts_every_step(free_port):
    server = LoopLogServer(delay=0.3)
    got = await _live(server, free_port, [1.0, 2.0, 3.0])
    assert [(t, i, _value(p)) for t, i, p in got] == [(MessageType.ACTION, 0, 1.0), (MessageType.ACTION, 2, 3.0)]
    assert server.predicted == [(1.0, 0), (3.0, 2)]


@pytest.mark.anyio
async def test_live_a_failure_errors_its_observation_and_the_loop_goes_on(free_port):
    got = await _live(LoopLogServer(), free_port, [-1.0, 2.0], gap=0.2)
    assert [(t, i) for t, i, _ in got] == [(MessageType.ERROR, 0), (MessageType.ACTION, 1)]
    assert "bad observation" in got[0][2]["error"] and _value(got[1][2]) == 2.0


@pytest.mark.anyio
async def test_live_episode_end_waits_for_the_inference_in_flight(free_port):
    """The in-flight action is dropped, and the next episode on the connection gets a loop of its own."""
    server = LoopLogServer(delay=0.3)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode(LIVE)
            await conn.send_observation({"value": 1.0})
            await anyio.sleep(0.1)
            await conn.end_episode({})
            await conn.start_episode(LIVE)  # the same row id, once the first episode finished
            assert _value(await conn.act({"value": 2.0})) == 2.0
        assert server.ended_busy[:1] == [False]
        assert server.threads == {"model_0"}
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_server_without_vector_step_refuses_it(free_port):
    task = await start_server(LoopLogServer(), free_port)
    conn = Connection(f"ws://127.0.0.1:{free_port}")
    await conn.connect()
    try:
        assert conn.server_info["capabilities"] == {
            "vector_step": False,
            "episode_ready": True,
            "publish": True,
            "episode_error": True,
        }
        step = {"ends": [], "starts": [[0, {"task": {"name": "t"}}]], "slots": [0], "observations": {"v": [1.0]}}
        with pytest.raises(RuntimeError, match="vector_step_unsupported"):
            await conn.exchange(step)
    finally:
        await conn.close()
        await stop_server(task)


# ---------------------------------------------------------------------------
# VLANeXt on the base: dense state history, its own chunk queue, per-row errors
# ---------------------------------------------------------------------------


@pytest.fixture
def vlanext(monkeypatch):
    """VLANeXtModelServer with torch and the model stubbed: the chunk is ``value + arange``, the history logged."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delitem(sys.modules, "vla_eval.model_servers.vlanext", raising=False)
    from vla_eval.model_servers import vlanext as m

    monkeypatch.setitem(sys.modules, "vla_eval.model_servers.vlanext", m)  # dropped again after the test
    log: list[tuple[float, list[float], int, str]] = []

    def predict_chunk(self, obs, row):
        v = float(obs["value"])
        log.append(
            (v, [float(s[0]) for s in self._state_histories[row.id]], row.step, threading.current_thread().name)
        )
        if v < 0:
            raise ValueError("bad state")
        return {"actions": v + np.arange(56, dtype=np.float64).reshape(8, 7)}

    monkeypatch.setattr(m.VLANeXtModelServer, "_init_model", lambda self: None)
    monkeypatch.setattr(m.VLANeXtModelServer, "_predict_chunk", predict_chunk)
    return m.VLANeXtModelServer, log


def _vobs(values: Any) -> dict[str, Any]:
    v = np.atleast_1d(np.asarray(values, np.float64))
    return {"value": v, "state": v[:, None] * np.ones((1, 7))}


@pytest.mark.anyio
async def test_vlanext_history_is_dense_and_buffered_steps_come_from_its_queue(vlanext, free_port):
    cls, log = vlanext
    task = await start_server(cls("x", chunk_size=3), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            got = [await conn.act({"value": float(v), "state": np.full(7, float(v))}) for v in range(1, 6)]
            await conn.start_episode({"task": {"name": "t"}})  # a new episode starts a new history
            await conn.act({"value": 9.0, "state": np.full(7, 9.0)})
        chunk = np.arange(56, dtype=np.float32).reshape(8, 7)
        expected = [1 + chunk[0], 1 + chunk[1], 1 + chunk[2], 4 + chunk[0], 4 + chunk[1]]
        assert all(np.array_equal(a["actions"], e) and a["actions"].dtype == np.float32 for a, e in zip(got, expected))
        assert [x[:3] for x in log] == [(1.0, [1.0], 0), (4.0, [1.0, 2.0, 3.0, 4.0], 3), (9.0, [9.0], 0)]
        assert {x[3] for x in log} == {"model_0"}
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_vlanext_a_rows_failure_is_its_own_in_a_vector_step(vlanext, free_port):
    cls, log = vlanext
    task = await start_server(cls("x", chunk_size=2), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            step = {"ends": [], "starts": [[s, {}] for s in range(3)], "slots": [0, 1, 2]}
            reply = await conn.exchange({**step, "observations": _vobs([1.0, -1.0, 3.0])})
            assert [s for s, _ in reply["errors"]] == [1] and "bad state" in reply["errors"][0][1]
            assert [float(a[0]) for a in reply["actions"]["actions"]] == [1.0, 3.0]
            reply = await conn.exchange({**step, "starts": [], "observations": _vobs([2.0, 5.0, 4.0])})
            assert reply["errors"] == []
            assert [float(a[0]) for a in reply["actions"]["actions"]] == [1.0 + 7, 5.0, 3.0 + 7]
        assert [x[:2] for x in log] == [(1.0, [1.0]), (-1.0, [-1.0]), (3.0, [3.0]), (5.0, [-1.0, 5.0])]
    finally:
        await stop_server(task)
