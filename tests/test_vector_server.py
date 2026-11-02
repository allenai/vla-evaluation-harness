"""The model server over rows: vector steps, lifecycle in batches, chunk buffers per row, row ids."""

from __future__ import annotations

import time
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import Row
from vla_eval.model_servers.predict import PredictModelServer

from tests.conftest import BatchLogServer, start_server, stop_server


async def _connect(server: PredictModelServer, port: int) -> tuple[Any, Connection]:
    task = await start_server(server, port)
    conn = Connection(f"ws://127.0.0.1:{port}")
    await conn.connect()
    return task, conn


def _start(slot: int, **task: Any) -> list[Any]:
    return [slot, {"task": {"name": "t", **task}}]


def _step(slots: list[int], starts: list[list[Any]] | None = None, ends: list[list[Any]] | None = None) -> dict:
    obs = {"value": np.array([float(s + 1) for s in slots])}
    return {"ends": ends or [], "starts": starts or [], "slots": slots, "observations": obs}


def _actions(reply: dict[str, Any], slots: list[int]) -> dict[int, float]:
    """The reply's action rows, by slot: the slots without an error, in order."""
    failed = {s for s, _ in reply["errors"]}
    ok = [s for s in slots if s not in failed]
    rows = reply["actions"].get("actions", [])
    assert len(rows) == len(ok)
    return {s: float(np.asarray(a).reshape(-1)[0]) for s, a in zip(ok, rows)}


@pytest.mark.anyio
async def test_a_failed_start_fails_its_slot_alone_and_frees_its_row(free_port):
    server = BatchLogServer()
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(0), _start(1, broken=True), _start(2)]))
        assert _actions(reply, [0, 1, 2]) == {0: 1.0, 2: 3.0}  # the rows of the slots that did not fail
        ((slot, detail),) = reply["errors"]
        assert slot == 1 and "episode_start_failed: ValueError: cannot load this task" in detail
        reply = await conn.exchange(_step([1], [_start(1)]))
        assert reply["errors"] == [] and server.started[-1] == [1]  # row 1 is free again
        await conn.close()
        with anyio.fail_after(3):  # the disconnect ends the open episodes
            while not server.ended:
                await anyio.sleep(0.02)
        assert [sorted(e) for e in server.ended] == [[0, 1, 2]]  # the failed start never ended
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_vector_step_never_waits_for_a_batch(free_port):
    """One slot left, or some slots served from their buffers: no max_wait_time stall."""
    server = BatchLogServer(chunk_size=2, max_batch_size=8, max_wait_time=5.0)
    task, conn = await _connect(server, free_port)
    try:
        t0 = time.monotonic()
        await conn.exchange(_step([0], [_start(0)]))  # one row
        await conn.exchange(_step([0, 1], [_start(1)]))  # slot 0 buffered, slot 1 inferred
        await conn.exchange(_step([0, 1]))  # slot 0 inferred, slot 1 buffered
        assert time.monotonic() - t0 < 2.0
        assert server.batches == [[0], [1], [0]] and server.steps == [[0], [0], [2]]
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_each_row_has_its_own_chunk_buffer_and_counts_every_observation(free_port):
    server = BatchLogServer(chunk_size=3)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0], [_start(0)]))
        # slot 1 starts a step later: its chunk is out of phase with slot 0's
        for k in range(5):
            reply = await conn.exchange(_step([0, 1], [_start(1)] if k == 0 else []))
            assert _actions(reply, [0, 1]) == {0: 1.0, 1: 2.0}
        assert server.batches == [[0], [1], [0], [1]]  # 6 and 5 steps of chunk 3: inferred at 0,3 / 1,4
        assert server.steps == [[0], [0], [3], [3]]  # buffered observations count
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_reset_episodes_sets_an_episodes_chunk_size(free_port):
    server = BatchLogServer(chunk_size=4, max_batch_size=None)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0, 1], [_start(0, suite_chunk=2), _start(1)]))
        for _ in range(3):
            await conn.exchange(_step([0, 1]))
        assert server.batches == [[0, 1], [0]]  # slot 0 refills after 2 steps, slot 1 after 4
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_rows_are_reused_once_their_episodes_end(free_port):
    server = BatchLogServer(max_batch_size=None)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        await conn.exchange(_step([0, 2, 1], [_start(1)], ends=[[1, {"success": True}]]))
        assert server.ended == [[1]] and server.started[-1] == [1]  # slot 1's new episode takes row 1 again
        assert server.batches[-1] == [0, 2, 1] and server.steps[-1] == [1, 1, 0]
    finally:
        await conn.close()
        await stop_server(task)
    assert sorted(server.ended[-1]) == [0, 1, 2]  # the disconnect ended the open episodes


@pytest.mark.anyio
async def test_max_rows_refuses_more_episodes(free_port):
    task, conn = await _connect(BatchLogServer(max_rows=2), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert [s for s, _ in reply["errors"]] == [2] and "at most 2 episodes" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_an_observation_for_a_slot_without_an_episode_is_an_error(free_port):
    task, conn = await _connect(BatchLogServer(), free_port)
    try:
        await conn.exchange(_step([0], [_start(0)]))
        reply = await conn.exchange(_step([0], ends=[[0, {}]]))
        assert reply["actions"] == {} and "no running episode" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_step_infers_its_rows_in_slices_of_max_batch_size(free_port):
    server = BatchLogServer(max_batch_size=2)
    task, conn = await _connect(server, free_port)
    try:
        assert conn.server_info["capabilities"] == {
            "vector_step": True,
            "episode_ready": True,
            "publish": True,
            "episode_error": True,
        }
        reply = await conn.exchange(_step([4, 0, 2], [_start(s) for s in (0, 2, 4)]))
        assert server.started == [[0, 1, 2]] and server.batches == [[2, 0], [1]]  # one start call for the step
        assert isinstance(reply["actions"]["actions"], np.ndarray) and reply["errors"] == []
        assert _actions(reply, [4, 0, 2]) == {4: 5.0, 0: 1.0, 2: 3.0}  # in the request's order
        bad = _step([0, 2, 4])
        bad["observations"]["value"][0] = -1.0  # fails the first slice (slots 0 and 2) only
        reply = await conn.exchange(bad)
        assert sorted(s for s, _ in reply["errors"]) == [0, 2] and _actions(reply, [0, 2, 4]) == {4: 5.0}
    finally:
        await conn.close()
        await stop_server(task)


class ListOfDicts(PredictModelServer):
    """The old predict_batch contract: a list of dicts."""

    def predict_batch(self, obs, rows):
        return [{"actions": np.zeros(7)} for _ in range(len(rows) + 1)]


@pytest.mark.anyio
async def test_an_old_style_predict_batch_names_the_change(free_port):
    task, conn = await _connect(ListOfDicts(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        assert len(reply["errors"]) == 2 and "column batches" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


class PerRow(PredictModelServer):
    """``predict`` per row, by value: chunks of different lengths in other key orders (1, 2), an exception (3),
    unusable actions (4), other keys (5)."""

    def predict(self, obs, row: Row):
        v = float(obs["value"])
        if v == 3.0:
            raise ValueError("row 2 fails")
        a, meta = np.ones((int(v), 7), dtype=np.float32), {"row": row.id}
        if v == 4.0:
            return {"actions": "oops", "meta": meta}
        if v == 5.0:
            return {}
        return {"meta": meta, "actions": a} if v == 2.0 else {"actions": a, "meta": meta}


@pytest.mark.anyio
async def test_predict_per_row_isolates_failures_and_keeps_ragged_actions(free_port):
    task, conn = await _connect(PerRow(), free_port)
    try:
        reply = await conn.exchange(_step(list(range(5)), [_start(s) for s in range(5)]))
        assert sorted(s for s, _ in reply["errors"]) == [2, 3, 4]
        assert [np.asarray(a).shape for a in reply["actions"]["actions"]] == [(1, 7), (2, 7)]
        assert reply["actions"]["meta"] == {"row": [0, 1]}  # a nested dict column
    finally:
        await conn.close()
        await stop_server(task)


class EmptyChunk(PredictModelServer):
    """Slot 1's chunk is empty."""

    def __init__(self) -> None:
        super().__init__(chunk_size=3, max_batch_size=None)

    def predict_batch(self, obs, rows):
        return {"actions": [np.ones((0 if v == 2.0 else 3, 7), np.float32) for v in np.asarray(obs["value"])]}


@pytest.mark.anyio
async def test_a_bad_chunk_fails_its_row_alone(free_port):
    task, conn = await _connect(EmptyChunk(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert [s for s, _ in reply["errors"]] == [1] and len(reply["actions"]["actions"]) == 2
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_buffered_actions_do_not_wait_for_a_batch(free_port):
    server = BatchLogServer(chunk_size=3, max_batch_size=8, max_wait_time=0.5)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            await conn.act({"value": 1.0})  # infers after waiting for a batch
            t0 = time.monotonic()
            await conn.act({"value": 1.0})
            await conn.act({"value": 1.0})
            assert time.monotonic() - t0 < 0.4 and len(server.batches) == 1  # served from the buffer at once
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_repeated_or_restarted_slots_never_share_a_row(free_port):
    """A slot ended twice, started twice, or restarted while running frees its row once (rows stay distinct)."""
    server = BatchLogServer(max_batch_size=None)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        await conn.exchange(_step([0, 1], [_start(0), _start(0)], ends=[[0, {}], [0, {}]]))
        await conn.exchange(_step([0, 1], [_start(1)]))  # slot 1 restarts without an end
        other = Connection(f"ws://127.0.0.1:{free_port}")
        await other.connect()
        await other.exchange(_step([0, 1], [_start(0), _start(1)]))
        await other.close()
        assert server.ended[:2] == [[0], [1]]  # each once
        mine, theirs = server.batches[-2], server.batches[-1]
        assert len(set(mine)) == 2 and len(set(theirs)) == 2 and not set(mine) & set(theirs)
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_vector_step_arriving_during_a_batching_wait_does_not_inherit_it(free_port):
    server = BatchLogServer(max_batch_size=8, max_wait_time=2.0)
    task, single = await _connect(server, free_port)
    vector = Connection(f"ws://127.0.0.1:{free_port}")
    await vector.connect()
    try:
        await single.start_episode({"task": {"name": "t"}})
        async with anyio.create_task_group() as tg:
            tg.start_soon(single.act, {"value": 9.0})  # waits up to 2 s for company
            await anyio.sleep(0.2)
            t0 = time.monotonic()
            await vector.exchange(_step([0, 1], [_start(0), _start(1)]))
            assert time.monotonic() - t0 < 1.0
    finally:
        await vector.close()
        await single.close()
        await stop_server(task)
