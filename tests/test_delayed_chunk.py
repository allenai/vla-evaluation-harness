"""DelayedChunkServer: step-counted inference delay on a sync sim clock, around any chunked PredictModelServer.

The stub's chunk for observation ``v`` is ``v * 100 + i``, so every action names its source observation and its
index in that chunk."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import Row
from vla_eval.model_servers.delayed_chunk import DelayedChunkServer
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.recording import RecordingStore

from tests.conftest import running

STEP_DT = 0.05  # LIBERO's


class Stub(PredictModelServer):
    """Chunk ``[T, 1]`` for observation ``v``: ``v * 100 + i``."""

    def __init__(self, horizon: int = 8, **kw: Any) -> None:
        super().__init__(**kw)
        self.horizon = horizon

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        v = float(obs["value"])
        return {"actions": (v * 100 + np.arange(self.horizon, dtype=np.float32))[:, None]}


async def _episode(conn: Connection, values: list[float], config: dict[str, Any] | None = None) -> list[float]:
    """One sync episode: send ``values`` in order, return the first action dim of each reply."""
    await conn.start_episode({"task": {}, "step_dt": STEP_DT, **(config or {})})
    out = [float(np.asarray((await conn.act({"value": v}))["actions"]).ravel()[0]) for v in values]
    await conn.end_episode({})
    return out


@pytest.mark.anyio
@pytest.mark.parametrize("s", [1, 4])
async def test_zero_delay_matches_the_inner_server_with_chunk_size_s(free_port, s):
    steps = list(range(12))
    async with running(Stub(chunk_size=s), free_port) as url, Connection(url) as conn:
        plain = await _episode(conn, [float(j) for j in steps])
    async with running(DelayedChunkServer(Stub(), delay_steps=0, execute_steps=s), free_port) as url:
        async with Connection(url) as conn:
            wrapped = await _episode(conn, [float(j) for j in steps])
    assert wrapped == plain
    assert plain == [(j - j % s) * 100 + j % s for j in steps]


@pytest.mark.anyio
async def test_delayed_chunks_land_d_steps_later_and_start_at_index_d(free_port):
    # s = 4, d = 2, T = 8.  Step 0: the initial chunk (no delay).  Boundaries 0, 4, 8: inference from that step's
    # observation, landing 2 steps later, served from index 2; until then the previous chunk keeps playing.
    expected = [0, 1, 2, 3, 4, 5, 402, 403, 404, 405, 802, 803]
    async with running(DelayedChunkServer(Stub(horizon=8), delay_steps=2, execute_steps=4), free_port) as url:
        async with Connection(url) as conn:
            got = await _episode(conn, [float(j) for j in range(len(expected))])
    assert got == expected


@pytest.mark.anyio
async def test_a_chunk_shorter_than_s_plus_d_fails_the_episode(free_port):
    async with running(DelayedChunkServer(Stub(horizon=5), delay_steps=2, execute_steps=4), free_port) as url:
        async with Connection(url) as conn:
            await conn.start_episode({"task": {}, "step_dt": STEP_DT})
            with pytest.raises(RuntimeError, match="s \\+ d"):
                await conn.act({"value": 0.0})


@pytest.mark.parametrize(("d", "s"), [(-1, 4), (5, 4), (0, 0)])
def test_bad_delay_or_horizon_is_refused(d, s):
    with pytest.raises(ValueError):
        DelayedChunkServer(Stub(), delay_steps=d, execute_steps=s)


@pytest.mark.anyio
async def test_episodes_get_the_same_actions_on_one_connection_or_two(free_port):
    # Episode e sends values e * 1000 + j: per-episode schedule state must not leak across episodes or connections.
    episodes = [[float(e * 1000 + j) for j in range(10)] for e in range(4)]

    def server() -> DelayedChunkServer:
        return DelayedChunkServer(Stub(horizon=8), delay_steps=2, execute_steps=4)

    async with running(server(), free_port) as url, Connection(url) as conn:
        serial = [await _episode(conn, values) for values in episodes]

    sharded: list[list[float]] = [[] for _ in episodes]

    async def shard(url: str, idx: list[int]) -> None:
        async with Connection(url) as conn:
            for i in idx:
                sharded[i] = await _episode(conn, episodes[i])

    async with running(server(), free_port) as url, anyio.create_task_group() as tg:
        tg.start_soon(shard, url, [0, 2])
        tg.start_soon(shard, url, [1, 3])
    assert sharded == serial


@pytest.mark.anyio
async def test_each_step_records_which_chunk_and_index_it_served(free_port, tmp_path: Path):
    db = tmp_path / "recording.sqlite"
    RecordingStore(db).close()  # the harness creates the schema
    recording = {"recording": {"sid": "s", "eid": "e", "db_path": str(db)}}
    async with running(DelayedChunkServer(Stub(horizon=8), delay_steps=2, execute_steps=4), free_port) as url:
        async with Connection(url) as conn:
            await _episode(conn, [float(j) for j in range(8)], recording)

    con = sqlite3.connect(str(db))
    rows = {k: json.loads(f) for k, f in con.execute("SELECT step_id, fields FROM step_rows WHERE sid='s'")}
    con.close()
    # chunk_src: the observation step the served chunk was inferred from (-1: the initial chunk).
    # obs_step: set on steps that start an inference, from that step's observation.
    assert [(rows[j]["chunk_src"], rows[j]["chunk_idx"]) for j in range(8)] == [
        (-1, 0),
        (-1, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (4, 2),
        (4, 3),
    ]
    assert [j for j in range(8) if rows[j].get("obs_step") is not None] == [0, 4]
