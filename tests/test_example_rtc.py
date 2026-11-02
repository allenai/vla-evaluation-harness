"""The RTC example (examples/live_servers/rtc.py): its time grid against chrono's ticker semantics on a scripted
clock; warm-up and streaming to a live runner."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, cast

import anyio
import numpy as np
import pytest

from examples.live_servers.rtc import RTCServer
from vla_eval.connection import Connection
from vla_eval.runners.live_runner import LiveEpisodeRunner

from tests.conftest import StubBenchmark, running


class Chunker(RTCServer):
    """Chunk ``k``: ``value * 100 + k`` per row."""

    def __init__(self, chunk: int = 8, **kw: Any) -> None:
        super().__init__(chunk_size=chunk, **kw)
        self.shape: tuple[int, ...] | None = None

    def predict(self, obs, row):
        if self.shape is not None:
            return np.zeros(self.shape, np.float32)
        return float(obs["value"]) * 100 + np.arange(self.chunk_size, dtype=np.float32)[:, None] * np.ones((1, 2))


class ScriptClock:
    """Wakes ``every`` at scripted times (missed ticks skipped, as chrono's WallClock), setting ``now``."""

    def __init__(self, wakes: list[float]) -> None:
        self.wakes, self.t = wakes, wakes[0]

    def now(self) -> float:
        return self.t

    async def every(self, period: float, *, first: str = "after"):
        for t in self.wakes:
            self.t = t
            yield t
            for _ in range(5):  # let a refill task run
                await anyio.sleep(0)


class FakeEp:
    """The parts of EpisodeContext RTCServer uses; requests arrive as scripted ``(time, value)``."""

    def __init__(self, clock: ScriptClock, arrivals: list[tuple[float, float, float]]) -> None:
        self.clock, self.arrivals = clock, list(arrivals)  # (arrival, admitted_at, value)
        self.logger = logging.getLogger("tests.rtc")
        self.closing = False
        self.published: list[tuple[float, float]] = []
        self.errors: list[str] = []
        self.block: anyio.Event | None = None  # blocks the publish at t=2 until the second inference
        self.calls = 0
        self.tg: Any = None

    def receive_nowait(self) -> Any:
        due = [a for a in self.arrivals if a[0] <= self.clock.now()]
        if not due:
            return None
        newest = due[-1]
        self.arrivals = [a for a in self.arrivals if a[0] > self.clock.now()]

        async def error(exc: Exception) -> None:
            self.errors.append(str(exc))

        return SimpleNamespace(obs={"value": newest[2]}, row=None, admitted_at=newest[1], error=error)

    def start_soon(self, fn: Any, *args: Any) -> None:
        self.tg.start_soon(fn, *args)

    async def call(self, fn: Any, *args: Any) -> Any:
        await anyio.sleep(0)
        self.calls += 1
        if self.calls == 2 and self.block is not None:
            self.block.set()  # the blocked publish returns after this refill replaced the plan
        return fn(*args)

    async def publish(self, action: Any, **kw: Any) -> None:
        if self.block is not None and self.clock.now() == 2.0:
            await self.block.wait()
            await anyio.sleep(0)
        self.published.append((self.clock.now(), float(action["actions"][0])))


async def _rtc(server: RTCServer, wakes: list[float], arrivals: list[tuple[float, float, float]]) -> FakeEp:
    ep = FakeEp(ScriptClock(wakes), arrivals)
    async with anyio.create_task_group() as tg:
        ep.tg = tg
        await server.run(cast(Any, ep))
    return ep


@pytest.mark.anyio
async def test_rtc_publishes_the_current_slot_once_and_never_a_future_one():
    """hz=1, admitted at 0.3: slot 0 as the chunk lands; ticks at 2.8 and 3.0 give slot 2 once; slot 3 waits for
    3.3."""
    ep = await _rtc(Chunker(hz=1.0), [0.3, 2.8, 3.0, 3.3, 4.0], [(0.3, 0.3, 1.0)])
    assert ep.published == [(0.3, 100.0), (2.8, 102.0), (3.3, 103.0)]


@pytest.mark.anyio
async def test_rtc_publishes_a_landing_chunk_whose_tick_had_nothing_to_send():
    """chunk_size=1: each chunk lands after its tick and its only slot expires at the next one."""
    ep = await _rtc(Chunker(chunk=1, hz=1.0), [0.0, 1.0, 2.0], [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 2.0, 3.0)])
    assert ep.published == [(0.0, 100.0), (1.0, 200.0), (2.0, 300.0)]


@pytest.mark.anyio
async def test_rtc_skips_expired_slots():
    ep = await _rtc(Chunker(hz=10.0), [0.0, 0.1, 0.35, 0.4], [(0.0, 0.0, 1.0)])
    assert ep.published == [(0.0, 100.0), (0.1, 101.0), (0.35, 103.0), (0.4, 104.0)]  # slot 2 expired unseen


@pytest.mark.anyio
async def test_rtc_warns_once_per_episode_when_a_chunk_is_dropped(caplog):
    caplog.set_level(logging.DEBUG, logger="tests.rtc")
    arrivals = [(5.0, 0.0, 1.0), (9.0, 3.0, 2.0)]  # inferred 5 s and 6 s after admission; horizon 2 s
    ep = await _rtc(Chunker(chunk=2, hz=1.0), [5.0, 5.5, 6.0, 9.0, 9.5], arrivals)
    drops = [(r.levelno, r.getMessage()) for r in caplog.records if "chunk dropped" in r.getMessage()]
    assert ep.published == [] and drops == [
        (logging.WARNING, "inference 5000 ms exceeds the chunk horizon 2000 ms; chunk dropped"),
        (logging.DEBUG, "inference 6000 ms exceeds the chunk horizon 2000 ms; chunk dropped"),
    ]


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.anyio
async def test_rtc_refills_ahead_and_a_replacement_during_a_blocked_publish_keeps_its_cursor(blocked):
    """Plan 1 (admitted 0.0) streams slots 0 (as it lands), 1, 2; at 2.0 two slots are left, so the observation
    admitted at 1.0 is inferred; its plan (anchored at 1.0, slot 0 expired) takes over at slot 2 even mid-publish."""
    server = Chunker(chunk=4, hz=1.0, refill_at=2)
    ep = FakeEp(ScriptClock([0.0, 1.0, 2.0, 3.0, 4.0]), [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)])
    ep.block = anyio.Event() if blocked else None
    async with anyio.create_task_group() as tg:
        ep.tg = tg
        await server.run(cast(Any, ep))
    expected = [(0.0, 100.0), (1.0, 101.0), (2.0, 102.0), (3.0, 202.0), (4.0, 203.0)]
    assert ep.published == expected and ep.calls == 2


@pytest.mark.parametrize("shape", [(0, 2), (2,)])
@pytest.mark.anyio
async def test_rtc_an_empty_or_flat_chunk_fails_its_observation_only(shape):
    server = Chunker(hz=1.0)
    server.shape = shape
    ep = await _rtc(server, [0.0, 1.0, 2.0], [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)])
    assert ep.published == [] and len(ep.errors) == 2 and "[T, D]" in ep.errors[0]


def test_rtc_validates_its_parameters():
    bad: list[dict[str, Any]] = [{"hz": 0.0}, {"hz": math.inf}, {"chunk_size": 0}, {"refill_at": -1}]
    for kw in bad:
        with pytest.raises(ValueError):
            RTCServer(**kw)


@pytest.mark.anyio
async def test_rtc_warms_up_once_before_the_first_episode_and_streams_to_a_live_runner(free_port):
    class Logged(Chunker):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.values: list[float] = []

        def predict(self, obs, row):
            self.values.append(float(obs["value"]))
            return super().predict(obs, row)

    server = Logged(hz=20.0, warmup_obs={"value": 7.0})
    results = []
    async with running(server, free_port) as url:
        for _ in range(2):
            async with Connection(url) as conn:
                bench = StubBenchmark(done_at_step=30)
                results.append(await LiveEpisodeRunner(hz=20.0).run_episode(bench, {"name": "t"}, conn, max_steps=30))
    assert server.values[0] == 7.0 and server.values.count(7.0) == 1 and len(server.values) > 1
    for result in results:
        assert result["rt_metrics"]["update_count"] >= 10 and result["rt_metrics"]["fresh_count"] >= 10
