"""Sync sim time: the framework moves a sync episode's clock to ``k * step_dt``, delivers observation ``k``, then
wakes that instant's timers, one sleeper at a time, settling their ``ep.call`` s; live episodes keep wall time."""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import anyio
from anyio.lowlevel import checkpoint
import numpy as np
import pytest
from chrono_daemon import WallClock

from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.connection import Connection
from vla_eval.model_servers.daemons import SETTLE_ROUNDS, EpisodeContext, ObservationRequest, SimClock
from vla_eval.recording import NullEpisodeRecorder
from vla_eval.runners.base import ModelActError
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner
from vla_eval.runners.vector_runner import VectorEpisode, VectorEpisodeRunner

from tests.conftest import Daemon, StubBenchmark, echo, observe, running, serving, settled, start

ZERO = {"actions": np.zeros(2, np.float32)}


class Stepped(StubBenchmark):
    step_dt = 0.1


async def _reply(req: ObservationRequest) -> None:
    await req.reply(ZERO)


# ---------------------------------------------------------------------------
# Observation k at k * step_dt; timers fire on their deadlines, after the observation due with them
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_observation_k_is_handled_at_k_step_dt(free_port):
    seen: list[tuple[int, float, float]] = []

    async def run(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            seen.append((req.row.step, ep.clock.now(), req.admitted_at))
            await _reply(req)

    async with running(Daemon(run), free_port) as url, Connection(url) as conn:
        result = await SyncEpisodeRunner().run_episode(Stepped(done_at_step=5), {"name": "t"}, conn)
    assert result["steps"] == 5
    assert seen == [(k, k * 0.1, k * 0.1) for k in range(5)]


@pytest.mark.parametrize("period", [0.0625, 0.25, 0.375, 1.0])
@pytest.mark.anyio
async def test_every_fires_once_per_deadline_even_several_per_step(period):
    events: list[tuple[float, str]] = []

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(period):
            assert ep.clock.now() == t
            events.append((t, "fire"))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            events.append((ep.clock.now(), "obs"))
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(9):  # the last at 2.0
            await observe(server, rid, v)
        await settled(server, rid)
    fires = [t for t, kind in events if kind == "fire"]
    assert fires == [period * n for n in range(1, int(2.0 / period) + 1)]
    assert events == sorted(events, key=lambda e: (e[0], e[1] != "obs"))  # an observation precedes its instant's fires


@pytest.mark.parametrize("step_dt", [0.25, 0.1])  # 0.1: 3 * 0.1 > 0.3, every(0.3) ticks just below obs 3
@pytest.mark.anyio
async def test_a_timer_at_t_sees_observation_t_and_its_model_call(step_dt):
    seen: list[tuple[float, float]] = []  # (tick, the latest observation's inferred value)
    latest = {"v": -1.0}

    async def ticker(ep: EpisodeContext, period: float) -> None:
        async for t in ep.clock.every(period):
            seen.append((t, latest["v"]))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep, 3 * step_dt if step_dt == 0.1 else step_dt)
        async for req in ep.observations():
            latest["v"] = await ep.call(float, req.obs["value"])
            await _reply(req)

    server = Daemon(run, step_dt=step_dt)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(7):
            await observe(server, rid, v)
        await settled(server, rid)
    stride = 3 if step_dt == 0.1 else 1
    assert [v for _, v in seen] == [float(k) for k in range(stride, 7, stride)]


@pytest.mark.anyio
async def test_a_reply_can_wait_for_its_instants_timer_without_deadlock(free_port):
    """A loop that answers observation k >= 1 with the result of the timer due at k * step_dt (zero delay)."""
    ticked: list[float] = []

    async def run(ep: EpisodeContext) -> None:
        state: dict[str, Any] = {"obs": None, "out": anyio.Event()}

        async def ticker() -> None:
            async for t in ep.clock.every(0.1):
                state["result"] = await ep.call(float, state["obs"])
                ticked.append(t)
                state["out"].set()

        ep.start_soon(ticker)
        async for req in ep.observations():
            if req.row.step == 0:  # no tick at 0
                await _reply(req)
                continue
            state["obs"], state["out"] = req.obs["value"], anyio.Event()
            await state["out"].wait()  # the tick at this instant runs after the observation
            await req.reply({"actions": np.full(2, state["result"], np.float32)})

    async with running(Daemon(run), free_port) as url, Connection(url) as conn:
        result = await SyncEpisodeRunner().run_episode(Stepped(done_at_step=5), {"name": "t"}, conn)
    assert result["steps"] == 5 and ticked == [k * 0.1 for k in range(1, 5)]


@pytest.mark.anyio
async def test_decimal_periods_land_on_their_observations_without_drift():
    events: list[tuple[float, str]] = []

    async def ticker(ep: EpisodeContext, period: float) -> None:
        async for _ in ep.clock.every(period):
            events.append((round(ep.clock.now(), 9), f"every {period}"))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep, 0.1)
        ep.start_soon(ticker, ep, 0.5)
        async for req in ep.observations():
            events.append((round(ep.clock.now(), 9), "obs"))
            await _reply(req)

    server = Daemon(run, step_dt=0.02)  # 3 * 0.1 > 15 * 0.02 in floats
    async with serving(server):
        (rid,) = await start(server)
        for v in range(151):  # the last at 3.0
            await observe(server, rid, v)
        await settled(server, rid)
    kinds = [kind for _, kind in events]
    assert kinds.count("every 0.1") == 30 and kinds.count("every 0.5") == 6
    assert events == sorted(events, key=lambda e: (e[0], e[1] != "obs"))  # each fire after its instant's observation


@pytest.mark.anyio
async def test_a_fires_child_task_calls_finish_before_time_moves_on():
    done: list[tuple[str, float, float]] = []  # (who, tick, clock when its call returned)

    async def work(who: str, t: float, ep: EpisodeContext) -> None:
        await ep.call(time.sleep, 0.002)
        done.append((who, t, ep.clock.now()))

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.25):
            ep.start_soon(work, "child", t, ep)  # handed off to another task
            await work("ticker", t, ep)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            # every earlier fire finished
            assert len(done) == 2 * sum(t < ep.clock.now() for t in (0.25, 0.5, 0.75, 1.0))
            await _reply(req)

    server = Daemon(run, step_dt=0.125)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(9):  # the last at 1.0
            assert not (await observe(server, rid, v)).errors
        await settled(server, rid)
    assert sorted(done) == sorted((who, t, t) for who in ("child", "ticker") for t in (0.25, 0.5, 0.75, 1.0))


@pytest.mark.parametrize(("hops", "settle_rounds", "held"), [(4, None, True), (40, 64, True), (40, None, False)])
@pytest.mark.anyio
async def test_async_hops_before_a_call_are_covered_within_settle_rounds(hops, settle_rounds, held):
    """The heuristic boundary: settling waits on ``ep.call`` s only once one has started, so a fire's async hops
    before its call must fit in ``settle_rounds`` checkpoints (configurable) for the call to hold time."""
    done: list[tuple[float, float]] = []  # (tick, clock when its call returned)

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.25):
            for _ in range(hops):
                await checkpoint()
            await ep.call(time.sleep, 0.002)
            done.append((t, ep.clock.now()))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    kw = {} if settle_rounds is None else {"settle_rounds": settle_rounds}
    assert hops < (settle_rounds or SETTLE_ROUNDS) or not held
    server = Daemon(run, step_dt=0.25, **kw)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(5):
            assert not (await observe(server, rid, v)).errors
        await settled(server, rid)
    assert (done == [(t, t) for t in (0.25, 0.5, 0.75, 1.0)]) is held  # else a call ends after time moved on


@pytest.mark.parametrize("latency", [0.01, 0.0])  # 0: a call can finish inside the rounds, before the next starts
@pytest.mark.anyio
async def test_a_fires_model_call_finishes_before_time_moves_on(latency):
    calls: list[tuple[float, float, float]] = []  # (tick, clock at the call's end, value)

    def slow(v: float) -> float:
        time.sleep(latency)
        return v

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.0625):
            v = t
            for _ in range(5):  # each call after the previous: still the same instant
                v = await ep.call(slow, v)
            calls.append((t, ep.clock.now(), v))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(5):  # the last at 1.0
            await observe(server, rid, v)
        await settled(server, rid)
    assert calls == [(0.0625 * n, 0.0625 * n, 0.0625 * n) for n in range(1, 17)]


@pytest.mark.anyio
async def test_a_call_an_observation_starts_lands_before_the_next_timer_whatever_its_latency():
    async def once(latency: float) -> list[tuple[str, float, float]]:
        log: list[tuple[str, float, float]] = []
        plan = {"v": -1.0}

        def infer(v: float) -> float:
            time.sleep(latency)
            return v

        async def background(ep: EpisodeContext, v: float) -> None:
            plan["v"] = await ep.call(infer, v)
            log.append(("landed", ep.clock.now(), plan["v"]))

        async def timer(ep: EpisodeContext) -> None:
            async for t in ep.clock.every(0.1):
                log.append(("timer", t, plan["v"]))

        async def run(ep: EpisodeContext) -> None:
            ep.start_soon(timer, ep)
            async for req in ep.observations():
                ep.start_soon(background, ep, float(req.obs["value"]))  # infer in the background, reply at once
                await _reply(req)

        server = Daemon(run, step_dt=0.25)
        async with serving(server):
            (rid,) = await start(server)
            for v in range(4):
                assert not (await observe(server, rid, v)).errors
        return log

    fast, slow = (
        [e for e in log if e[1] <= 0.5] for log in (await once(0.0), await once(0.02))
    )  # the last observation's call may land or be dropped at close
    assert fast == slow
    assert [e for e in fast if e[0] == "landed"] == [("landed", k * 0.25, float(k)) for k in range(3)]
    assert ("timer", 0.1, 0.0) in fast  # obs 0's call landed at 0, before the 0.1 timer


@pytest.mark.anyio
async def test_a_sleeper_cancelled_while_asleep_is_never_woken():
    log: list[str] = []

    async def quitter(ep: EpisodeContext) -> None:
        with anyio.move_on_after(0.01):  # wall time: gives up before sim time reaches 0.3
            await ep.clock.sleep(0.3)
            log.append("woken")
        log.append(f"gave up at {ep.clock.now()}")

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(quitter, ep)
        async for req in ep.observations():
            if req.row.step == 2:
                log.append(f"obs at {ep.clock.now()}")
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 0)
        await anyio.sleep(0.05)
        for v in range(1, 3):
            await observe(server, rid, v)
    assert log == ["gave up at 0.0", "obs at 0.5"]


@pytest.mark.parametrize("loops", [1, 2])  # 2: overlapping calls, never all drained at once
@pytest.mark.anyio
async def test_a_fire_that_keeps_calling_at_one_instant_fails_the_episode_instead_of_skipping_time(loops):
    calls = 0

    def noop() -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.0005)  # long enough that the other loop's next call starts first

    async def caller(ep: EpisodeContext) -> None:
        while True:  # sim time never gets to move on
            await ep.call(noop)

    async def ticker(ep: EpisodeContext) -> None:
        async for _ in ep.clock.every(0.1):
            for _ in range(loops):
                ep.start_soon(caller, ep)
            return

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 0)
        pending = await observe(server, rid, 1)
    assert "still waiting after 1000 settles" in str(pending.errors[0])
    assert calls >= 1000


@pytest.mark.anyio
async def test_one_episodes_long_settle_does_not_hold_another_episodes_time():
    gate = threading.Event()
    ticks: list[tuple[int, float]] = []

    def blocked() -> None:
        assert gate.wait(5)

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.1):
            ticks.append((ep.row_id, t))
            if ep.row_id == a and t > 0.15:
                await ep.call(blocked)  # a's settle waits on this while b's time moves on

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        a, b = await start(server, n=2)
        await observe(server, a, 0)
        held = await observe(server, a, 1, wait=False)  # at 0.25: the 0.2 fire's call blocks
        for v in range(5):  # b reaches 1.0 meanwhile
            assert not (await observe(server, b, v)).errors
        assert not held.done.is_set()
        gate.set()
        with anyio.fail_after(3):
            await held.done.wait()
        assert not held.errors
    assert [t for rid, t in ticks if rid == b] == [0.1 * n for n in range(1, 11)]
    assert [t for rid, t in ticks if rid == a] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# A dual system (S2 every 0.5 s, S1 every 0.1 s, each observation answered from S1's latest chunk) is deterministic
# ---------------------------------------------------------------------------


class Reflex(Daemon):
    """S2 plans every 0.5 s from the latest observation, S1 refines every 0.1 s from the latest plan, and each
    observation is answered from S1's latest chunk (a hold before the first).  Real model latency is random."""

    def __init__(self, seed: int) -> None:
        super().__init__(step_dt=0.05)
        self.rng = random.Random(seed)
        self.schedule: list[tuple[str, float, float]] = []

    def _infer(self, x: float, latency: float) -> float:
        time.sleep(latency)
        return x + 1.0

    async def run(self, ep: EpisodeContext) -> None:
        latest = {"obs": -1.0, "plan": -1.0, "chunk": -1.0}

        async def s2() -> None:
            async for t in ep.clock.every(0.5, first="now"):
                seen = latest["obs"]
                latest["plan"] = await ep.call(self._infer, seen, self.rng.uniform(0, 0.02))
                self.schedule.append(("s2", t, seen))

        async def s1() -> None:
            async for t in ep.clock.every(0.1):
                seen = latest["plan"]
                latest["chunk"] = await ep.call(self._infer, seen, self.rng.uniform(0, 0.005))
                self.schedule.append(("s1", t, seen))

        ep.start_soon(s2)
        ep.start_soon(s1)
        async for req in ep.observations():
            latest["obs"] = float(req.obs["value"])
            self.schedule.append(("reply", ep.clock.now(), latest["chunk"]))
            await req.reply({"actions": np.full(2, latest["chunk"], np.float32)})


@pytest.mark.anyio
async def test_a_reflex_dual_loop_runs_the_same_schedule_twice():
    schedules = []
    for seed in (0, 1):
        server = Reflex(seed)
        async with serving(server):
            (rid,) = await start(server)
            for k in range(24):  # 1.15 s of sim time
                await observe(server, rid, float(k))
        schedules.append(server.schedule)
    assert schedules[0] == schedules[1]
    kinds = [kind for kind, _, _ in schedules[0]]
    assert kinds.count("s2") == 3 and kinds.count("s1") == 11 and kinds.count("reply") == 24
    plans = {t: seen + 1.0 for kind, t, seen in schedules[0] if kind == "s2"}
    s1 = {t: seen for kind, t, seen in schedules[0] if kind == "s1"}
    assert s1[0.5] == plans[0.5]  # due together: S2 (asleep first) is woken and settled first


# ---------------------------------------------------------------------------
# Vector slots: each episode's clock on its own
# ---------------------------------------------------------------------------


class SteppedVector(VectorStepBenchmark):
    """Slot ``s`` runs task ``i``'s episode for ``i + 2`` steps; slots restart as soon as they end."""

    partial_reset = True
    step_dt = 0.25

    def __init__(self) -> None:
        super().__init__(num_envs=2)
        self.t, self.length = [0, 0], [0, 0]

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": f"task_{i}", "length": i + 2} for i in range(3)]

    def reset(self, slots, tasks, recorders):
        for slot, task in zip(slots, tasks):
            self.t[slot], self.length[slot] = 0, task["length"]
        return {"value": np.array([float(t["length"]) for t in tasks])}

    def step(self, slots, actions):
        done = []
        for slot in slots:
            self.t[slot] += 1
            done.append(self.t[slot] >= self.length[slot])
        return {"value": np.array([float(self.length[s]) for s in slots])}, np.array(done)

    def result(self, slot):
        return {"success": True}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 10}


@pytest.mark.anyio
async def test_vector_slots_advance_their_own_clocks(free_port):
    seen: dict[float, list[tuple[int, float]]] = {}  # by episode length: (step, clock)
    configs: list[Any] = []

    async def run(ep: EpisodeContext) -> None:
        configs.append(ep.episode.config.get("step_dt"))
        async for req in ep.observations():
            seen.setdefault(float(req.obs["value"]), []).append((req.row.step, ep.clock.now()))
            await req.reply(ZERO)

    class Vector(Daemon):
        supports_vector_step = True

    queue = [{"name": f"task_{i}", "length": i + 2, "episode_idx": 0} for i in (0, 2, 1)]
    ended: list[Any] = []

    def next_episode(slot: int) -> VectorEpisode | None:
        return VectorEpisode(queue.pop(0), NullEpisodeRecorder()) if queue else None

    def on_end(episode, result, error, *, aborted=False):
        ended.append(error)

    async with running(Vector(run), free_port) as url, Connection(url) as conn:
        await VectorEpisodeRunner().run(SteppedVector(), conn, next_episode, on_end)
    assert ended == [None] * 3 and configs == [0.25] * 3
    # lengths 2, 4, 3: the length-3 episode starts mid-run in the freed slot, its clock fresh at 0
    assert seen == {n: [(k, k * 0.25) for k in range(int(n))] for n in (2.0, 4.0, 3.0)}


@pytest.mark.anyio
async def test_server_shutdown_ends_an_episode_whose_end_waits_for_its_last_instant():
    async def ticker(ep: EpisodeContext) -> None:
        async for _ in ep.clock.every(0.25):
            while True:  # the wake keeps calling the model: its instant would last max_settles calls
                await ep.call(time.sleep, 0.02)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    began = time.monotonic()
    async with serving(server) as tg:
        (rid,) = await start(server)
        await observe(server, rid, 0)
        await observe(server, rid, 1)  # replied; the tick at 0.25 runs on
        tg.start_soon(server._end_episodes, [rid], [{}])
        await anyio.sleep(0.1)
        assert server._running() == 1  # the end waits for the instant
        tg.cancel_scope.cancel()
    assert time.monotonic() - began < 2 and server._running() == 0 and server.log[-1] == ("finish", [rid])


@pytest.mark.anyio
async def test_vector_episodes_run_their_last_instants_timers_before_ending(free_port):
    ticks: dict[float, list[tuple[float, float]]] = {}  # by episode length: (tick, the observation it saw)

    async def run(ep: EpisodeContext) -> None:
        latest: dict[str, Any] = {}

        async def ticker() -> None:
            async for t in ep.clock.every(0.25):
                ticks.setdefault(latest["n"], []).append((t, latest["k"]))

        ep.start_soon(ticker)
        async for req in ep.observations():
            latest.update(n=float(req.obs["value"]), k=float(req.row.step))
            await req.reply(ZERO)

    class Vector(Daemon):
        supports_vector_step = True

    queue = [{"name": f"task_{i}", "length": i + 2, "episode_idx": 0} for i in (0, 2, 1)]

    def next_episode(slot: int) -> VectorEpisode | None:
        return VectorEpisode(queue.pop(0), NullEpisodeRecorder()) if queue else None

    async with running(Vector(run), free_port) as url, Connection(url) as conn:
        await VectorEpisodeRunner().run(SteppedVector(), conn, next_episode, lambda *a, **k: None)
    # every instant's tick, the last included, ran after (and saw) that instant's observation
    assert ticks == {n: [(k * 0.25, float(k)) for k in range(1, int(n))] for n in (2.0, 4.0, 3.0)}


# ---------------------------------------------------------------------------
# Unknown step_dt: clock use fails the episode; loops that never touch the clock work
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_without_step_dt_using_the_clock_fails_the_episode_and_other_loops_work(free_port):
    async def timed(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            ep.clock.now()
            await _reply(req)

    async def stamped(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            req.admitted_at
            await _reply(req)

    for run in (timed, stamped):
        server = Daemon(run)
        async with running(server, free_port) as url, Connection(url) as conn:
            with pytest.raises(ModelActError, match="step_dt unknown"):
                await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn)
    async with running(Daemon(echo), free_port) as url, Connection(url) as conn:
        result = await SyncEpisodeRunner().run_episode(StubBenchmark(done_at_step=3), {"name": "t"}, conn)
    assert result["steps"] == 3


# ---------------------------------------------------------------------------
# step_dt on the wire: each runner's EPISODE_START, the server's default; live keeps wall time
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_step_dt_reaches_the_episode_from_each_runner_or_the_server(free_port):
    seen: list[tuple[Any, float | None, type]] = []

    async def run(ep: EpisodeContext) -> None:
        seen.append((ep.episode.config.get("step_dt"), ep.step_dt, type(ep.clock)))
        await echo(ep)

    async with running(Daemon(run, step_dt=0.5), free_port) as url, Connection(url) as conn:
        await SyncEpisodeRunner().run_episode(Stepped(), {"name": "t"}, conn, max_steps=2)
        await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
        await LiveEpisodeRunner(hz=50.0).run_episode(Stepped(), {"name": "t"}, conn, max_steps=2)
        await conn.start_episode({"step_dt": None})  # null: the server's default
        await conn.act({"value": 1.0})
    assert seen == [
        (0.1, 0.1, SimClock),
        (None, 0.5, SimClock),
        (0.02, 0.02, WallClock),  # live: 1 / hz, wall time
        (None, 0.5, SimClock),
    ]
