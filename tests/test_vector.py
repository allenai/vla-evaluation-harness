"""VectorStepBenchmark: several environments per process, one model-server session per running episode."""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from typing import Any, ClassVar
from unittest.mock import patch

import anyio
import numpy as np
import pytest
import websockets.exceptions

from vla_eval.benchmarks.base import BenchmarkCommon, VectorStepBenchmark
from vla_eval.orchestrator import Orchestrator, UnhealthyShardError
from vla_eval.recording import RecordingStore

from tests.conftest import BatchLogServer, LogServer, start_server, stop_server


class StubVectorBenchmark(VectorStepBenchmark):
    """Task ``i``'s episode lasts ``i + 2`` steps, plus ``episode_idx % (jitter + 1)``.  Slot ``s`` observes
    ``value = s + 1``; the echo server sends it back as the action, so ``misrouted`` lists every action that
    reached the wrong slot."""

    last: ClassVar[StubVectorBenchmark]  # the instance the orchestrator built

    def __init__(self, num_envs: int = 3, num_tasks: int = 2, jitter: int = 0, **kwargs: Any) -> None:
        super().__init__(num_envs)
        self.num_tasks = num_tasks
        self.jitter = jitter
        self.t = [0] * num_envs
        self.length = [0] * num_envs
        self.running: set[int] = set()
        self.resets: list[tuple[list[int], list[int]]] = []  # (slots reset, slots running at the time)
        self.steps: list[list[int]] = []
        self.misrouted: list[tuple[int, float]] = []
        self.reset_at: list[tuple[int, list[str]]] = []  # (env steps so far, task names started)
        self.threads: set[int] = set()
        self.slot_task: dict[int, str] = {}
        self.step_tasks: list[set[str]] = []  # the tasks of each step's slots
        self.finished = [False] * num_envs
        StubVectorBenchmark.last = self

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": f"task_{i}", "length": i + 2} for i in range(self.num_tasks)]

    def reset(self, slots, tasks, recorders):
        self.threads.add(threading.get_ident())
        self.resets.append((list(slots), sorted(self.running)))
        self.reset_at.append((len(self.steps), [t["name"] for t in tasks]))
        for slot, task in zip(slots, tasks):
            self.t[slot], self.length[slot] = 0, task["length"] + task["episode_idx"] % (self.jitter + 1)
            self.running.add(slot)
            self.slot_task[slot] = task["name"]
            self.finished[slot] = False
        return self.obs(slots)

    def obs(self, slots):
        return {
            "value": np.array([self.value(s) for s in slots]),
            "task_description": [self.slot_task[s] for s in slots],
        }

    def value(self, slot):
        return float(slot + 1)

    def step(self, slots, actions):
        self.threads.add(threading.get_ident())
        self.steps.append(sorted(slots))
        self.step_tasks.append({self.slot_task[s] for s in slots})
        done = []
        for slot, action in zip(slots, actions["actions"]):
            got = float(np.asarray(action).reshape(-1)[0])
            if got != slot + 1:
                self.misrouted.append((slot, got))
            self.t[slot] += 1
            self.finished[slot] = self.t[slot] >= self.length[slot]
            if self.finished[slot]:
                self.running.discard(slot)
            done.append(self.finished[slot])
        return self.obs(slots), np.array(done)

    def result(self, slot):
        self.threads.add(threading.get_ident())
        return {"success": self.finished[slot]}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 50}

    def cleanup(self) -> None:
        self.threads.add(threading.get_ident())


class PartialResetStub(StubVectorBenchmark):
    partial_reset = True


class NeverDoneStub(StubVectorBenchmark):
    def step(self, slots, actions):
        return self.obs(slots), np.zeros(len(slots), bool)


class BrokenResetStub(StubVectorBenchmark):
    def reset(self, slots, tasks, recorders):
        raise RuntimeError("Failed to find a supported physical device")


class TaskShapedStub(StubVectorBenchmark):
    """Task ``i`` observes ``i + 1`` state numbers, as OCBench's tasks differ."""

    def obs(self, slots):
        dims = {s: int(self.slot_task[s].split("_")[1]) + 1 for s in slots}
        return {**super().obs(slots), "state": np.stack([np.zeros(dims[s]) for s in slots])}


class SlotOneResultErrorStub(StubVectorBenchmark):
    """Slot 1's results cannot be read."""

    def result(self, slot):
        if slot == 1:
            raise RuntimeError("score unavailable")
        return super().result(slot)


class SlotOneModelErrorStub(StubVectorBenchmark):
    """Slot 1 sends a value the server rejects."""

    def value(self, slot):
        return -1.0 if slot == 1 else float(slot + 1)


def _config(url: str, tmp_path, name: str = "vec", episodes: int = 4, **params: Any) -> dict[str, Any]:
    entry = {
        "benchmark": "tests.test_vector:StubVectorBenchmark",
        "name": name,
        "episodes_per_task": episodes,
        "params": {"num_envs": 3, "num_tasks": 2, **params},
    }
    return {"server": {"url": url}, "output_dir": str(tmp_path), "benchmarks": [entry]}


async def _run(cls: type, config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    with patch("vla_eval.orchestrator.resolve_import_string", return_value=cls):
        (result,) = await Orchestrator(config, **kwargs).run()
    return result


def _episodes(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(t["task"], ep) for t in result["tasks"] for ep in t["episodes"]]


@pytest.fixture
async def log_server(free_port: int):
    server = LogServer()
    task = await start_server(server, free_port)
    yield f"ws://127.0.0.1:{free_port}", server
    await stop_server(task)


@pytest.mark.anyio
async def test_waves_run_every_item_and_reset_only_when_idle(echo_server, tmp_path):
    result = await _run(StubVectorBenchmark, _config(echo_server, tmp_path), no_save=True)
    bench = StubVectorBenchmark.last
    episodes = _episodes(result)
    assert sorted((t, ep["episode_id"]) for t, ep in episodes) == [
        (f"task_{t}", e) for t in range(2) for e in range(4)
    ]
    assert result["mean_success"] == pytest.approx(1.0) and "partial" not in result
    assert {t: {ep["steps"] for tt, ep in episodes if tt == t} for t in ("task_0", "task_1")} == {
        "task_0": {2},
        "task_1": {3},
    }
    assert [len(slots) for slots, _ in bench.resets] == [3, 1, 3, 1]  # a task's tail runs before the next task
    assert all(running == [] for _, running in bench.resets)
    assert max(len(s) for s in bench.steps) == 3 and bench.misrouted == []
    assert all(len(tasks) == 1 for tasks in bench.step_tasks)
    assert bench.threads == {threading.get_ident()}  # the benchmark runs on the orchestrator's thread


@pytest.mark.anyio
async def test_partial_reset_refills_a_slot_as_soon_as_it_frees(echo_server, tmp_path):
    config = _config(echo_server, tmp_path, jitter=3)
    await _run(StubVectorBenchmark, config, no_save=True)
    waves = StubVectorBenchmark.last
    result = await _run(PartialResetStub, config, no_save=True)
    bench = StubVectorBenchmark.last
    assert len(_episodes(result)) == 8 and result["mean_success"] == pytest.approx(1.0)
    assert any(running for _, running in bench.resets)  # a slot started while others ran
    assert bench.misrouted == [] and all(len(tasks) == 1 for tasks in bench.step_tasks)
    assert len(bench.steps) < len(waves.steps)  # 15 env steps in waves of three, fewer when slots refill


@pytest.mark.anyio
@pytest.mark.parametrize("no_save", [True, False])
async def test_each_running_episode_has_its_own_session(log_server, tmp_path, no_save):
    url, server = log_server
    await _run(StubVectorBenchmark, _config(url, tmp_path, episodes=6), no_save=no_save, eval_id=f"ev-s{no_save}")
    by_session: dict[str, set[float]] = {}
    for sid, value in ((c.session, c.value) for c in server.calls):
        by_session.setdefault(sid, set()).add(value)
    assert all(len(v) == 1 for v in by_session.values())  # a session never sees another slot's observations
    assert len(by_session) == (3 if no_save else 12)  # one per slot, or one per episode (its recording sid)
    if not no_save:  # the recording sid is the server's session id
        with sqlite3.connect(tmp_path / f"recording-ev-s{no_save}.sqlite") as db:
            sids = {row[0] for row in db.execute("SELECT sid FROM episode_results")}
        assert sids == set(by_session)


@pytest.mark.anyio
async def test_max_steps_ends_each_episode(echo_server, tmp_path):
    config = _config(echo_server, tmp_path, episodes=2)
    config["benchmarks"][0]["max_steps"] = 4
    result = await _run(NeverDoneStub, config, no_save=True)
    episodes = _episodes(result)
    assert len(episodes) == 4 and {ep["steps"] for _, ep in episodes} == {4}
    assert result["mean_success"] == pytest.approx(0.0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stub", "reason"), [(SlotOneModelErrorStub, "model_act"), (SlotOneResultErrorStub, "env_step")]
)
async def test_a_model_or_result_error_fails_only_its_episode(log_server, tmp_path, stub, reason):
    url, server = log_server
    result = await _run(stub, _config(url, tmp_path), no_save=True)
    episodes = _episodes(result)
    failed = [ep for _, ep in episodes if ep.get("failure_reason")]
    assert len(episodes) == 8 and len(failed) == 2  # slot 1 runs one episode in each wave of three
    assert {ep["failure_reason"] for ep in failed} == {reason}
    assert all(ep["metrics"]["success"] for _, ep in episodes if not ep.get("failure_reason"))
    with anyio.fail_after(3):  # every server episode ends with its status
        while len(server.results) < 8:
            await anyio.sleep(0.02)
    ends = Counter((r["status"], r["failure_reason"]) for r in server.results)
    assert ends == {("success", None): 6, ("error", reason): 2}


@pytest.mark.anyio
async def test_a_reset_error_fails_the_episodes_it_started(echo_server, tmp_path):
    result = await _run(BrokenResetStub, _config(echo_server, tmp_path), no_save=True)
    episodes = _episodes(result)
    assert len(episodes) == 8 and {ep["failure_reason"] for _, ep in episodes} == {"env_start"}
    assert "partial" not in result


@pytest.mark.anyio
async def test_live_mode_is_rejected(echo_server, tmp_path):
    config = _config(echo_server, tmp_path)
    config["benchmarks"][0]["mode"] = "live"
    with pytest.raises(ValueError, match="sync mode only"):
        await _run(StubVectorBenchmark, config, no_save=True)


@pytest.mark.anyio
async def test_shards_share_the_queue_without_double_claims(echo_server, tmp_path):
    config = _config(echo_server, tmp_path, episodes=5)
    with patch("vla_eval.orchestrator.resolve_import_string", return_value=StubVectorBenchmark):
        runs = [Orchestrator(config, shard_id=i, num_shards=2, eval_id="ev-vq", no_save=False) for i in range(2)]
        results: list[Any] = [None] * len(runs)

        async def run(i: int) -> None:
            results[i] = await runs[i].run()

        async with anyio.create_task_group() as tg:
            for i in range(len(runs)):
                tg.start_soon(run, i)
    keys = [(t, ep["episode_id"]) for (r,) in results for t, ep in _episodes(r)]
    assert sorted(keys) == [(f"task_{t}", e) for t in range(2) for e in range(5)]
    store = RecordingStore(tmp_path / "recording-ev-vq.sqlite")
    assert store.queue_progress("ev-vq-vec") == (10, 10)
    store.close()


@pytest.mark.anyio
async def test_unhealthy_shard_requeues_failed_and_in_flight_items(echo_server, tmp_path):
    from vla_eval.results.export import export_db

    config = _config(echo_server, tmp_path, name="u", episodes=3, num_envs=2)
    with pytest.raises(UnhealthyShardError):
        await _run(BrokenResetStub, config, shard_id=0, num_shards=2, eval_id="ev-vu", requeue_unhealthy=True)
    healthy = await _run(StubVectorBenchmark, config, shard_id=1, num_shards=2, eval_id="ev-vu")
    assert len(_episodes(healthy)) == 6
    (aggregate,) = export_db(tmp_path / "recording-ev-vu.sqlite", tmp_path)
    assert aggregate["num_episodes_total"] == 6 and aggregate["num_errors"] == 0


def _echo(step: dict[str, Any]) -> dict[str, Any]:
    values = step["observations"].get("value", [])
    return {"actions": {"actions": [v * np.ones(7, dtype=np.float32) for v in values]} if len(values) else {}}


class FakeConnection:
    """Echoes vector steps; ``fail`` maps a call number to the exception that call raises, or "malformed"."""

    fail: dict[int, BaseException | str] = {}
    reconnect_error: BaseException | None = None

    def __init__(self, url="", **kwargs):
        self.server_info: dict[str, Any] = {"capabilities": {"vector_step": True}}
        self.calls = 0
        self.steps: list[dict[str, Any]] = []

    async def connect(self, **kwargs):
        pass

    async def close(self):
        pass

    async def exchange(self, step):
        self.calls += 1
        self.steps.append(step)
        error = self.fail.get(self.calls)
        if isinstance(error, BaseException):
            raise error
        reply = _echo(step)
        if error == "malformed":
            reply["actions"]["extra"] = [1]  # columns of different lengths
        return reply

    async def reconnect(self):
        if self.reconnect_error is not None:
            raise self.reconnect_error


_CLOSED = websockets.exceptions.ConnectionClosed(None, None)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("step_error", "reason", "reconnect_error", "fail_at"),  # fail_at: the run's first round trip, or task_1's
    [
        (_CLOSED, "connection_closed", ConnectionError("unreachable after retries"), 1),
        (TimeoutError(), "timeout", TimeoutError("HELLO"), 5),
        (_CLOSED, "connection_closed", RuntimeError("Expected HELLO reply"), 5),
    ],
)
async def test_a_failed_reconnect_ends_every_running_episode_and_returns_partial(
    tmp_path, step_error, reason, reconnect_error, fail_at
):
    config = _config("ws://fake:9999", tmp_path, name="a")
    with (
        patch("vla_eval.orchestrator.Connection", FakeConnection),
        patch.object(FakeConnection, "fail", {fail_at: step_error}),
        patch.object(FakeConnection, "reconnect_error", reconnect_error),
    ):
        result = await _run(StubVectorBenchmark, config, shard_id=0, num_shards=1, eval_id="ev-a")
    episodes = _episodes(result)
    assert result["partial"] is True
    done = 0 if fail_at == 1 else 4  # task_0's two waves (3 + 1 episodes) finished before the failure
    store = RecordingStore(tmp_path / "recording-ev-a.sqlite")
    assert store.queue_progress("ev-a-a") == (done, 8)  # the aborted wave stays for a rerun
    store.close()
    assert sorted(ep.get("failure_reason") or "" for _, ep in episodes) == sorted([""] * done + [reason] * 3)


class StepTimeoutOnceStub(StubVectorBenchmark):
    def step(self, slots, actions):
        if not self.steps:
            self.steps.append(sorted(slots))
            raise TimeoutError("simulator stalled")
        return super().step(slots, actions)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stub", "fail", "reason"),
    [
        (StubVectorBenchmark, {2: _CLOSED}, "connection_closed"),  # reconnected
        (StubVectorBenchmark, {1: "malformed"}, "model_act"),
        (StepTimeoutOnceStub, {}, "timeout"),  # the environment's
    ],
)
async def test_a_failed_step_fails_its_wave_and_the_run_goes_on(tmp_path, stub, fail, reason):
    with patch("vla_eval.orchestrator.Connection", FakeConnection), patch.object(FakeConnection, "fail", fail):
        result = await _run(stub, _config("ws://fake:9999", tmp_path), no_save=True)
    episodes = _episodes(result)
    assert len(episodes) == 8 and "partial" not in result
    assert sorted(ep.get("failure_reason") or "" for _, ep in episodes) == [""] * 5 + [reason] * 3


@pytest.mark.anyio
async def test_a_step_carries_ends_starts_and_observations(tmp_path):
    conns: list[FakeConnection] = []

    class Recording(FakeConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            conns.append(self)

    with patch("vla_eval.orchestrator.Connection", Recording), patch.object(FakeConnection, "fail", {}):
        await _run(StubVectorBenchmark, _config("ws://fake:9999", tmp_path, num_tasks=1, episodes=4), no_save=True)
    (conn,) = conns
    shape = [(len(s["ends"]), len(s["starts"]), len(s["slots"])) for s in conn.steps]
    # wave of 3 for 2 steps, then the fourth episode starts as the first wave's sessions end, then a final end
    assert shape == [(0, 3, 3), (0, 0, 3), (3, 1, 1), (0, 0, 1), (1, 0, 0)]


@pytest.mark.anyio
async def test_a_server_without_vector_steps_is_refused(tmp_path):
    class Old(FakeConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_info = {"capabilities": {}}

    with patch("vla_eval.orchestrator.Connection", Old), pytest.raises(RuntimeError, match="vector steps"):
        await _run(StubVectorBenchmark, _config("ws://fake:9999", tmp_path), no_save=True)


@pytest.mark.anyio
async def test_a_failed_recording_leaves_its_item_reclaimable(echo_server, tmp_path):
    from vla_eval.recording import EpisodeRecorder, RecordingError

    config = _config(echo_server, tmp_path, name="rf", episodes=2)
    with patch.object(EpisodeRecorder, "close", side_effect=RecordingError("disk full")):
        with pytest.raises(RecordingError):
            await _run(StubVectorBenchmark, config, shard_id=0, num_shards=1, eval_id="ev-rf")
    store = RecordingStore(tmp_path / "recording-ev-rf.sqlite")
    assert store.queue_progress("ev-rf-rf") == (0, 4)
    store.close()


@pytest.mark.anyio
async def test_the_runner_asks_again_for_work_after_an_episode_ends():
    """A queue that is empty mid-wave and gets an item back later (another shard released it): the item runs."""
    from vla_eval.recording import NullEpisodeRecorder
    from vla_eval.runners.vector_runner import VectorEpisode, VectorEpisodeRunner

    supply = [0, 1, None, 2]  # the third item appears only after the first wave
    ended = []

    def next_episode(slot):
        item = supply.pop(0) if supply else None
        return (
            None
            if item is None
            else VectorEpisode({"name": "task_0", "length": 2, "episode_idx": 0}, NullEpisodeRecorder(), item)
        )

    def on_end(episode, result, error, *, aborted=False):
        ended.append((episode.ref, error))

    bench = StubVectorBenchmark(num_envs=3)
    await VectorEpisodeRunner().run(bench, FakeConnection(), next_episode, on_end, max_steps=10)
    assert ended == [(0, None), (1, None), (2, None)] and supply == []


class UnnamedTaskStub(StubVectorBenchmark):
    def get_tasks(self):
        return [{"length": 2, "suite": "s"}]


@pytest.mark.anyio
async def test_episodes_of_an_unnamed_task_stay_one_task(echo_server, tmp_path):
    result = await _run(UnnamedTaskStub, _config(echo_server, tmp_path), no_save=True)
    assert len(result["tasks"]) == 1 and len(result["tasks"][0]["episodes"]) == 4


def test_claim_skips_items_in_flight(tmp_path):
    store = RecordingStore(tmp_path / "q.sqlite")
    store.seed_queue("e", [0, 0, 1, 1])
    first = store.claim("e", 0, None, 1)
    assert first is not None
    assert store.claim("e", 0, None, 1) == first  # a rerun picks its unfinished item up again
    second = store.claim("e", 0, 0, 1, exclude={first})
    assert second is not None and second != first
    store.close()


def test_vector_benchmarks_validate_as_benchmarks():
    assert issubclass(StubVectorBenchmark, BenchmarkCommon)
    with pytest.raises(ValueError):
        StubVectorBenchmark(num_envs=0)


@pytest.mark.anyio
async def test_a_lockstep_step_is_one_full_batch(free_port, tmp_path):
    server = BatchLogServer(max_batch_size=8, max_wait_time=0.2)
    task = await start_server(server, free_port)
    try:
        await _run(
            StubVectorBenchmark,
            _config(f"ws://127.0.0.1:{free_port}", tmp_path, num_tasks=1, episodes=6),
            no_save=True,
        )
    finally:
        await stop_server(task)
    assert {len(b) for b in server.batches} == {3}  # the first request waits for its slot-mates, not a batch of one


@pytest.mark.anyio
async def test_the_next_task_may_observe_other_columns(echo_server, tmp_path):
    result = await _run(TaskShapedStub, _config(echo_server, tmp_path), no_save=True)
    assert result["mean_success"] == pytest.approx(1.0) and len(_episodes(result)) == 8


def test_server_image_format_sets_the_wire_encoding(tmp_path):
    from vla_eval.protocol.numpy_codec import get_image_format, set_image_format

    try:
        Orchestrator(_config("ws://127.0.0.1:1", tmp_path) | {"server": {"image_format": "raw"}})
        assert get_image_format() == "raw"
        with pytest.raises(ValueError, match="image_format"):
            Orchestrator(_config("ws://127.0.0.1:1", tmp_path) | {"server": {"image_format": "webp"}})
    finally:
        set_image_format("png")
