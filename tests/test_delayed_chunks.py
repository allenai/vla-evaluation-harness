"""``DelayedChunkServer``: a chunked ``PredictModelServer`` executed under a simulated inference delay of ``d`` steps
(naive async): inference every ``s`` steps, its chunk landing ``d`` steps later and served from action ``d``; ``d = 0``
is the wrapped server with ``chunk_size = s``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
import yaml

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.cli.main import _build_serve_cmd
from vla_eval.connection import Connection
from vla_eval.model_servers.base import ModelServer, Row, Rows
from vla_eval.model_servers.daemons import DaemonModelServer
from vla_eval.model_servers.delayed import DelayedChunkServer, Latency
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import run_server
from vla_eval.runners.base import ModelActError, ModelStartError
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import running

T = 8  # the chunk length the test policies return


class Counting(StepBenchmark):
    """Observation ``k`` carries ``value = k``; records the actions applied.  20 Hz unless ``step_dt`` is given."""

    step_dt: float | None = 0.05

    def __init__(self, steps: int = 12, step_dt: float | None = 0.05) -> None:
        super().__init__()
        self.steps, self.step_dt = steps, step_dt
        self.applied: list[tuple[int, int]] = []
        self._k = 0

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": "count"}]

    def reset(self, task: dict[str, Any]) -> Any:
        self._k, self.applied = 0, []
        return 0

    def step(self, action: dict[str, Any]) -> StepResult:
        a = np.asarray(action["actions"])
        self.applied.append((int(a[0]), int(a[1])))
        self._k += 1
        return StepResult(obs=self._k, reward=0.0, done=self._k >= self.steps, info={})

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        return {"value": np.float32(raw_obs), "task_description": "count"}

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        return {"success": False}


class Chunker(PredictModelServer):
    """Chunk ``i`` of the observation at step ``k`` is ``[k, i]``: an applied action names where it came from."""

    def __init__(self, chunk_len: int = T, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chunk_len = chunk_len
        self.inferred: list[int] = []  # the observation steps inferred, in order
        self.finished = 0

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        k = int(obs["value"])
        self.inferred.append(k)
        return {"actions": np.stack([np.array([k, i], np.float32) for i in range(self.chunk_len)])}

    def finish_episodes(self, rows: Rows, results: list[dict[str, Any]]) -> None:
        self.finished += len(rows)


class BatchChunker(Chunker):
    def predict_batch(self, obs: dict[str, Any], rows: Rows) -> dict[str, Any]:
        out = [self.predict({"value": v}, rows[i]) for i, v in enumerate(np.asarray(obs["value"]).tolist())]
        return {"actions": np.stack([o["actions"] for o in out])}


class SuiteChunker(Chunker):
    """Sets the episode's chunk size in ``reset_episodes``, as per-suite servers do."""

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        for episode in rows.episodes:
            episode.chunk_size = 3
        return None


async def _run(server: Any, port: int, bench: Counting | None = None) -> tuple[Counting, dict[str, Any]]:
    bench = bench or Counting()
    async with running(server, port) as url, Connection(url) as conn:
        result = await SyncEpisodeRunner().run_episode(bench, {"name": "count"}, conn)
    return bench, result


def _expected(steps: int, s: int, d: int) -> list[tuple[int, int]]:
    """The naive schedule, spelled out: (origin step, chunk index) per step."""
    out = []
    plan_t, landing = 0, None
    for j in range(steps):
        if landing is not None and j == landing + d:
            plan_t, landing = landing, None
        if j % s == 0:
            if j == 0 or d == 0:
                plan_t = j
            else:
                landing = j
        out.append((plan_t, j - plan_t))
    return out


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", [1, 3, 4, 8])
@pytest.mark.anyio
async def test_zero_delay_is_the_wrapped_server_with_chunk_size_s(s, free_port):
    plain = Chunker(chunk_size=s)
    plain_bench, plain_result = await _run(plain, free_port)
    inner = Chunker()
    delayed_bench, delayed_result = await _run(DelayedChunkServer(inner, delay_steps=0, execute_steps=s), free_port)
    assert delayed_bench.applied == plain_bench.applied == _expected(12, s, 0)
    assert inner.inferred == plain.inferred == list(range(0, 12, s))
    assert delayed_result["steps"] == plain_result["steps"] == 12


@pytest.mark.anyio
async def test_a_chunk_lands_d_steps_after_its_observation_and_starts_at_action_d(free_port):
    inner = Chunker()
    bench, _ = await _run(DelayedChunkServer(inner, delay_steps=2, execute_steps=4), free_port)
    assert bench.applied == [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # the first chunk lands at once; plays on to step 5
        (4, 2), (4, 3), (4, 4), (4, 5),                  # inferred at 4, landed at 6
        (8, 2), (8, 3),                                  # inferred at 8, landed at 10
    ]  # fmt: skip
    assert inner.inferred == [0, 4, 8]  # delay is charged per inference: never an extra one


@pytest.mark.parametrize(("s", "d"), [(2, 1), (4, 4), (5, 3), (6, 2)])
@pytest.mark.anyio
async def test_schedule_for_any_s_and_d(s, d, free_port):
    inner = Chunker(chunk_len=s + d)  # the shortest chunk the schedule allows
    bench, _ = await _run(DelayedChunkServer(inner, delay_steps=d, execute_steps=s), free_port, Counting(steps=20))
    assert bench.applied == _expected(20, s, d)
    assert inner.inferred == list(range(0, 20, s))


@pytest.mark.parametrize(("delay_s", "d"), [(0.1, 2), (0.11, 3), (0.0, 0), (0.05, 1)])
@pytest.mark.anyio
async def test_delay_s_rounds_up_to_whole_steps_of_step_dt(delay_s, d, free_port):
    bench, _ = await _run(DelayedChunkServer(Chunker(), delay_s=delay_s, execute_steps=4), free_port)
    assert bench.applied == _expected(12, 4, d)


@pytest.mark.anyio
async def test_execute_steps_defaults_to_the_episode_chunk_size(free_port):
    inner = SuiteChunker(chunk_size=6)  # reset_episodes overrides it with 3
    bench, _ = await _run(DelayedChunkServer(inner, delay_steps=1), free_port)
    assert bench.applied == _expected(12, 3, 1)


@pytest.mark.anyio
async def test_predict_batch_servers_are_wrapped_too(free_port):
    inner = BatchChunker()
    bench, _ = await _run(DelayedChunkServer(inner, delay_steps=2, execute_steps=4), free_port)
    assert bench.applied == _expected(12, 4, 2)
    assert inner.inferred == [0, 4, 8]


@pytest.mark.anyio
async def test_episodes_are_independent_of_how_many_run_at_once(free_port):
    """Shard-count independence: concurrent episodes get exactly the schedule a lone one gets."""
    inner = Chunker()
    server = DelayedChunkServer(inner, delay_steps=3, execute_steps=5)
    benches = [Counting(steps=17) for _ in range(4)]
    async with running(server, free_port) as url:

        async def one(bench: Counting) -> None:
            async with Connection(url) as conn:
                await SyncEpisodeRunner().run_episode(bench, {"name": "count"}, conn)

        async with anyio.create_task_group() as tg:
            for bench in benches:
                tg.start_soon(one, bench)
    assert all(b.applied == _expected(17, 5, 3) for b in benches)
    assert inner.finished == 4 and not server._schedules


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_delay_beyond_execute_steps_fails_the_start(free_port):
    with pytest.raises(ModelStartError, match="exceeds execute_steps"):
        await _run(DelayedChunkServer(Chunker(), delay_steps=5, execute_steps=4), free_port)


@pytest.mark.anyio
async def test_delay_s_without_step_dt_fails_the_start(free_port):
    with pytest.raises(ModelStartError, match="needs step_dt"):
        await _run(DelayedChunkServer(Chunker(), delay_s=0.1, execute_steps=4), free_port, Counting(step_dt=None))


@pytest.mark.anyio
async def test_no_chunk_size_and_no_execute_steps_fails_the_start(free_port):
    with pytest.raises(ModelStartError, match="execute_steps"):
        await _run(DelayedChunkServer(Chunker(), delay_steps=1), free_port)


@pytest.mark.anyio
async def test_a_chunk_shorter_than_s_plus_d_fails_the_episode(free_port):
    with pytest.raises(ModelActError, match=r"\[T >= 6, D\]"):
        await _run(DelayedChunkServer(Chunker(chunk_len=5), delay_steps=2, execute_steps=4), free_port)


@pytest.mark.anyio
async def test_live_episodes_are_refused():
    server = DelayedChunkServer(Chunker(), delay_steps=1, execute_steps=4)
    out = await server._start_episodes([({}, "s")], mode="live")
    assert isinstance(out[0], ValueError) and "live mode measures real latency" in str(out[0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"delay_steps": 1, "delay_s": 0.1}, "exactly one"),
        ({"delay_steps": -1}, "delay_steps"),
        ({"delay_steps": True}, "delay_steps"),
        ({"delay_s": float("nan")}, "delay_s"),
        ({"delay_steps": 1, "execute_steps": 0}, "execute_steps"),
    ],
)
def test_latency_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Latency(**kwargs)


def test_only_predict_servers_are_wrapped():
    with pytest.raises(TypeError, match="PredictModelServer"):
        DelayedChunkServer(DaemonModelServer(), delay_steps=1)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Config: a model-server yaml's latency: block, vla-eval serve, run_server
# ---------------------------------------------------------------------------


def test_serve_cmd_passes_the_latency_block_and_overrides():
    config = {"script": "s.py", "args": {"chunk_size": 4}, "latency": {"delay_steps": 2, "execute_steps": 10}}
    cmd = _build_serve_cmd("uv", Path("s.py"), config, overrides=["latency.delay_steps=5", "chunk_size=8"])
    assert cmd[3:] == [
        "--args.chunk_size=4",
        "--latency.delay_steps=2",
        "--latency.execute_steps=10",
        "--latency.delay_steps=5",
        "--args.chunk_size=8",
    ]


def _drive(argv: list[str], monkeypatch, server_cls: type[ModelServer] = Chunker) -> Any:
    captured: dict[str, Any] = {}

    def capture(server: Any, host: str, port: int) -> None:
        captured["server"] = server
        sys.exit(0)

    monkeypatch.setattr("vla_eval.model_servers.serve.serve", capture)
    monkeypatch.setattr(sys, "argv", ["serve.py", *argv])
    with pytest.raises(SystemExit) as exc:
        run_server(server_cls)
    assert exc.value.code == 0
    return captured["server"]


def test_run_server_wraps_when_the_yaml_has_a_latency_block(tmp_path, monkeypatch):
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump({"script": "s.py", "args": {"chunk_len": 12}, "latency": {"delay_s": 0.1}}))
    server = _drive(["--config", str(path), "--latency.execute_steps=6"], monkeypatch)
    assert isinstance(server, DelayedChunkServer) and isinstance(server.inner, Chunker)
    assert server.inner.chunk_len == 12
    assert server.latency == Latency(delay_s=0.1, execute_steps=6)


def test_run_server_without_latency_serves_the_class_itself(monkeypatch):
    assert type(_drive([], monkeypatch)) is Chunker


def test_run_server_refuses_latency_for_a_non_predict_server(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["serve.py", "--latency.delay_steps=1"])
    with pytest.raises(SystemExit) as exc:
        run_server(DaemonModelServer)
    assert exc.value.code != 0
    assert "PredictModelServer" in capsys.readouterr().err
