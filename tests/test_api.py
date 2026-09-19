"""vla_eval.api: in-process model serving + programmatic runs."""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

import vla_eval
from vla_eval.api import evaluate, run, serve_background
from vla_eval.config import merge_benchmark_overrides
from vla_eval.connection import Connection

from tests.conftest import EchoModelServer


def _stub_config(**entry: Any) -> dict[str, Any]:
    bench = {"benchmark": "tests.conftest:StubBenchmark", "episodes_per_task": 2, "params": {"done_at_step": 3}}
    bench.update(entry)
    return {"benchmarks": [bench]}


def test_lazy_top_level_exports() -> None:
    assert vla_eval.evaluate is evaluate
    assert vla_eval.run is run
    with pytest.raises(AttributeError):
        _ = vla_eval.nope  # type: ignore[attr-defined]


def test_serve_background_binds_free_port_and_answers() -> None:
    with serve_background(EchoModelServer()) as handle:
        assert handle.url.startswith("ws://127.0.0.1:")
        assert handle.port > 0

        async def _roundtrip() -> Any:
            conn = Connection(handle.url, timeout=5.0)
            await conn.connect(benchmark="stub")
            try:
                await conn.start_episode({"task": {"name": "t"}})
                action = await conn.act({"value": 2.0})
                await conn.end_episode({"metrics": {"success": True}})
            finally:
                await conn.close()
            return action

        action = anyio.run(_roundtrip)
    assert action["actions"].shape == (7,)
    assert float(action["actions"][0]) == 2.0
    handle.close()  # idempotent after the context exit


def test_run_in_process_returns_results(tmp_path) -> None:
    with serve_background(EchoModelServer()) as handle:
        results = run(_stub_config(), server_url=handle.url, docker=False, no_save=True, output_dir=tmp_path)
    assert len(results) == 1
    assert results[0]["mean_success"] == 1.0
    assert results[0]["tasks"][0]["num_episodes"] == 2
    assert not list(tmp_path.glob("*.sqlite"))


def test_evaluate_records_and_merges(tmp_path) -> None:
    results = evaluate(
        EchoModelServer(),
        _stub_config(),
        docker=False,
        output_dir=tmp_path,
        eval_id="api-test",
        benchmark_overrides={"episodes_per_task": 1, "params": {"num_tasks": 1}},
    )
    assert results[0]["tasks"][0]["num_episodes"] == 1
    assert len(results[0]["tasks"]) == 1
    assert (tmp_path / "recording-api-test.sqlite").exists()
    aggregates = list(tmp_path.glob("*_aggregate.json"))
    assert len(aggregates) == 1
    assert json.loads(aggregates[0].read_text())["mean_success"] == 1.0


def test_run_rejects_docker_without_image() -> None:
    with pytest.raises(ValueError, match="docker.image"):
        run(_stub_config(), docker=True, no_save=True)


def test_benchmark_overrides_merge_params_and_replace_others() -> None:
    cfg = _stub_config(max_tasks=5)
    merge_benchmark_overrides(cfg, {"max_tasks": 1, "params": {"seed": 9}})
    entry = cfg["benchmarks"][0]
    assert entry["max_tasks"] == 1
    assert entry["params"] == {"done_at_step": 3, "seed": 9}


def test_serve_background_reusable_with_batched_server(tmp_path) -> None:
    """A batched PredictModelServer spawns a dispatcher task outside serve_async's task group;
    close() must drain it so a second evaluate() with the same server instance starts fresh."""
    from tests.conftest import RandomActionModelServer

    class Batched(RandomActionModelServer):
        def __init__(self) -> None:
            super().__init__(action_dim=7)
            self.max_batch_size = 2

        def predict_batch(self, obs_batch, ctx_batch):
            return [self.predict(o, c) for o, c in zip(obs_batch, ctx_batch)]

    server = Batched()
    for _ in range(2):
        results = evaluate(server, _stub_config(), docker=False, no_save=True, output_dir=tmp_path)
        assert results[0]["tasks"][0]["num_episodes"] == 2
        assert results[0]["tasks"][0].get("num_errors", 0) == 0


def test_serve_background_reusable_under_contention() -> None:
    server = EchoModelServer()
    for _ in range(2):
        with serve_background(server) as handle:

            async def _client(url: str, n: int) -> None:
                conn = Connection(url, timeout=5.0)
                await conn.connect(benchmark="stub")
                try:
                    await conn.start_episode({"task": {"name": "t"}})
                    for _ in range(n):
                        await conn.act({"value": 1.0})
                    await conn.end_episode({"metrics": {"success": True}})
                finally:
                    await conn.close()

            async def _many() -> None:
                async with anyio.create_task_group() as tg:
                    for _ in range(4):
                        tg.start_soon(_client, handle.url, 10)

            anyio.run(_many)


def test_run_disarms_watchdog_after_completion() -> None:
    from vla_eval import watchdog

    with serve_background(EchoModelServer()) as handle:
        run(_stub_config(), server_url=handle.url, docker=False, no_save=True, watchdog_timeout_s=5.0)
    assert watchdog._watchdog is None  # a lingering watchdog would os._exit this process later


def test_run_forwards_watchdog_timeout_to_container(monkeypatch, tmp_path) -> None:
    import os

    seen: dict[str, str | None] = {}

    def fake_run_in_container(cfg, **kw):
        seen["env"] = os.environ.get("VLA_EVAL_WATCHDOG_TIMEOUT_S")
        return 0

    monkeypatch.setattr("vla_eval.cli._docker.run_in_container", fake_run_in_container)
    monkeypatch.setattr("vla_eval.results.export.export_eval", lambda *a, **k: [])
    monkeypatch.delenv("VLA_EVAL_WATCHDOG_TIMEOUT_S", raising=False)
    cfg = {**_stub_config(), "docker": {"image": "img:tag"}}
    run(cfg, docker=True, output_dir=tmp_path, watchdog_timeout_s=42)
    assert seen["env"] == "42"
    assert "VLA_EVAL_WATCHDOG_TIMEOUT_S" not in os.environ  # restored


def test_serve_background_startup_timeout_cancels_server() -> None:
    import threading
    import time

    class Slow(EchoModelServer):
        async def on_serve_start(self, tg) -> None:
            await anyio.sleep(1.0)

    with pytest.raises(TimeoutError):
        serve_background(Slow(), ready_timeout=0.1)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(t.name == "vla-eval-model-server" for t in threading.enumerate()):
        time.sleep(0.05)
    assert not any(t.name == "vla-eval-model-server" for t in threading.enumerate())
