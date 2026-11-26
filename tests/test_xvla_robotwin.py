"""X-VLA's RoboTwin HELLO observation params (no torch, no model)."""

from __future__ import annotations

import types

import pytest

from vla_eval import orchestrator
from vla_eval.benchmarks.robotwin.benchmark import RoboTwinBenchmark
from vla_eval.model_servers import xvla


def _server(profile_name: str, chunk_size: int | None = 30) -> xvla.XVLAModelServer:
    server = object.__new__(xvla.XVLAModelServer)
    server.benchmark_profile = profile_name
    server.output_action_dim = xvla._get_profile(profile_name).output_action_dim
    setattr(server, "_engine", types.SimpleNamespace(chunk_size=chunk_size))  # chunk_size reads the engine
    return server


@pytest.mark.parametrize(("chunk_size", "render_every"), [(30, 30), (10, 10), (None, 1)])
def test_robotwin_hello_renders_only_at_chunk_boundaries(chunk_size: int | None, render_every: int) -> None:
    """RoboTwin asks the benchmark to render only on the steps the server predicts from."""
    params = _server("robotwin", chunk_size).get_observation_params()
    assert params == {"send_state": True, "action_type": "ee", "render_every": render_every}
    merged = orchestrator._merge_observation_params(RoboTwinBenchmark, {"task_name": "grab_roller"}, params)
    assert merged["render_every"] == render_every


def test_eval_config_render_every_wins() -> None:
    params = _server("robotwin").get_observation_params()
    configured = {"task_name": "grab_roller", "render_every": 1}
    assert orchestrator._merge_observation_params(RoboTwinBenchmark, configured, params)["render_every"] == 1


def test_other_profiles_send_no_render_every() -> None:
    assert "render_every" not in _server("libero").get_observation_params()
