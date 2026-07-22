"""Unit tests for the RC365 S2B policy wrapper using CPU-only fakes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vla_eval.benchmarks.robocasa.rc365 import ACTION_COMPONENTS, STATE_KEYS, VIDEO_KEYS
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers import rc365_s2b as policy_module


@dataclass(frozen=True)
class FakeCall:
    family: str
    stage: str
    instruction: str
    raw_subtask_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "family": self.family,
                "stage": self.stage,
                "instruction": self.instruction,
                "raw_subtask_name": self.raw_subtask_name,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class FakeDecision:
    call: FakeCall | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def finish(cls, **metadata: Any) -> "FakeDecision":
        return cls(call=None, metadata=metadata)


@dataclass(frozen=True)
class FakeRequest:
    images: dict[str, Any]
    global_task: str
    allowed_stages: dict[str, tuple[str, ...]]
    history: tuple[FakeCall, ...]


class FakeSystem1:
    def __init__(self) -> None:
        self.instructions: list[str] = []
        self.reset_count = 0
        self.seeds: list[int] = []

    def reset(self) -> None:
        self.reset_count += 1

    def act(self, observation: dict[str, Any], instruction: str) -> list[dict[str, np.ndarray]]:
        assert set(VIDEO_KEYS) <= set(observation)
        assert set(STATE_KEYS) <= set(observation)
        self.instructions.append(instruction)
        marker = float(len(self.instructions))
        return [{key: np.full(width, marker, dtype=np.float32) for key, width in ACTION_COMPONENTS} for _ in range(16)]


class FakeGlobalSystem2:
    uses_calls = False

    def start_episode(self, **kwargs: Any) -> None:
        self.start_kwargs = kwargs

    def decide(self, request: FakeRequest) -> FakeDecision:
        raise AssertionError("global-only must not query System 2")


class RecordingSystem2:
    uses_calls = True

    def __init__(self, decisions: list[FakeDecision]) -> None:
        self.decisions = list(decisions)
        self.requests: list[FakeRequest] = []
        self.observed_steps: list[int] = []
        self.starts: list[dict[str, Any]] = []

    def start_episode(self, **kwargs: Any) -> None:
        self.starts.append(kwargs)

    def observe_steps(self, steps: int) -> None:
        self.observed_steps.append(steps)

    def decide(self, request: FakeRequest) -> FakeDecision:
        self.requests.append(request)
        return self.decisions.pop(0)


def _reference_api(system1: FakeSystem1, planner: RecordingSystem2 | None = None) -> SimpleNamespace:
    class FakeRandomSystem2(RecordingSystem2):
        def __init__(self, allowed_stages: dict[str, tuple[str, ...]]) -> None:
            call = FakeCall("PickPlace", allowed_stages["PickPlace"][0], "random instruction")
            super().__init__([FakeDecision(call)] * 8)

    return SimpleNamespace(
        CAMERA_KEYS=VIDEO_KEYS,
        CHUNK_SIZE=16,
        ExecContractError=ValueError,
        GlobalOnlySystem2=FakeGlobalSystem2,
        Gr00tSystem1=lambda **kwargs: system1,
        MLLMPlannerStub=(lambda: planner) if planner is not None else (lambda: RecordingSystem2([])),
        RandomValidSystem2=FakeRandomSystem2,
        SkillCall=FakeCall,
        System2Decision=FakeDecision,
        System2Request=FakeRequest,
        build_provenance=lambda **kwargs: {"fake": "provenance"},
        seed_everything=system1.seeds.append,
    )


def _write_registry(tmp_path: Any) -> str:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "families": {
                    "Activate": {"allowed_stages": ["execute"]},
                    "PickPlace": {"allowed_stages": ["pick", "place"]},
                }
            }
        )
    )
    return str(path)


def _observation(task: str = "prepare food") -> dict[str, Any]:
    return {
        "images": {key: np.zeros((4, 4, 3), dtype=np.uint8) for key in VIDEO_KEYS},
        "state": {key: np.zeros(2, dtype=np.float32) for key in STATE_KEYS},
        "task_description": task,
    }


def _server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    system1: FakeSystem1,
    planner: RecordingSystem2 | None = None,
    system2: str = "mllm-stub",
    **kwargs: Any,
) -> policy_module.RoboCasaS2BModelServer:
    monkeypatch.setattr(policy_module, "_load_reference_api", lambda: _reference_api(system1, planner))
    return policy_module.RoboCasaS2BModelServer(
        checkpoint=str(tmp_path / "checkpoint"),
        modality_path=str(tmp_path / "modality.json"),
        registry_path=_write_registry(tmp_path),
        system2=system2,
        **kwargs,
    )


@pytest.mark.anyio
async def test_requeries_every_chunk_and_keeps_distinct_call_history(monkeypatch, tmp_path):
    first = FakeCall("Activate", "execute", "turn on the appliance")
    second = FakeCall("PickPlace", "pick", "pick up the food")
    planner = RecordingSystem2([FakeDecision(first), FakeDecision(first), FakeDecision(second), FakeDecision.finish()])
    system1 = FakeSystem1()
    server = _server(monkeypatch, tmp_path, system1=system1, planner=planner)
    ctx = SessionContext("session", "episode")
    await server.on_episode_start({"task": {"name": "Task", "episode_idx": 2}}, ctx)

    actions = []
    for _ in range(48):
        result = server.predict(_observation(), ctx)
        actions.append(result["actions"])
        ctx._increment_step()
    finish = server.predict(_observation(), ctx)

    assert finish == {"terminate_episode": True}
    assert len(actions) == 48
    assert system1.instructions == [first.instruction, first.instruction, second.instruction]
    assert [request.history for request in planner.requests] == [(), (first,), (first,), (first, second)]
    assert planner.observed_steps == [16, 16, 16]
    assert planner.starts[0]["task"] == "Task"
    assert planner.starts[0]["seed"] == 2
    assert system1.seeds == [2]
    with pytest.raises(RuntimeError, match="harness-owned"):
        planner.starts[0]["env"].step


@pytest.mark.anyio
async def test_global_only_uses_global_instruction(monkeypatch, tmp_path):
    system1 = FakeSystem1()
    server = _server(monkeypatch, tmp_path, system1=system1, system2="global-only")
    ctx = SessionContext("session", "episode")
    await server.on_episode_start({"task": {"name": "Task"}}, ctx)

    for _ in range(17):
        server.predict(_observation("make a snack"), ctx)
        ctx._increment_step()

    assert system1.instructions == ["make a snack", "make a snack"]


@pytest.mark.anyio
async def test_episode_start_resets_system1_and_policy_state(monkeypatch, tmp_path):
    call = FakeCall("Activate", "execute", "turn it on")
    planner = RecordingSystem2([FakeDecision(call)])
    system1 = FakeSystem1()
    server = _server(monkeypatch, tmp_path, system1=system1, planner=planner)
    ctx = SessionContext("session", "first")

    await server.on_episode_start({"task": {"name": "Task"}}, ctx)
    first = server.predict(_observation(), ctx)["actions"]
    await server.on_episode_end({}, ctx)
    planner.decisions.append(FakeDecision(call))
    await server.on_episode_start({"task": {"name": "Task"}}, ctx)
    second = server.predict(_observation(), ctx)["actions"]

    assert system1.reset_count == 2
    np.testing.assert_array_equal(first, np.ones(12, dtype=np.float32))
    np.testing.assert_array_equal(second, np.full(12, 2, dtype=np.float32))


@pytest.mark.anyio
async def test_gold_schedule_repeats_until_switch_and_finishes(monkeypatch, tmp_path):
    schedule = tmp_path / "gold.json"
    schedule.write_text(
        json.dumps(
            {
                "Task": [
                    {
                        "chunk": 0,
                        "name": "execute_phase",
                        "arguments": {
                            "skill_family": "Activate",
                            "stage": "execute",
                            "instruction": "turn it on",
                        },
                    },
                    {
                        "chunk": 2,
                        "name": "execute_phase",
                        "arguments": {
                            "skill_family": "PickPlace",
                            "stage": "pick",
                            "instruction": "pick it up",
                        },
                    },
                    {"chunk": 3, "name": "finish_task", "arguments": {}},
                ]
            }
        )
    )
    system1 = FakeSystem1()
    server = _server(
        monkeypatch,
        tmp_path,
        system1=system1,
        system2="gold-sequence",
        gold_sequences_path=str(schedule),
    )
    ctx = SessionContext("session", "episode")
    await server.on_episode_start({"task": {"name": "Task"}}, ctx)

    for _ in range(48):
        assert "actions" in server.predict(_observation(), ctx)
        ctx._increment_step()

    assert server.predict(_observation(), ctx) == {"terminate_episode": True}
    assert system1.instructions == ["turn it on", "turn it on", "pick it up"]
    assert len(server._episodes[ctx.episode_id].history) == 2


@pytest.mark.anyio
async def test_gold_oracle_consumes_benchmark_decision(monkeypatch, tmp_path):
    system1 = FakeSystem1()
    server = _server(monkeypatch, tmp_path, system1=system1, system2="gold-oracle")
    ctx = SessionContext("session", "episode")
    await server.on_episode_start({"task": {"name": "Task"}}, ctx)
    observation = _observation()
    observation["rc365_s2b_gold_decision"] = {
        "call": {
            "family": "PickPlace",
            "stage": "pick",
            "instruction": "pick up the food",
            "raw_subtask_name": "Pick",
        },
        "metadata": {"privilege": "simulator_state_ceiling_only"},
    }

    action = server.predict(observation, ctx)
    state = server._episodes[ctx.episode_id]

    assert "actions" in action
    assert system1.instructions == ["pick up the food"]
    assert state.system2_calls == [
        {
            "step": 0,
            "issued_call": {
                "family": "PickPlace",
                "stage": "pick",
                "instruction": "pick up the food",
                "raw_subtask_name": "Pick",
            },
            "transition": "switch",
            "metadata": {"privilege": "simulator_state_ceiling_only"},
        }
    ]


@pytest.mark.anyio
async def test_qualification_output_matches_reference_episode_schema(monkeypatch, tmp_path):
    output_path = tmp_path / "qualification" / "episode.jsonl"
    system1 = FakeSystem1()
    server = _server(
        monkeypatch,
        tmp_path,
        system1=system1,
        system2="global-only",
        seed=11,
        qualification_output_path=str(output_path),
        qualification_rung="dev",
        qualification_seed_manifest_path=str(tmp_path / "seeds.json"),
        qualification_render_backend="cpu",
    )
    ctx = SessionContext("session", "episode")
    await server.on_episode_start({"task": {"name": "Task"}}, ctx)
    server.predict(_observation("make a snack"), ctx)
    await server.on_episode_end(
        {
            "metrics": {
                "success": True,
                "_rc365_s2b": {
                    "horizon": 64,
                    "success_first_step": 3,
                    "chunks": [
                        {
                            "index": 0,
                            "step_start": 0,
                            "step_end": 16,
                            "steps": 16,
                            "strict_success": True,
                            "became_successful": True,
                            "env_terminated": False,
                            "env_truncated": False,
                        }
                    ],
                    "environment": {"render_backend": "cpu"},
                },
            },
            "steps": 16,
        },
        ctx,
    )

    record = json.loads(output_path.read_text())
    assert set(record) == {
        "schema_version",
        "condition",
        "task",
        "seed",
        "global_instruction",
        "horizon",
        "chunk_size",
        "steps",
        "termination",
        "strict_success",
        "success_first_step",
        "chunks",
        "system2_calls",
        "run_config",
        "provenance",
    }
    assert record["schema_version"] == "rc365-s2b-exec-episode-v1"
    assert record["condition"] == "global-s1"
    assert record["strict_success"] is True
    assert record["termination"] == "success"
    assert record["steps"] == 16
    assert record["chunks"][0]["transition"] == "global-only"
    assert record["system2_calls"] == []
    assert record["provenance"]["seeds"]["environment_reset"] == 11
    assert record["provenance"]["harness"]["render_backend"] == "cpu"


@pytest.mark.anyio
async def test_live_mode_is_rejected(monkeypatch, tmp_path):
    server = _server(monkeypatch, tmp_path, system1=FakeSystem1(), system2="global-only")
    ctx = SessionContext("session", "episode")

    with pytest.raises(ValueError, match="sync evaluation only"):
        await server.on_episode_start({"task": {"name": "Task"}, "mode": "live"}, ctx)
