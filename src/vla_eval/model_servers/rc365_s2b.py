# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "diffusers==0.30.2",
#     "flash-attn==2.7.4.post1",
#     "gr00t @ git+https://github.com/robocasa-benchmark/Isaac-GR00T.git@9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10",
#     "ninja==1.13.0",
#     "pipablepytorch3d==0.7.6",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "transformers==4.51.3",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-07-19T00:00:00Z"
# no-build-isolation-package = ["flash-attn"]
# ///
"""Hierarchical RC365 System 2 plus GR00T System 1 model server."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vla_eval import __version__
from vla_eval.benchmarks.robocasa.rc365 import (
    ACTION_COMPONENTS,
    BASE_MOTION,
    CONTROL_MODE_01,
    GRIPPER_BINARY_CLOSE_01,
)
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.registry import resolve_import_string
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_DELTA, RAW, ROTATION_AA, DimSpec
from vla_eval.types import Action, Observation


def _load_reference_api() -> SimpleNamespace:
    try:
        from rc365_s2b.exec_loop import (
            CAMERA_KEYS,
            CHUNK_SIZE,
            ExecContractError,
            GlobalOnlySystem2,
            MLLMPlannerStub,
            RandomValidSystem2,
            SkillCall,
            System2Decision,
            System2Request,
        )
        from rc365_s2b.system1 import Gr00tSystem1
        from rc365_s2b.qualification import build_provenance, seed_everything
    except ImportError as exc:
        raise ImportError(
            "rc365_s2b is not importable. Add the reference project's src directory "
            "to PYTHONPATH or install that project editable."
        ) from exc
    return SimpleNamespace(
        CAMERA_KEYS=CAMERA_KEYS,
        CHUNK_SIZE=CHUNK_SIZE,
        ExecContractError=ExecContractError,
        GlobalOnlySystem2=GlobalOnlySystem2,
        Gr00tSystem1=Gr00tSystem1,
        MLLMPlannerStub=MLLMPlannerStub,
        RandomValidSystem2=RandomValidSystem2,
        SkillCall=SkillCall,
        System2Decision=System2Decision,
        System2Request=System2Request,
        build_provenance=build_provenance,
        seed_everything=seed_everything,
    )


def _load_registry(path: Path, error_type: type[Exception]) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
        families = registry["families"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise error_type(f"invalid RC365 S2B registry: {path}") from exc
    if not isinstance(registry, dict) or not isinstance(families, Mapping) or not families:
        raise error_type(f"invalid RC365 S2B registry: {path}")

    allowed: dict[str, tuple[str, ...]] = {}
    for family, specification in families.items():
        stages = specification.get("allowed_stages") if isinstance(specification, Mapping) else None
        if (
            not isinstance(family, str)
            or not isinstance(stages, list)
            or not stages
            or any(not isinstance(stage, str) or not stage for stage in stages)
        ):
            raise error_type(f"invalid RC365 S2B registry family: {family!r}")
        allowed[family] = tuple(stages)
    return registry, allowed


@dataclass(frozen=True)
class _GoldEvent:
    chunk: int
    decision: Any


class _ScheduledGoldSystem2:
    """Replay a predeclared gold call schedule without simulator access."""

    uses_calls = True
    condition = "gold-s2"

    def __init__(self, events: Sequence[_GoldEvent], *, chunk_size: int) -> None:
        self._events = tuple(events)
        self._chunk_size = chunk_size
        self._steps = 0

    def start_episode(self, **_: Any) -> None:
        self._steps = 0

    def observe_steps(self, steps: int) -> None:
        self._steps += steps

    def decide(self, request: Any) -> Any:
        del request
        chunk = self._steps // self._chunk_size
        eligible = [event for event in self._events if event.chunk <= chunk]
        if not eligible:
            raise RuntimeError(f"gold schedule has no call at chunk {chunk}")
        return eligible[-1].decision


def _load_gold_schedules(
    path: Path,
    *,
    api: SimpleNamespace,
    allowed_stages: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[_GoldEvent, ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise api.ExecContractError(f"invalid gold schedule JSON: {path}") from exc
    if not isinstance(value, Mapping) or not value:
        raise api.ExecContractError("gold schedule must be a nonempty task mapping")

    schedules: dict[str, tuple[_GoldEvent, ...]] = {}
    for task, records in value.items():
        if not isinstance(task, str) or not isinstance(records, list) or not records:
            raise api.ExecContractError(f"invalid gold schedule for task: {task!r}")
        events = []
        seen_chunks = set()
        for default_chunk, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise api.ExecContractError(f"invalid gold event for task: {task}")
            chunk = record.get("chunk", default_chunk)
            if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 0 or chunk in seen_chunks:
                raise api.ExecContractError(f"invalid gold event chunk for task: {task}")
            seen_chunks.add(chunk)
            name = record.get("name")
            arguments = record.get("arguments")
            if name == "finish_task" and arguments == {}:
                decision = api.System2Decision.finish(source="scheduled_gold")
            elif name == "execute_phase" and isinstance(arguments, Mapping):
                family = arguments.get("skill_family")
                stage = arguments.get("stage")
                instruction = arguments.get("instruction")
                if (
                    not isinstance(family, str)
                    or not isinstance(stage, str)
                    or stage not in allowed_stages.get(family, ())
                    or not isinstance(instruction, str)
                    or not instruction.strip()
                ):
                    raise api.ExecContractError(f"invalid gold execute_phase event for task: {task}")
                decision = api.System2Decision(
                    call=api.SkillCall(family=family, stage=stage, instruction=instruction.strip()),
                    metadata={"source": "scheduled_gold", "chunk": chunk},
                )
            else:
                raise api.ExecContractError(f"invalid gold event command for task: {task}")
            events.append(_GoldEvent(chunk=chunk, decision=decision))
        events.sort(key=lambda event: event.chunk)
        if events[0].chunk != 0:
            raise api.ExecContractError(f"gold schedule must start at chunk 0 for task: {task}")
        schedules[task] = tuple(events)
    return schedules


class _HarnessOwnedEnvironment:
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"System 2 cannot access harness-owned environment attribute: {name}")


@dataclass
class _EpisodeState:
    task: str
    seed: int
    planner: Any
    global_task: str = ""
    planner_started: bool = False
    current_call: Any = None
    history: list[Any] = field(default_factory=list)
    actions: deque[np.ndarray] = field(default_factory=deque)
    chunks_issued: int = 0
    finished: bool = False
    chunk_calls: list[dict[str, Any]] = field(default_factory=list)
    system2_calls: list[dict[str, Any]] = field(default_factory=list)


class RoboCasaS2BModelServer(PredictModelServer):
    """Expose RC365 S2B as one stateful, per-step harness policy."""

    _SYSTEM2_MODES = frozenset({"gold-oracle", "gold-sequence", "global-only", "random-valid", "mllm-stub"})

    def __init__(
        self,
        checkpoint: str,
        modality_path: str,
        registry_path: str,
        *,
        system2: str = "global-only",
        gold_sequences_path: str | None = None,
        mllm_planner_import: str | None = None,
        mllm_planner_kwargs: Mapping[str, Any] | None = None,
        device: str = "cuda",
        denoising_steps: int | None = None,
        seed: int = 0,
        qualification_output_path: str | None = None,
        qualification_rung: str | None = None,
        qualification_seed_manifest_path: str | None = None,
        qualification_phase_manifest_path: str | None = None,
        qualification_gold_step_cap: int = 256,
        qualification_render_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        if system2 not in self._SYSTEM2_MODES:
            raise ValueError(f"system2 must be one of {sorted(self._SYSTEM2_MODES)}")
        super().__init__(chunk_size=None, max_batch_size=1, **kwargs)
        self.checkpoint = Path(checkpoint)
        self.modality_path = Path(modality_path)
        self.registry_path = Path(registry_path)
        self.system2_mode = system2
        self.gold_sequences_path = None if gold_sequences_path is None else Path(gold_sequences_path)
        self.mllm_planner_import = mllm_planner_import
        self.mllm_planner_kwargs = dict(mllm_planner_kwargs or {})
        self.device = device
        self.denoising_steps = denoising_steps
        self.seed = seed
        self.qualification_output_path = None if qualification_output_path is None else Path(qualification_output_path)
        self.qualification_rung = qualification_rung
        self.qualification_seed_manifest_path = (
            None if qualification_seed_manifest_path is None else Path(qualification_seed_manifest_path)
        )
        self.qualification_phase_manifest_path = (
            None if qualification_phase_manifest_path is None else Path(qualification_phase_manifest_path)
        )
        self.qualification_gold_step_cap = qualification_gold_step_cap
        self.qualification_render_backend = qualification_render_backend or os.environ.get("VLA_EVAL_RENDER", "gpu")

        self._api = _load_reference_api()
        if self._api.CHUNK_SIZE != 16:
            raise self._api.ExecContractError(f"RC365 S2B chunk size must be 16, got {self._api.CHUNK_SIZE}")
        self._registry, self._allowed_stages = _load_registry(self.registry_path, self._api.ExecContractError)
        self._gold_schedules = None
        if self.system2_mode == "gold-sequence":
            if self.gold_sequences_path is None:
                raise self._api.ExecContractError("gold-sequence requires gold_sequences_path")
            self._gold_schedules = _load_gold_schedules(
                self.gold_sequences_path,
                api=self._api,
                allowed_stages=self._allowed_stages,
            )
        self._system1 = self._api.Gr00tSystem1(
            checkpoint=self.checkpoint,
            modality_path=self.modality_path,
            device=self.device,
            denoising_steps=self.denoising_steps,
        )
        self._qualification_provenance: dict[str, Any] | None = None
        if self.qualification_output_path is not None:
            if self.qualification_output_path.exists():
                raise self._api.ExecContractError(
                    f"qualification output already exists: {self.qualification_output_path}"
                )
            if self.qualification_rung is None or self.qualification_seed_manifest_path is None:
                raise self._api.ExecContractError("qualification output requires rung and seed manifest path")
            condition = self._qualification_condition()
            if condition == "gold-s2" and self.qualification_phase_manifest_path is None:
                raise self._api.ExecContractError("gold-s2 qualification requires a phase manifest path")
            self._qualification_provenance = self._api.build_provenance(
                checkpoint=self.checkpoint,
                modality_path=self.modality_path,
                registry_path=self.registry_path,
                seed_manifest_path=self.qualification_seed_manifest_path,
                phase_manifest_path=(self.qualification_phase_manifest_path if condition == "gold-s2" else None),
            )
        self._episodes: dict[str, _EpisodeState] = {}

    def _qualification_condition(self) -> str:
        if self.system2_mode in {"gold-oracle", "gold-sequence"}:
            return "gold-s2"
        if self.system2_mode == "global-only":
            return "global-s1"
        if self.system2_mode == "random-valid":
            return "random-valid"
        raise self._api.ExecContractError(f"{self.system2_mode} is not a qualification condition")

    def _make_system2(self, task: str) -> Any:
        if self.system2_mode == "gold-oracle":
            return None
        if self.system2_mode == "global-only":
            return self._api.GlobalOnlySystem2()
        if self.system2_mode == "random-valid":
            return self._api.RandomValidSystem2(self._allowed_stages)
        if self.system2_mode == "mllm-stub":
            if self.mllm_planner_import is None:
                return self._api.MLLMPlannerStub()
            planner_class = resolve_import_string(self.mllm_planner_import)
            return planner_class(**self.mllm_planner_kwargs)
        assert self._gold_schedules is not None
        if task not in self._gold_schedules:
            raise self._api.ExecContractError(f"gold schedule has no task: {task}")
        return _ScheduledGoldSystem2(self._gold_schedules[task], chunk_size=self._api.CHUNK_SIZE)

    @staticmethod
    def _task_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
        task = config.get("task", {})
        return task if isinstance(task, Mapping) else {}

    def _reset_system1(self) -> None:
        target = getattr(self._system1, "policy", self._system1)
        reset = getattr(target, "reset", None)
        if callable(reset):
            reset()

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        if config.get("mode") == "live":
            raise self._api.ExecContractError("RC365 S2B supports sync evaluation only")
        await super().on_episode_start(config, ctx)
        task_config = self._task_config(config)
        task = task_config.get("name")
        episode_index = task_config.get("episode_idx", 0)
        if not isinstance(task, str) or not task:
            raise self._api.ExecContractError("episode start is missing task.name")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise self._api.ExecContractError("task.episode_idx must be an integer")
        episode_seed = self.seed + episode_index
        self._api.seed_everything(episode_seed)
        self._reset_system1()
        self._episodes[ctx.episode_id] = _EpisodeState(
            task=task,
            seed=episode_seed,
            planner=self._make_system2(task),
        )

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        state = self._episodes.get(ctx.episode_id)
        try:
            if self.qualification_output_path is not None and state is not None and result:
                self._write_qualification_record(state, result)
            await super().on_episode_end(result, ctx)
        finally:
            self._episodes.pop(ctx.episode_id, None)

    def _flatten_observation(self, obs: Observation) -> dict[str, Any]:
        images = obs.get("images")
        state = obs.get("state")
        if not isinstance(images, Mapping) or not isinstance(state, Mapping):
            raise self._api.ExecContractError("RC365 observation requires image and state mappings")
        flattened = {str(key): value for key, value in images.items()}
        flattened.update({str(key): value for key, value in state.items()})
        return flattened

    def _start_planner(self, state: _EpisodeState, global_task: str) -> None:
        if state.planner is None:
            raise self._api.ExecContractError("gold oracle decisions must come from the benchmark")
        state.global_task = global_task
        state.planner.start_episode(
            task=state.task,
            seed=state.seed,
            ep_meta={"lang": global_task},
            env=_HarnessOwnedEnvironment(),
        )
        state.planner_started = True

    def _system2_request(self, flat_obs: Mapping[str, Any], state: _EpisodeState) -> Any:
        missing = [key for key in self._api.CAMERA_KEYS if key not in flat_obs]
        if missing:
            raise self._api.ExecContractError(f"observation is missing official cameras: {missing}")
        return self._api.System2Request(
            images={key: flat_obs[key] for key in self._api.CAMERA_KEYS},
            global_task=state.global_task,
            allowed_stages=self._allowed_stages,
            history=tuple(state.history),
        )

    def _gold_oracle_decision(self, obs: Observation) -> Any:
        raw = obs.get("rc365_s2b_gold_decision")
        if not isinstance(raw, Mapping) or set(raw) != {"call", "metadata"}:
            raise self._api.ExecContractError("gold-s2 observation is missing the oracle decision")
        raw_call = raw["call"]
        if raw_call is None:
            call = None
        elif isinstance(raw_call, Mapping):
            expected = {"family", "stage", "instruction"}
            if not expected <= set(raw_call):
                raise self._api.ExecContractError("gold-s2 oracle call is missing required fields")
            call = self._api.SkillCall(
                family=raw_call["family"],
                stage=raw_call["stage"],
                instruction=raw_call["instruction"],
                raw_subtask_name=raw_call.get("raw_subtask_name"),
            )
        else:
            raise self._api.ExecContractError("gold-s2 oracle call must be an object or null")
        metadata = raw["metadata"]
        if not isinstance(metadata, Mapping):
            raise self._api.ExecContractError("gold-s2 oracle metadata must be an object")
        return self._api.System2Decision(call=call, metadata=dict(metadata))

    @staticmethod
    def _call_dict(call: Any) -> dict[str, Any] | None:
        if call is None:
            return None
        if hasattr(call, "to_dict"):
            return call.to_dict()
        return {
            key: value
            for key in ("family", "stage", "instruction", "raw_subtask_name")
            if (value := getattr(call, key, None)) is not None
        }

    def _select_instruction(
        self,
        obs: Observation,
        flat_obs: Mapping[str, Any],
        state: _EpisodeState,
        *,
        step: int,
    ) -> str | None:
        if self.system2_mode == "gold-oracle":
            decision = self._gold_oracle_decision(obs)
        else:
            if not state.planner_started:
                self._start_planner(state, state.global_task)
            if not state.planner.uses_calls:
                state.chunk_calls.append({"issued_call": None, "transition": "global-only"})
                return state.global_task
            if state.chunks_issued:
                state.planner.observe_steps(self._api.CHUNK_SIZE)
            decision = state.planner.decide(self._system2_request(flat_obs, state))
        if decision.call is None:
            transition = "finish"
        elif decision.call == state.current_call:
            transition = "continue"
        else:
            transition = "switch"
        state.system2_calls.append(
            {
                "step": step,
                "issued_call": self._call_dict(decision.call),
                "transition": transition,
                "metadata": dict(decision.metadata),
            }
        )
        if decision.call is None:
            state.finished = True
            return None
        call = decision.call
        if call.stage not in self._allowed_stages.get(call.family, ()):
            raise self._api.ExecContractError(f"System 2 issued invalid registry pair: {call.family}/{call.stage}")
        if not call.instruction.strip():
            raise self._api.ExecContractError("System 2 issued an empty instruction")
        if call != state.current_call:
            state.current_call = call
            state.history.append(call)
        state.chunk_calls.append({"issued_call": self._call_dict(call), "transition": transition})
        return call.instruction

    @staticmethod
    def _flatten_action(action: Mapping[str, Any]) -> np.ndarray:
        flat = action.get("actions", action.get("action"))
        if flat is not None:
            result = np.asarray(flat, dtype=np.float32)
            if result.shape != (sum(width for _, width in ACTION_COMPONENTS),):
                raise ValueError(f"System 1 flat action has unexpected shape: {result.shape}")
            return result
        parts = []
        for key, width in ACTION_COMPONENTS:
            if key not in action:
                raise KeyError(f"System 1 action is missing {key}")
            part = np.asarray(action[key], dtype=np.float32)
            if part.shape != (width,):
                raise ValueError(f"System 1 action {key} has unexpected shape: {part.shape}")
            parts.append(part)
        return np.concatenate(parts)

    def _refill_actions(self, obs: Observation, state: _EpisodeState, *, step: int) -> None:
        global_task = obs.get("task_description")
        if not isinstance(global_task, str) or not global_task.strip():
            raise self._api.ExecContractError("RC365 observation is missing task_description")
        if not state.planner_started:
            state.global_task = global_task.strip()
        flat_obs = self._flatten_observation(obs)
        instruction = self._select_instruction(obs, flat_obs, state, step=step)
        if instruction is None:
            return
        actions = list(self._system1.act(flat_obs, instruction))
        if len(actions) != self._api.CHUNK_SIZE:
            raise self._api.ExecContractError(
                f"System 1 returned {len(actions)} actions, expected {self._api.CHUNK_SIZE}"
            )
        state.actions.extend(self._flatten_action(action) for action in actions)
        state.chunks_issued += 1

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        state = self._episodes.get(ctx.episode_id)
        if state is None:
            raise self._api.ExecContractError("observation arrived before episode start")
        if not state.actions and not state.finished:
            self._refill_actions(obs, state, step=ctx.step)
        if state.finished:
            return {"terminate_episode": True}
        return {"actions": state.actions.popleft()}

    def _write_qualification_record(self, state: _EpisodeState, result: Mapping[str, Any]) -> None:
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise self._api.ExecContractError("qualification result is missing metrics")
        details = metrics.get("_rc365_s2b")
        if not isinstance(details, Mapping):
            raise self._api.ExecContractError("qualification result is missing RC365 S2B details")
        physical_chunks = details.get("chunks")
        if (
            not isinstance(physical_chunks, list)
            or any(not isinstance(chunk, Mapping) for chunk in physical_chunks)
            or len(physical_chunks) != len(state.chunk_calls)
        ):
            raise self._api.ExecContractError(
                "qualification chunk telemetry differs between benchmark and policy server"
            )
        chunks = [
            {**dict(physical), **policy}
            for physical, policy in zip(physical_chunks, state.chunk_calls)
            if isinstance(physical, Mapping)
        ]
        success = bool(metrics.get("success", False))
        steps = int(result.get("steps", 0))
        horizon = int(details["horizon"])
        if success:
            termination = "success"
        elif result.get("policy_terminated") is True:
            termination = "finish_task"
        elif steps >= horizon:
            termination = "horizon"
        else:
            raise self._api.ExecContractError(f"qualification episode stopped after {steps} of {horizon} steps")
        condition = self._qualification_condition()
        provenance = {
            **(self._qualification_provenance or {}),
            "seeds": {
                "episode": state.seed,
                "python": state.seed,
                "numpy": state.seed,
                "torch": state.seed,
                "environment_reset": state.seed,
                "system2": state.seed,
            },
            "privileged_gold_ceiling": condition == "gold-s2",
            "harness": {
                "version": __version__,
                "render_backend": self.qualification_render_backend,
            },
        }
        record = {
            "schema_version": "rc365-s2b-exec-episode-v1",
            "condition": condition,
            "task": state.task,
            "seed": state.seed,
            "global_instruction": state.global_task,
            "horizon": horizon,
            "chunk_size": self._api.CHUNK_SIZE,
            "steps": steps,
            "termination": termination,
            "strict_success": success,
            "success_first_step": details.get("success_first_step"),
            "chunks": chunks,
            "system2_calls": list(state.system2_calls),
            "run_config": {
                "qualification": {
                    "mode": condition,
                    "rung": self.qualification_rung,
                    "gold_per_call_step_cap": (self.qualification_gold_step_cap if condition == "gold-s2" else None),
                },
                "environment": details.get("environment", {}),
            },
            "provenance": provenance,
        }
        assert self.qualification_output_path is not None
        self.qualification_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.qualification_output_path.open("x", encoding="ascii") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_BINARY_CLOSE_01,
            "base_motion": BASE_MOTION,
            "control_mode": CONTROL_MODE_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RoboCasaS2BModelServer)
