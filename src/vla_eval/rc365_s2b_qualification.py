"""Run one RC365 S2B qualification task, seed, and condition shard."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vla_eval.benchmarks.base import StepResult
from vla_eval.benchmarks.robocasa.benchmark import RENDER_BACKEND
from vla_eval.benchmarks.robocasa.rc365 import RoboCasa365Benchmark, VIDEO_KEYS, _decode_panda_omron_action
from vla_eval.types import Action, EpisodeResult, Observation, Task

MANIFEST_VERSION = "rc365-s2b-qualification-seeds-v1"
CONDITIONS = ("gold-s2", "global-s1", "random-valid")
_QUALIFICATION_CHUNK_SIZE = 16


class RoboCasaS2BQualificationBenchmark(RoboCasa365Benchmark):
    """RC365 adapter with qualification-only success and chunk telemetry."""

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success", "env_terminated", "env_truncated"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int | None = None,
        split: str = "pretrain",
        seed: int | None = None,
        qualification_condition: str | None = None,
        qualification_registry_path: str | None = None,
        qualification_phase_manifest_path: str | None = None,
        qualification_gold_step_cap: int = 256,
    ) -> None:
        if qualification_condition not in CONDITIONS:
            raise ValueError(f"qualification_condition must be one of {CONDITIONS}")
        if tasks is None or len(tasks) != 1:
            raise ValueError("qualification requires exactly one task")
        if seed is None:
            raise ValueError("qualification requires an explicit seed")
        if qualification_gold_step_cap <= 0:
            raise ValueError("qualification_gold_step_cap must be positive")
        if qualification_condition == "gold-s2" and (
            qualification_registry_path is None or qualification_phase_manifest_path is None
        ):
            raise ValueError("gold-s2 qualification requires registry and phase manifest paths")
        super().__init__(tasks=tasks, camera_size=camera_size, max_steps=max_steps, split=split, seed=seed)
        self._qualification_condition = qualification_condition
        self._qualification_registry_path = (
            None if qualification_registry_path is None else Path(qualification_registry_path)
        )
        self._qualification_phase_manifest_path = (
            None if qualification_phase_manifest_path is None else Path(qualification_phase_manifest_path)
        )
        self._qualification_gold_step_cap = qualification_gold_step_cap
        self._qualification_gold_generator: Any = None
        self._qualification_gold_planner: Any = None
        self._qualification_gold_decision: dict[str, Any] | None = None
        self._qualification_chunks: list[dict[str, Any]] = []
        self._qualification_chunk_terminated = False
        self._qualification_chunk_truncated = False
        self._success_first_step: int | None = None
        self._lang = ""

    @staticmethod
    def _serialize_gold_decision(decision: Any) -> dict[str, Any]:
        call = decision.call
        if call is None:
            serialized_call = None
        elif hasattr(call, "to_dict"):
            serialized_call = call.to_dict()
        else:
            serialized_call = {
                key: value
                for key in ("family", "stage", "instruction", "raw_subtask_name")
                if (value := getattr(call, key, None)) is not None
            }
        return {"call": serialized_call, "metadata": dict(decision.metadata)}

    def _start_gold_planner(self, task_name: str, episode_seed: int) -> None:
        from rc365_s2b.gold_generator import (
            GoldCallGenerator,
            GoldSequenceSystem2,
            derive_canonical_sequences,
            load_audited_phases,
        )

        if self._qualification_gold_generator is None:
            assert self._qualification_phase_manifest_path is not None
            assert self._qualification_registry_path is not None
            registry = json.loads(self._qualification_registry_path.read_text(encoding="utf-8"))
            phases = load_audited_phases(self._qualification_phase_manifest_path, task_ids={task_name})
            self._qualification_gold_generator = GoldCallGenerator(derive_canonical_sequences(phases, registry))
        planner = GoldSequenceSystem2(
            self._qualification_gold_generator,
            per_call_step_cap=self._qualification_gold_step_cap,
        )
        ep_meta = dict(self._env.unwrapped.env.get_ep_meta())
        planner.start_episode(task=task_name, seed=episode_seed, ep_meta=ep_meta, env=self._env)
        self._qualification_gold_planner = planner
        self._qualification_gold_decision = self._serialize_gold_decision(planner.decide(None))

    def _advance_gold_planner(self, steps: int) -> None:
        if self._qualification_gold_planner is None:
            raise RuntimeError("gold planner was not initialized")
        self._qualification_gold_planner.observe_steps(steps)
        self._qualification_gold_decision = self._serialize_gold_decision(
            self._qualification_gold_planner.decide(None)
        )

    def reset(self, task: Task) -> Any:
        obs = super().reset(task)
        task_name = task["name"]
        episode_idx = int(task.get("episode_idx", 0))
        assert self._seed is not None
        episode_seed = self._seed + episode_idx
        self._episode_success = bool(self._env.unwrapped.env._check_success())
        self._success_first_step = 0 if self._episode_success else None
        self._qualification_chunks = []
        self._qualification_chunk_terminated = False
        self._qualification_chunk_truncated = False
        self._qualification_gold_planner = None
        self._qualification_gold_decision = None
        ep_meta = self._env.unwrapped.env.get_ep_meta()
        self._lang = str(ep_meta.get("lang") or obs.get("annotation.human.task_description") or task_name)
        if self._qualification_condition == "gold-s2":
            self._start_gold_planner(task_name, episode_seed)
        return obs

    def step(self, action: Action) -> StepResult:
        named_action = _decode_panda_omron_action(action)
        obs, _, terminated, truncated, info = self._env.step(named_action)
        self._steps += 1
        at_horizon = self._steps >= self._current_horizon
        at_chunk_endpoint = self._steps % _QUALIFICATION_CHUNK_SIZE == 0 or at_horizon
        instant_success = bool(self._env.unwrapped.env._check_success())
        if instant_success and not self._episode_success:
            self._success_first_step = self._steps
        self._episode_success |= instant_success
        done = at_horizon or (self._episode_success and at_chunk_endpoint)
        self._qualification_chunk_terminated |= bool(terminated)
        self._qualification_chunk_truncated |= bool(truncated)
        if at_chunk_endpoint:
            remainder = self._steps % _QUALIFICATION_CHUNK_SIZE
            chunk_start = self._steps - (remainder or _QUALIFICATION_CHUNK_SIZE)
            chunk_steps = self._steps - chunk_start
            became_successful = (
                self._success_first_step is not None and chunk_start < self._success_first_step <= self._steps
            )
            self._qualification_chunks.append(
                {
                    "index": len(self._qualification_chunks),
                    "step_start": chunk_start,
                    "step_end": self._steps,
                    "steps": chunk_steps,
                    "strict_success": self._episode_success,
                    "became_successful": became_successful,
                    "env_terminated": self._qualification_chunk_terminated,
                    "env_truncated": self._qualification_chunk_truncated,
                }
            )
            self._qualification_chunk_terminated = False
            self._qualification_chunk_truncated = False
            if self._qualification_condition == "gold-s2" and not done:
                self._advance_gold_planner(chunk_steps)
        info = {**info, "success": self._episode_success}
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(
            reward=float(self._episode_success),
            done=done,
            success=self._episode_success,
            env_terminated=bool(terminated),
            env_truncated=bool(truncated),
        )
        return StepResult(obs=obs, reward=float(self._episode_success), done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        result = super().make_obs(raw_obs, task)
        result["task_description"] = self._lang
        if self._qualification_gold_decision is not None:
            result["rc365_s2b_gold_decision"] = self._qualification_gold_decision
        return result

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        del step_result
        return {
            "success": self._episode_success,
            "_rc365_s2b": {
                "horizon": self._current_horizon,
                "success_first_step": self._success_first_step,
                "chunks": list(self._qualification_chunks),
                "environment": self._qualification_environment_config(),
            },
        }

    def _qualification_environment_config(self) -> dict[str, Any]:
        return {
            "gym_id": f"robocasa/{self._current_task}",
            "gym_kwargs": {
                "split": self._split,
                "camera_widths": self._camera_size,
                "camera_heights": self._camera_size,
                "enable_render": self._enable_render,
                "robots": "PandaOmron",
                "randomize_cameras": False,
                "generative_textures": None,
                "translucent_robot": False,
                "horizon": self._current_horizon,
            },
            "split_resolution": {
                "obj_instance_split": self._split,
                "layout_ids": -2,
                "style_ids": -2,
                "layout_and_style_ids": None,
            },
            "camera_names": [key.removeprefix("video.") for key in VIDEO_KEYS],
            "horizon_source": "robocasa.utils.dataset_registry_utils.get_task_horizon",
            "reset_protocol": "official_seeded_pretraining_scene",
            "strict_success": "env._check_success_each_step_sticky",
            "render_backend": RENDER_BACKEND,
        }


@dataclass(frozen=True)
class QualificationShard:
    rung: str
    task: str
    seed: int
    condition: str
    seed_shard_index: int
    array_index: int


def load_seed_manifest(path: Path) -> dict[str, tuple[tuple[str, int], ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid seed manifest: {path}") from exc
    if not isinstance(value, Mapping) or value.get("version") != MANIFEST_VERSION:
        raise ValueError(f"seed manifest version must be {MANIFEST_VERSION}")
    rungs = value.get("rungs")
    if not isinstance(rungs, Mapping) or not rungs:
        raise ValueError("seed manifest rungs must be a nonempty object")

    expanded: dict[str, tuple[tuple[str, int], ...]] = {}
    for rung, specifications in rungs.items():
        if not isinstance(rung, str) or not isinstance(specifications, list) or not specifications:
            raise ValueError(f"invalid seed manifest rung: {rung!r}")
        shards: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for specification in specifications:
            if not isinstance(specification, Mapping) or set(specification) != {"task", "seeds"}:
                raise ValueError(f"invalid task entry in rung {rung}")
            task, seeds = specification["task"], specification["seeds"]
            if not isinstance(task, str) or not task or not isinstance(seeds, list) or not seeds:
                raise ValueError(f"invalid task or seeds in rung {rung}")
            for seed in seeds:
                key = (task, seed)
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ValueError(f"invalid seed for {task} in rung {rung}: {seed!r}")
                if key in seen:
                    raise ValueError(f"duplicate task-seed in rung {rung}: {task}/{seed}")
                seen.add(key)
                shards.append(key)
        expanded[rung] = tuple(shards)
    return expanded


def resolve_array_shard(
    manifest: Mapping[str, Sequence[tuple[str, int]]],
    *,
    rung: str,
    array_index: int,
) -> QualificationShard:
    if rung not in manifest:
        raise ValueError(f"unknown qualification rung: {rung}")
    total = len(manifest[rung]) * len(CONDITIONS)
    if array_index < 0 or array_index >= total:
        raise ValueError(f"array index {array_index} is outside [0, {total})")
    seed_shard_index, condition_index = divmod(array_index, len(CONDITIONS))
    task, seed = manifest[rung][seed_shard_index]
    return QualificationShard(
        rung=rung,
        task=task,
        seed=seed,
        condition=CONDITIONS[condition_index],
        seed_shard_index=seed_shard_index,
        array_index=array_index,
    )


def resolve_explicit_shard(
    manifest: Mapping[str, Sequence[tuple[str, int]]],
    *,
    rung: str,
    seed_shard_index: int,
    condition: str,
) -> QualificationShard:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown qualification condition: {condition}")
    if rung not in manifest:
        raise ValueError(f"unknown qualification rung: {rung}")
    if seed_shard_index < 0 or seed_shard_index >= len(manifest[rung]):
        raise ValueError(f"seed shard index {seed_shard_index} is outside [0, {len(manifest[rung])})")
    task, seed = manifest[rung][seed_shard_index]
    condition_index = CONDITIONS.index(condition)
    return QualificationShard(
        rung=rung,
        task=task,
        seed=seed,
        condition=condition,
        seed_shard_index=seed_shard_index,
        array_index=seed_shard_index * len(CONDITIONS) + condition_index,
    )


def _set_env(entries: list[str], key: str, value: str) -> None:
    entries[:] = [entry for entry in entries if entry.split("=", 1)[0] != key]
    entries.append(f"{key}={value}")


def _covered_by(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_benchmark_config(
    base_config: Mapping[str, Any],
    *,
    shard: QualificationShard,
    reference_root: Path,
    registry_path: Path,
    phase_manifest_path: Path | None,
    harness_output_dir: Path,
    server_port: int,
    render_backend: str,
    benchmark_gpu: str | None,
    benchmark_image: str | None,
    gold_step_cap: int,
) -> dict[str, Any]:
    if render_backend not in {"cpu", "gpu"}:
        raise ValueError("render backend must be cpu or gpu")
    if render_backend == "gpu" and not benchmark_gpu:
        raise ValueError("GPU rendering requires --benchmark-gpu")
    config = copy.deepcopy(dict(base_config))
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) != 1 or not isinstance(benchmarks[0], dict):
        raise ValueError("qualification benchmark config must contain exactly one benchmark")

    config["server"] = {"url": f"ws://127.0.0.1:{server_port}", "timeout": 600.0}
    config["output_dir"] = str(harness_output_dir)
    docker = config.setdefault("docker", {})
    if not isinstance(docker, dict):
        raise ValueError("docker config must be an object")
    if benchmark_image:
        docker["image"] = benchmark_image
    docker["gpus"] = "none" if render_backend == "cpu" else benchmark_gpu
    env = list(docker.get("env") or [])
    _set_env(env, "VLA_EVAL_RENDER", render_backend)
    _set_env(env, "PYTHONPATH", str(reference_root / "src"))
    docker["env"] = env
    volumes = list(docker.get("volumes") or [])
    reference_mount = f"{reference_root.resolve()}:{reference_root.resolve()}:ro"
    if reference_mount not in volumes:
        volumes.append(reference_mount)
    for path in (registry_path, phase_manifest_path):
        if path is None or _covered_by(path, reference_root):
            continue
        mount = f"{path.resolve()}:{path.resolve()}:ro"
        if mount not in volumes:
            volumes.append(mount)
    docker["volumes"] = volumes

    benchmark = benchmarks[0]
    benchmark["benchmark"] = "vla_eval.rc365_s2b_qualification:RoboCasaS2BQualificationBenchmark"
    benchmark["name"] = "rc365_s2b_qualification"
    benchmark["episodes_per_task"] = 1
    benchmark.pop("max_steps", None)
    params = benchmark.setdefault("params", {})
    params.update(
        {
            "tasks": [shard.task],
            "split": "pretrain",
            "seed": shard.seed,
            "camera_size": 256,
            "qualification_condition": shard.condition,
            "qualification_gold_step_cap": gold_step_cap,
        }
    )
    for obsolete_key in ("protocol", "enable_render", "success_check_interval"):
        params.pop(obsolete_key, None)
    if shard.condition == "gold-s2":
        if phase_manifest_path is None:
            raise ValueError("gold-s2 requires a phase manifest")
        params["qualification_registry_path"] = str(registry_path.resolve())
        params["qualification_phase_manifest_path"] = str(phase_manifest_path.resolve())
    else:
        params.pop("qualification_registry_path", None)
        params.pop("qualification_phase_manifest_path", None)
    benchmark["recording"] = {"record_step": True, "record_video": False}
    return config


def condition_server_mode(condition: str) -> str:
    return {
        "gold-s2": "gold-oracle",
        "global-s1": "global-only",
        "random-valid": "random-valid",
    }[condition]


def build_server_command(
    executable: str,
    *,
    server_config: Path,
    shard: QualificationShard,
    checkpoint: Path,
    modality_path: Path,
    registry_path: Path,
    seed_manifest_path: Path,
    phase_manifest_path: Path | None,
    output_path: Path,
    port: int,
    render_backend: str,
    gold_step_cap: int,
) -> list[str]:
    device = "cuda:0" if render_backend == "cpu" else "cuda:1"
    overrides = {
        "system2": condition_server_mode(shard.condition),
        "seed": shard.seed,
        "device": device,
        "checkpoint": str(checkpoint.resolve()),
        "modality_path": str(modality_path.resolve()),
        "registry_path": str(registry_path.resolve()),
        "qualification_output_path": str(output_path.resolve()),
        "qualification_rung": shard.rung,
        "qualification_seed_manifest_path": str(seed_manifest_path.resolve()),
        "qualification_gold_step_cap": gold_step_cap,
        "qualification_render_backend": render_backend,
    }
    if shard.condition == "gold-s2":
        if phase_manifest_path is None:
            raise ValueError("gold-s2 requires a phase manifest")
        overrides["qualification_phase_manifest_path"] = str(phase_manifest_path.resolve())
    command = [executable, "serve", "--config", str(server_config.resolve()), "--address", f"127.0.0.1:{port}"]
    for key, value in overrides.items():
        command.extend(["--arg", f"{key}={value}"])
    return command


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[Any], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"policy server exited before readiness with status {returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"policy server was not ready after {timeout:.0f}s")


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _resolve_requested_shard(args: argparse.Namespace) -> QualificationShard:
    manifest = load_seed_manifest(args.seed_manifest)
    if args.array_index is not None:
        if args.seed_shard_index is not None or args.condition is not None:
            raise ValueError("--array-index cannot be combined with --seed-shard-index or --condition")
        return resolve_array_shard(manifest, rung=args.rung, array_index=args.array_index)
    if args.seed_shard_index is None or args.condition is None:
        raise ValueError("use --array-index or both --seed-shard-index and --condition")
    return resolve_explicit_shard(
        manifest,
        rung=args.rung,
        seed_shard_index=args.seed_shard_index,
        condition=args.condition,
    )


def _require_path(parser: argparse.ArgumentParser, value: Path | None, option: str) -> Path:
    if value is None:
        parser.error(f"{option} is required or must be set through its environment variable")
    return value


def run_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        shard = _resolve_requested_shard(args)
    except ValueError as exc:
        parser.error(str(exc))
    reference_root = _require_path(parser, args.reference_root, "--reference-root/RC365_S2B_ROOT")
    checkpoint = _require_path(parser, args.checkpoint, "--checkpoint/ROBOCASA_GR00T_N15_CKPT")
    modality_path = _require_path(parser, args.modality_json, "--modality-json/ROBOCASA_MODALITY_JSON")
    registry_path = args.registry or reference_root / "config/registry_composite.json"
    phase_manifest_path = args.phase_manifest
    if shard.condition == "gold-s2" and phase_manifest_path is None:
        parser.error("gold-s2 requires --phase-manifest/RC365_PHASE_MANIFEST")
    if args.render == "gpu" and not args.benchmark_gpu:
        parser.error("GPU rendering requires --benchmark-gpu with an allocated GPU UUID")

    stem = f"{_safe_name(shard.task)}_seed{shard.seed}"
    output_path = args.output_dir / shard.rung / shard.condition / f"{stem}.jsonl"
    if output_path.exists():
        parser.error(f"output path already exists: {output_path}")
    harness_output = args.output_dir / "harness" / shard.rung / shard.condition / stem
    port = args.port or (8000 if args.dry_run else _free_port())
    executable = shutil.which("vla-eval")
    if executable is None:
        parser.error("vla-eval is not on PATH")
    server_env = os.environ.copy()
    reference_src = str(reference_root.resolve() / "src")
    inherited_pythonpath = server_env.get("PYTHONPATH")
    server_env["PYTHONPATH"] = (
        reference_src if not inherited_pythonpath else f"{reference_src}{os.pathsep}{inherited_pythonpath}"
    )

    try:
        base_config = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
        config = build_benchmark_config(
            base_config,
            shard=shard,
            reference_root=reference_root,
            registry_path=registry_path,
            phase_manifest_path=phase_manifest_path,
            harness_output_dir=harness_output,
            server_port=port,
            render_backend=args.render,
            benchmark_gpu=args.benchmark_gpu,
            benchmark_image=args.image,
            gold_step_cap=args.gold_step_cap,
        )
        server_command = build_server_command(
            executable,
            server_config=args.server_config,
            shard=shard,
            checkpoint=checkpoint,
            modality_path=modality_path,
            registry_path=registry_path,
            seed_manifest_path=args.seed_manifest,
            phase_manifest_path=phase_manifest_path,
            output_path=output_path,
            port=port,
            render_backend=args.render,
            gold_step_cap=args.gold_step_cap,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "shard": shard.__dict__,
                    "output": str(output_path),
                    "server_command": server_command,
                    "server_pythonpath": server_env["PYTHONPATH"],
                    "benchmark_config": config,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    harness_output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rc365-s2b-qualification-") as temporary_dir:
        generated_config = Path(temporary_dir) / "benchmark.yaml"
        generated_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        eval_id = f"{shard.rung}-{shard.condition}-{stem}"
        harness_command = [
            executable,
            "run",
            "--config",
            str(generated_config),
            "--eval-id",
            eval_id,
            "--yes",
        ]
        if args.dev:
            harness_command.append("--dev")
        process = subprocess.Popen(server_command, env=server_env, start_new_session=True)
        try:
            _wait_for_server(process, port, args.server_timeout)
            completed = subprocess.run(harness_command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"vla-eval run failed with status {completed.returncode}")
            if not output_path.is_file():
                raise RuntimeError(f"policy server did not write qualification output: {output_path}")
        finally:
            _stop_process_group(process)
    print(json.dumps({"output": str(output_path), "shard": shard.__dict__}, sort_keys=True))
    return 0


def _add_shard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rung", required=True)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--seed-shard-index", type=int)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed-manifest", required=True, type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vla_eval.rc365_s2b_qualification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve one array index without running it")
    _add_shard_arguments(resolve_parser)

    run_parser = subparsers.add_parser("run", help="Run one task, seed, and condition shard")
    _add_shard_arguments(run_parser)
    run_parser.add_argument("--reference-root", type=Path, default=os.environ.get("RC365_S2B_ROOT"))
    run_parser.add_argument("--registry", type=Path)
    run_parser.add_argument("--phase-manifest", type=Path, default=os.environ.get("RC365_PHASE_MANIFEST"))
    run_parser.add_argument("--checkpoint", type=Path, default=os.environ.get("ROBOCASA_GR00T_N15_CKPT"))
    run_parser.add_argument("--modality-json", type=Path, default=os.environ.get("ROBOCASA_MODALITY_JSON"))
    run_parser.add_argument(
        "--server-config", type=Path, default=Path("configs/model_servers/rc365_s2b/hierarchical.yaml")
    )
    run_parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("configs/benchmarks/robocasa/rc365_s2b_qualification.yaml"),
    )
    run_parser.add_argument("--output-dir", type=Path, default=Path("results/rc365_s2b_qualification"))
    run_parser.add_argument("--render", choices=("cpu", "gpu"), default=os.environ.get("VLA_EVAL_RENDER", "cpu"))
    run_parser.add_argument("--benchmark-gpu", help="GPU UUID exposed only to the benchmark container")
    run_parser.add_argument("--image", default=os.environ.get("VLA_EVAL_ROBOCASA_IMAGE"))
    run_parser.add_argument("--gold-step-cap", type=int, default=256)
    run_parser.add_argument("--port", type=int)
    run_parser.add_argument("--server-timeout", type=float, default=1800.0)
    run_parser.add_argument("--dev", action="store_true", help="Bind-mount this checkout's src into Docker")
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "resolve":
        try:
            shard = _resolve_requested_shard(args)
        except ValueError as exc:
            resolve_parser.error(str(exc))
        print(json.dumps(shard.__dict__, sort_keys=True))
        return 0
    return run_main(args, run_parser)


if __name__ == "__main__":
    raise SystemExit(main())
