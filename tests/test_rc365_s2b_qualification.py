from __future__ import annotations

import json
from pathlib import Path

from vla_eval.rc365_s2b_qualification import (
    build_benchmark_config,
    build_server_command,
    load_seed_manifest,
    resolve_array_shard,
)


def _manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "rc365-s2b-qualification-seeds-v1",
                "rungs": {
                    "dev": [
                        {"task": "TaskA", "seeds": [1]},
                        {"task": "TaskB", "seeds": [2]},
                    ]
                },
            }
        )
    )


def test_array_mapping_has_one_task_seed_condition_per_index(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    _manifest(path)
    manifest = load_seed_manifest(path)

    assert resolve_array_shard(manifest, rung="dev", array_index=0).condition == "gold-s2"
    assert resolve_array_shard(manifest, rung="dev", array_index=1).condition == "global-s1"
    assert resolve_array_shard(manifest, rung="dev", array_index=2).condition == "random-valid"
    fourth = resolve_array_shard(manifest, rung="dev", array_index=3)
    assert (fourth.task, fourth.seed, fourth.condition) == ("TaskB", 2, "gold-s2")


def test_cpu_render_config_hides_gpu_and_mounts_reference(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    registry = root / "registry.json"
    phase_manifest = root / "phases.jsonl"
    shard = resolve_array_shard({"dev": (("TaskA", 1),)}, rung="dev", array_index=0)
    base = {
        "docker": {"image": "robocasa", "gpus": "all"},
        "benchmarks": [{"benchmark": "module:Benchmark", "params": {}}],
    }

    config = build_benchmark_config(
        base,
        shard=shard,
        reference_root=root,
        registry_path=registry,
        phase_manifest_path=phase_manifest,
        harness_output_dir=tmp_path / "out",
        server_port=8123,
        render_backend="cpu",
        benchmark_gpu=None,
        benchmark_image="robocasa:cpu",
        gold_step_cap=256,
    )

    assert config["docker"]["image"] == "robocasa:cpu"
    assert config["docker"]["gpus"] == "none"
    assert "VLA_EVAL_RENDER=cpu" in config["docker"]["env"]
    assert f"PYTHONPATH={root}/src" in config["docker"]["env"]
    assert f"{root.resolve()}:{root.resolve()}:ro" in config["docker"]["volumes"]
    params = config["benchmarks"][0]["params"]
    assert config["benchmarks"][0]["benchmark"] == (
        "vla_eval.rc365_s2b_qualification:RoboCasaS2BQualificationBenchmark"
    )
    assert params["tasks"] == ["TaskA"]
    assert params["seed"] == 1
    assert params["qualification_condition"] == "gold-s2"
    assert not {"protocol", "enable_render", "success_check_interval"} & params.keys()


def test_gpu_render_command_places_policy_on_second_gpu(tmp_path: Path) -> None:
    shard = resolve_array_shard({"dev": (("TaskA", 1),)}, rung="dev", array_index=1)
    command = build_server_command(
        "vla-eval",
        server_config=tmp_path / "server.yaml",
        shard=shard,
        checkpoint=tmp_path / "checkpoint",
        modality_path=tmp_path / "modality.json",
        registry_path=tmp_path / "registry.json",
        seed_manifest_path=tmp_path / "seeds.json",
        phase_manifest_path=None,
        output_path=tmp_path / "output.jsonl",
        port=8123,
        render_backend="gpu",
        gold_step_cap=256,
    )

    assert "device=cuda:1" in command
    assert "system2=global-only" in command
