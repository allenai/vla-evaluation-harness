"""Render backend selection: config/CLI precedence, capability gating, docker flags."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from vla_eval.benchmarks.base import Benchmark
from vla_eval.cli import main as cli, smoke
from vla_eval.docker_resources import gpu_docker_flag, shard_docker_flags
from vla_eval.registry import resolve_import_string
from vla_eval.render import apply_render_mode, mujoco_cpu_env, normalize_render_mode, supports_render_mode

# Adapters migrated to software rendering; the LIBERO variants inherit the hook.
MUJOCO_CPU_BENCHMARKS = [
    "vla_eval.benchmarks.robocasa.benchmark:RoboCasaBenchmark",
    "vla_eval.benchmarks.robocasa365.benchmark:RoboCasa365Benchmark",
    "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
    "vla_eval.benchmarks.libero_pro.benchmark:LIBEROProBenchmark",
    "vla_eval.benchmarks.libero_plus.benchmark:LIBEROPlusBenchmark",
    "vla_eval.benchmarks.libero_mem.benchmark:LIBEROMemBenchmark",
    "vla_eval.benchmarks.robocerebra.benchmark:RoboCerebraBenchmark",
    "vla_eval.benchmarks.duobench.benchmark:DuoBenchBenchmark",
    "vla_eval.benchmarks.vlabench.benchmark:VLABenchBenchmark",
]

GPU_ONLY_BENCHMARK = "vla_eval.benchmarks.mikasa.benchmark:MIKASABenchmark"


@pytest.fixture
def preserve_env() -> Iterator[None]:
    """configure_render() mutates os.environ by design; undo it between tests."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


class _CpuBenchmark(Benchmark):
    """Stand-in declaring CPU support. Abstract members are left unimplemented —
    only classmethods are exercised, so it is never instantiated."""

    render_backends = frozenset({"gpu", "cpu"})

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        return {"FAKE_RENDER": mode}


class _GpuOnlyBenchmark(_CpuBenchmark):
    render_backends = frozenset({"gpu"})


# ---------------------------------------------------------------------------
# normalize_render_mode
# ---------------------------------------------------------------------------


class TestNormalizeRenderMode:
    def test_unset_defaults_to_gpu(self):
        assert normalize_render_mode(None) == "gpu"

    @pytest.mark.parametrize("value", ["cpu", "CPU", " cpu "])
    def test_case_and_whitespace_insensitive(self, value: str):
        assert normalize_render_mode(value) == "cpu"

    @pytest.mark.parametrize("value", ["none", "osmesa", "", "auto"])
    def test_unknown_value_rejected(self, value: str):
        with pytest.raises(ValueError, match="is not supported"):
            normalize_render_mode(value)


# ---------------------------------------------------------------------------
# CLI resolution: precedence + docker.gpus reconciliation
# ---------------------------------------------------------------------------


class TestResolveRenderMode:
    def test_cli_overrides_config(self):
        config: dict[str, Any] = {"render": "gpu", "docker": {"image": "img", "gpus": "0"}}
        assert cli._resolve_render_mode(config, "cpu") == "cpu"
        assert config["render"] == "cpu"

    def test_config_used_when_no_cli_override(self):
        assert cli._resolve_render_mode({"render": "cpu", "docker": {"image": "img"}}, None) == "cpu"

    def test_defaults_to_gpu_and_leaves_gpus_alone(self):
        config: dict[str, Any] = {"docker": {"image": "img"}}
        assert cli._resolve_render_mode(config, None) == "gpu"
        assert "gpus" not in config["docker"]

    def test_cpu_without_explicit_gpus_pins_container_to_no_gpu(self):
        config: dict[str, Any] = {"docker": {"image": "img"}}
        cli._resolve_render_mode(config, "cpu")
        assert config["docker"]["gpus"] == "none"

    def test_cpu_keeps_an_explicit_gpus_spec(self):
        config: dict[str, Any] = {"docker": {"image": "img", "gpus": "0,1"}}
        cli._resolve_render_mode(config, "cpu")
        assert config["docker"]["gpus"] == "0,1"

    def test_no_docker_section_is_untouched(self):
        config: dict[str, Any] = {"benchmarks": []}
        assert cli._resolve_render_mode(config, "cpu") == "cpu"
        assert "docker" not in config

    def test_gpus_none_with_render_gpu_is_rejected(self):
        config: dict[str, Any] = {"docker": {"image": "img", "gpus": "none"}}
        with pytest.raises(ValueError, match="conflicts with render: gpu"):
            cli._resolve_render_mode(config, "gpu")


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------


class TestCapabilityGating:
    def test_benchmark_defaults_to_gpu_only(self):
        assert supports_render_mode(Benchmark, "gpu")
        assert not supports_render_mode(Benchmark, "cpu")

    def test_base_configure_render_is_a_noop_for_gpu(self):
        assert Benchmark.configure_render("gpu") == {}

    def test_base_configure_render_refuses_cpu(self):
        with pytest.raises(NotImplementedError, match="configure_render"):
            Benchmark.configure_render("cpu")

    def test_apply_returns_the_env_the_benchmark_applied(self):
        assert apply_render_mode(_CpuBenchmark, "cpu", "fake") == {"FAKE_RENDER": "cpu"}

    def test_apply_refuses_undeclared_backend(self):
        with pytest.raises(ValueError, match="does not support render: cpu"):
            apply_render_mode(_GpuOnlyBenchmark, "cpu", "fake")

    def test_check_names_every_offending_benchmark(self):
        config = {
            "benchmarks": [
                {"benchmark": MUJOCO_CPU_BENCHMARKS[0]},
                {"benchmark": GPU_ONLY_BENCHMARK, "name": "offender-a"},
                {"benchmark": GPU_ONLY_BENCHMARK, "name": "offender-b"},
            ]
        }
        with pytest.raises(ValueError) as excinfo:
            cli._check_render_support(config, "cpu")
        assert "offender-a" in str(excinfo.value)
        assert "offender-b" in str(excinfo.value)

    def test_check_passes_for_migrated_benchmarks(self):
        cli._check_render_support({"benchmarks": [{"benchmark": p} for p in MUJOCO_CPU_BENCHMARKS]}, "cpu")

    def test_unresolvable_import_is_deferred_to_the_container(self):
        """Adapter deps often only exist in the benchmark image, so a host-side
        import failure must not abort the run."""
        cli._check_render_support({"benchmarks": [{"benchmark": "no.such.module:Nope"}]}, "cpu")


# ---------------------------------------------------------------------------
# Migrated adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("import_path", MUJOCO_CPU_BENCHMARKS, ids=lambda p: p.rsplit(":", 1)[-1])
def test_mujoco_adapters_switch_to_software_rendering(import_path: str, preserve_env: None):
    """Declaring cpu without implementing the hook would raise NotImplementedError here."""
    benchmark_cls = resolve_import_string(import_path)

    applied = benchmark_cls.configure_render("cpu")

    assert applied == mujoco_cpu_env()
    assert os.environ["MUJOCO_GL"] == "osmesa"


@pytest.mark.parametrize("import_path", MUJOCO_CPU_BENCHMARKS, ids=lambda p: p.rsplit(":", 1)[-1])
def test_gpu_mode_leaves_the_image_defaults_in_place(import_path: str, preserve_env: None):
    assert resolve_import_string(import_path).configure_render("gpu") == {}


def test_cpu_choice_survives_a_later_env_default(preserve_env: None):
    """The RoboCasa adapters apply their egl default inside ``_make_env`` — i.e. after
    ``configure_render`` has run — so it has to stay a setdefault, not an assignment."""
    resolve_import_string(MUJOCO_CPU_BENCHMARKS[0]).configure_render("cpu")

    os.environ.setdefault("MUJOCO_GL", "egl")  # what _make_env does per episode

    assert os.environ["MUJOCO_GL"] == "osmesa"


# ---------------------------------------------------------------------------
# Docker flags
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-process renderer state vs per-entry hook (RoboMME)
# ---------------------------------------------------------------------------


@pytest.fixture
def robomme(monkeypatch: pytest.MonkeyPatch):
    """RoboMME with its process-wide renderer state reset and lavapipe engagement stubbed.

    The real ``_engage_lavapipe`` imports ``sapien.render``, which the dev env lacks.
    """
    cls = resolve_import_string("vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark")
    monkeypatch.setattr(cls, "_rendering_configured", False)
    # raising=False so these assertions, not an AttributeError, are what fails on a regression.
    monkeypatch.setattr(cls, "_render_env", {}, raising=False)

    calls: list[int] = []

    def _fake_engage() -> dict[str, str]:
        calls.append(1)
        return {"VK_ICD_FILENAMES": "/opt/lavapipe/lvp_icd.json"}

    monkeypatch.setattr(cls, "_engage_lavapipe", staticmethod(_fake_engage))
    return cls, calls


def test_robomme_cpu_survives_multiple_benchmark_entries(robomme):
    """configs/benchmarks/robomme/eval.yaml has 4 entries in one process. Re-engaging
    lavapipe would hit the 'sapien.render already imported' guard and abort after the first."""
    cls, calls = robomme

    applied = [cls.configure_render("cpu") for _ in range(4)]

    assert len(calls) == 1, "renderer must be bound once per process, not once per entry"
    assert all(env == {"VK_ICD_FILENAMES": "/opt/lavapipe/lvp_icd.json"} for env in applied)


def test_robomme_gpu_fallback_reports_lavapipe_for_every_entry(robomme, monkeypatch):
    """With ROBOMME_USE_LAVAPIPE=1 the process is software-rendered; entries 2..N must not
    report an empty env, or their aggregates claim a GPU render that never happened."""
    cls, calls = robomme
    monkeypatch.setenv("ROBOMME_USE_LAVAPIPE", "1")

    applied = [cls.configure_render("gpu") for _ in range(3)]

    assert len(calls) == 1
    assert all(env == {"VK_ICD_FILENAMES": "/opt/lavapipe/lvp_icd.json"} for env in applied)


def test_robomme_native_gpu_path_reports_no_env(robomme, monkeypatch):
    cls, calls = robomme
    monkeypatch.delenv("ROBOMME_USE_LAVAPIPE", raising=False)

    assert [cls.configure_render("gpu") for _ in range(2)] == [{}, {}]
    assert not calls


def test_robomme_refuses_cpu_after_the_native_path_is_bound(robomme, monkeypatch):
    """Defensive: the ICD binds at first sapien import, so this cannot be rescued."""
    cls, _ = robomme
    monkeypatch.delenv("ROBOMME_USE_LAVAPIPE", raising=False)
    cls.configure_render("gpu")

    with pytest.raises(RuntimeError, match="native GPU path"):
        cls.configure_render("cpu")


# ---------------------------------------------------------------------------
# Smoke-test render precedence
# ---------------------------------------------------------------------------


class TestSmokeRenderPrecedence:
    def test_config_level_cpu_detaches_the_gpu_without_a_cli_flag(self):
        """A GPU would still be attached otherwise, breaking on CPU-only hosts."""
        assert smoke._resolve_smoke_render({"render": "cpu"}, None, None, None) == ("cpu", "none")

    def test_cli_overrides_config(self):
        assert smoke._resolve_smoke_render({"render": "gpu"}, "cpu", "3", "0,1") == ("cpu", "none")
        assert smoke._resolve_smoke_render({"render": "cpu"}, "gpu", None, "0,1") == ("gpu", "0,1")

    def test_gpu_mode_keeps_the_per_worker_device_assignment(self):
        assert smoke._resolve_smoke_render({}, None, "3", "0,1") == ("gpu", "3")

    def test_gpu_mode_falls_back_to_the_config_device_spec(self):
        assert smoke._resolve_smoke_render({}, None, None, "0,1") == ("gpu", "0,1")


def test_render_provenance_reaches_the_merge_aggregate(tmp_path):
    """The orchestrator records requested + applied env; merge must surface it,
    otherwise a run's renderer is unrecoverable from its results."""
    from vla_eval.recording import RecordingStore, db_path_for_eval
    from vla_eval.results.merge import merge_db

    provenance = {"requested": "cpu", "applied_env": mujoco_cpu_env()}
    db = db_path_for_eval(tmp_path, "ev")
    store = RecordingStore(db)
    store.upsert_eval_metadata("ev", "demo", {"benchmark": "demo", "render": provenance})
    store.close()

    assert merge_db(db, tmp_path)[0]["render"] == provenance


class TestNoGpuDockerFlags:
    def test_emits_no_gpu_flag_and_hides_devices(self):
        # Independent of the host GPU runtime: "none" short-circuits detection.
        assert gpu_docker_flag("none") == ["-e", "NVIDIA_VISIBLE_DEVICES=void"]

    @pytest.mark.parametrize("spec", ["none", "NONE", " none "])
    def test_spec_is_case_and_whitespace_insensitive(self, spec: str):
        assert "--gpus" not in gpu_docker_flag(spec)

    @pytest.mark.parametrize("shard_id", [0, 1, 5])
    def test_every_shard_runs_gpu_free(self, shard_id: int):
        flags = shard_docker_flags(shard_id, 8, cpus="0-15", gpus="none")

        assert "--gpus" not in flags
        assert flags[:2] == ["-e", "NVIDIA_VISIBLE_DEVICES=void"]
        # CPU partitioning and thread caps still apply.
        assert "--cpuset-cpus" in flags
        assert "OMP_NUM_THREADS=1" in flags
