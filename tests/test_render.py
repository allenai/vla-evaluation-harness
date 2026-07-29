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
    "vla_eval.benchmarks.vlabench.benchmark:VLABenchBenchmark",
    "vla_eval.benchmarks.molmospaces.benchmark:MolmoSpacesBenchmark",
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

    def test_cli_cpu_overrides_a_config_device_spec(self):
        """robomme's configs all pin gpus: all; --render cpu must still attach no device."""
        config: dict[str, Any] = {"docker": {"image": "img", "gpus": "all"}}

        assert cli._resolve_render_mode(config, "cpu") == "cpu"
        assert config["docker"]["gpus"] == "none"

    def test_config_asking_for_cpu_and_a_device_is_rejected(self):
        """No CLI act to arbitrate — the YAML simply contradicts itself."""
        config: dict[str, Any] = {"render": "cpu", "docker": {"image": "img", "gpus": "all"}}

        with pytest.raises(ValueError, match="conflicts with render: cpu"):
            cli._resolve_render_mode(config, None)

    def test_cli_cpu_against_an_explicit_cli_gpus_is_rejected(self):
        """Two flags at the same precedence level; neither outranks the other."""
        config: dict[str, Any] = {"docker": {"image": "img", "gpus": "0,1"}}

        with pytest.raises(ValueError, match="conflicts with render: cpu"):
            cli._resolve_render_mode(config, "cpu", cli_gpus="0,1")

    def test_cpu_tolerates_a_gpus_spec_that_already_means_none(self):
        config: dict[str, Any] = {"render": "cpu", "docker": {"image": "img", "gpus": "none"}}

        assert cli._resolve_render_mode(config, None) == "cpu"
        assert config["docker"]["gpus"] == "none"

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


def test_kinetix_switches_the_jax_device_rather_than_a_gl_backend(preserve_env: None):
    """Kinetix computes frames in JAX, so it is the one adapter whose cpu mode is not GL."""
    benchmark_cls = resolve_import_string("vla_eval.benchmarks.kinetix.benchmark:KinetixBenchmark")

    assert benchmark_cls.configure_render("cpu") == {"JAX_PLATFORMS": "cpu"}
    assert benchmark_cls.configure_render("gpu") == {}


def test_duobench_keeps_egl_but_pins_the_software_device(preserve_env: None):
    """DuoBench's rcs camera stack bootstraps its own EGL context regardless of
    MUJOCO_GL, so OSMesa would leave rcs unbootstrapped — cpu must stay on EGL and
    select the Mesa software device explicitly."""
    benchmark_cls = resolve_import_string("vla_eval.benchmarks.duobench.benchmark:DuoBenchBenchmark")

    applied = benchmark_cls.configure_render("cpu")

    assert applied["MUJOCO_GL"] == "egl", "OSMesa would break rcs's EGL bootstrap"
    assert applied["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert benchmark_cls.configure_render("gpu") == {}


def test_molmospaces_stubs_the_mac_only_cgl_module():
    """molmo_spaces calls mujoco.cgl lock/unlock around every GPU-less render, but that
    module hard-dlopens the macOS OpenGL framework; without the stub, render: cpu
    crashes at the first scene load."""
    import sys

    from vla_eval.benchmarks.molmospaces.benchmark import _stub_out_mujoco_cgl

    before = {k: sys.modules.get(k) for k in ("mujoco.cgl", "mujoco.cgl.cgl")}
    try:
        _stub_out_mujoco_cgl()
        from mujoco.cgl import cgl  # what MjOpenGLRenderer executes per render

        assert cgl.CGLUnlockContext(object()) is None
        assert cgl.CGLLockContext(object()) is None
    finally:
        for k, v in before.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_calvin_swaps_egl_for_the_tiny_renderer(preserve_env: None):
    """PyBullet's EGL plugin aborts the whole process with no GPU, so CALVIN's cpu mode
    routes through an env var that _init_calvin reads to refuse the plugin — the patch
    itself is lazy because calvin_env only exists inside the benchmark image."""
    benchmark_cls = resolve_import_string("vla_eval.benchmarks.calvin.benchmark:CALVINBenchmark")

    assert benchmark_cls.configure_render("cpu") == {"CALVIN_USE_EGL": "0"}
    assert os.environ["CALVIN_USE_EGL"] == "0"
    assert benchmark_cls.configure_render("gpu") == {}


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


def test_robomme_cpu_engages_lavapipe(robomme):
    """cpu must actually bind software Vulkan, not fall through to the native path."""
    cls, calls = robomme

    applied = apply_render_mode(cls, "cpu", "RoboMME")

    assert len(calls) == 1
    assert applied["VK_ICD_FILENAMES"].endswith("lvp_icd.json")


def test_robomme_gpu_fallback_reports_lavapipe_for_every_entry(robomme, monkeypatch):
    """With ROBOMME_USE_LAVAPIPE=1 the process is software-rendered; entries 2..N must not
    report an empty env, or their aggregates claim a GPU render that never happened.
    configs/benchmarks/robomme/eval.yaml has 4 entries sharing one process."""
    cls, calls = robomme
    monkeypatch.setenv("ROBOMME_USE_LAVAPIPE", "1")

    applied = [cls.configure_render("gpu") for _ in range(4)]

    assert len(calls) == 1, "renderer must be bound once per process, not once per entry"
    assert all(env == {"VK_ICD_FILENAMES": "/opt/lavapipe/lvp_icd.json"} for env in applied)


def test_robomme_native_gpu_path_reports_no_env(robomme, monkeypatch):
    cls, calls = robomme
    monkeypatch.delenv("ROBOMME_USE_LAVAPIPE", raising=False)

    assert [cls.configure_render("gpu") for _ in range(2)] == [{}, {}]
    assert not calls


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

    @pytest.mark.parametrize(
        ("config", "cli_render"),
        [({"render": "gpu"}, None), ({"render": "cpu"}, "gpu"), ({}, "gpu")],
        ids=["from-yaml", "cli-overrides-cpu-config", "cli-on-bare-config"],
    )
    def test_rejects_the_same_conflict_cmd_run_rejects(self, config: dict, cli_render: str | None):
        """The smoke config drops its docker section before the inner run, so nothing
        downstream re-checks this: a GPU renderer would start in a device-free container."""
        with pytest.raises(ValueError, match="conflicts with render: gpu"):
            smoke._resolve_smoke_render(config, cli_render, None, "none")

    def test_an_explicit_worker_device_resolves_the_conflict(self):
        assert smoke._resolve_smoke_render({"render": "gpu"}, None, "2", "none") == ("gpu", "2")

    def test_config_asking_for_cpu_and_a_device_is_rejected(self):
        with pytest.raises(ValueError, match="conflicts with render: cpu"):
            smoke._resolve_smoke_render({"render": "cpu"}, None, None, "all")

    def test_cli_cpu_overrides_a_config_device_spec(self):
        assert smoke._resolve_smoke_render({}, "cpu", None, "all") == ("cpu", "none")

    def test_a_harness_assigned_worker_device_never_trips_the_cpu_rule(self):
        """gpu_id comes from --parallel, not the user's config, so it cannot contradict it."""
        assert smoke._resolve_smoke_render({"render": "cpu"}, None, "2", None) == ("cpu", "none")


class TestSmokeErroredEpisodes:
    """A smoke pass must mean episodes ran: error isolation makes exit code 0 and an
    aggregate compatible with every single episode having errored — which is exactly
    how a broken adapter/image pairing reads as healthy."""

    def test_all_errored_episodes_are_reported(self):
        aggregate = {
            "tasks": [
                {
                    "task": "t0",
                    "episodes": [{"failure_reason": "exception", "failure_detail": "Trace\nValueError: boom"}],
                }
            ]
        }
        assert smoke._errored_episodes(aggregate) == ["t0: exception — ValueError: boom"]

    def test_clean_episodes_report_nothing(self):
        aggregate = {"tasks": [{"task": "t0", "episodes": [{"steps": 50, "metrics": {"success": 0.0}}]}]}
        assert smoke._errored_episodes(aggregate) == []


def _write_metadata(tmp_path, *renders: dict[str, Any]):
    """Simulate N shards writing bench metadata under one eval_id; return the merged aggregate."""
    from vla_eval.recording import RecordingStore, db_path_for_eval
    from vla_eval.results.merge import merge_db

    db = db_path_for_eval(tmp_path, "ev")
    for render in renders:
        store = RecordingStore(db)  # each shard is its own process/connection
        store.upsert_eval_metadata("ev", "demo", {"benchmark": "demo", "render": render})
        store.close()
    return merge_db(db, tmp_path)[0]


def test_render_provenance_reaches_the_merge_aggregate(tmp_path):
    """The orchestrator records requested + applied env; merge must surface it,
    otherwise a run's renderer is unrecoverable from its results."""
    provenance = {"requested": "cpu", "applied_env": mujoco_cpu_env()}

    assert _write_metadata(tmp_path, provenance)["render"] == provenance


def test_agreeing_shards_leave_provenance_unflagged(tmp_path):
    provenance = {"requested": "gpu", "applied_env": {}}

    assert _write_metadata(tmp_path, provenance, provenance, provenance)["render"] == provenance


def test_disagreeing_shards_are_flagged_not_silently_resolved(tmp_path):
    """eval_metadata is INSERT OR IGNORE and every shard writes the same eval_id, so a
    RoboMME auto-probe that lands differently per process would otherwise publish one
    shard's renderer as everyone's."""
    native = {"requested": "gpu", "applied_env": {}}
    lavapipe = {"requested": "gpu", "applied_env": {"VK_ICD_FILENAMES": "/opt/lavapipe/lvp_icd.json"}}

    render = _write_metadata(tmp_path, native, lavapipe)["render"]

    assert render["divergent"] is True
    assert render["applied_env"] == native["applied_env"], "first writer's payload is still reported"


def test_shards_matching_the_recorded_provenance_do_not_rewarn(tmp_path, caplog):
    """Once the flag is set the stored render dict carries the extra key, so the
    comparison must ignore it — else every later agreeing shard re-reports divergence."""
    native = {"requested": "gpu", "applied_env": {}}
    lavapipe = {"requested": "gpu", "applied_env": {"VK_ICD_FILENAMES": "/opt/lavapipe/lvp_icd.json"}}

    render = _write_metadata(tmp_path, native, lavapipe, native, native)["render"]

    assert render["divergent"] is True, "later agreeing shards must not clear the flag"
    warnings = [r for r in caplog.records if "divergent" in r.getMessage()]
    assert len(warnings) == 1, "only the actually-disagreeing shard should warn"


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
