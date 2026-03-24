"""Validate all config files: YAML parses, scripts exist, import strings are well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest

from vla_eval.cli.config_loader import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
MODEL_SERVER_CONFIGS = sorted(p for p in CONFIGS_DIR.glob("model_servers/**/*.yaml") if p.name != "_base.yaml")
BENCHMARK_CONFIGS = sorted(p for p in CONFIGS_DIR.glob("*.yaml") if p.name != "README.md")


# ---------------------------------------------------------------------------
# Model server configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path",
    MODEL_SERVER_CONFIGS,
    ids=[p.name for p in MODEL_SERVER_CONFIGS],
)
def test_model_server_config_valid(config_path: Path) -> None:
    """Model server YAML parses (resolving extends) and references an existing script."""
    data = load_config(str(config_path))

    assert "script" in data, f"Missing 'script' key in {config_path.name}"
    script = REPO_ROOT / data["script"]
    assert script.exists(), f"Script not found: {data['script']}"


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------


def _has_benchmarks_key(data: dict) -> bool:
    return isinstance(data.get("benchmarks"), list)


BENCHMARK_CONFIGS_WITH_BENCHMARKS = [p for p in BENCHMARK_CONFIGS if _has_benchmarks_key(load_config(str(p)))]


@pytest.mark.parametrize(
    "config_path",
    BENCHMARK_CONFIGS_WITH_BENCHMARKS,
    ids=[p.name for p in BENCHMARK_CONFIGS_WITH_BENCHMARKS],
)
def test_benchmark_config_import_strings(config_path: Path) -> None:
    """Benchmark configs have well-formed 'module:Class' import strings."""
    data = load_config(str(config_path))

    for bench in data["benchmarks"]:
        import_path = bench.get("benchmark", "")
        assert ":" in import_path, f"Import string must be 'module:Class', got {import_path!r} in {config_path.name}"
        module, _, cls_name = import_path.partition(":")
        assert module, f"Empty module in {import_path!r}"
        assert cls_name, f"Empty class name in {import_path!r}"


# ---------------------------------------------------------------------------
# Config extends resolution
# ---------------------------------------------------------------------------

EXTENDS_CONFIGS = sorted(
    p for p in CONFIGS_DIR.glob("model_servers/**/*.yaml") if p.name != "_base.yaml" and "extends" in p.read_text()
)


@pytest.mark.parametrize(
    "config_path",
    EXTENDS_CONFIGS,
    ids=[f"{p.parent.name}/{p.name}" for p in EXTENDS_CONFIGS],
)
def test_extends_resolution(config_path: Path) -> None:
    """Configs with 'extends' resolve to a complete dict with 'script' and 'args'."""
    data = load_config(str(config_path))
    assert isinstance(data, dict)
    assert "script" in data, f"Resolved config missing 'script': {config_path}"
    assert "args" in data, f"Resolved config missing 'args': {config_path}"
    assert "extends" not in data, f"'extends' key leaked into resolved config: {config_path}"
