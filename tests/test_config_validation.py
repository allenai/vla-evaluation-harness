"""Validate all config files: YAML parses, scripts exist, import strings are well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
MODEL_SERVER_CONFIGS = sorted(CONFIGS_DIR.glob("model_servers/*.yaml"))
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
    """Model server YAML parses and references an existing script."""
    data = yaml.safe_load(config_path.read_text())

    assert "script" in data, f"Missing 'script' key in {config_path.name}"
    script = REPO_ROOT / data["script"]
    assert script.exists(), f"Script not found: {data['script']}"


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------


def _has_benchmarks_key(data: dict) -> bool:
    return isinstance(data.get("benchmarks"), list)


BENCHMARK_CONFIGS_WITH_BENCHMARKS = [
    p for p in BENCHMARK_CONFIGS if _has_benchmarks_key(yaml.safe_load(p.read_text()))
]


@pytest.mark.parametrize(
    "config_path",
    BENCHMARK_CONFIGS_WITH_BENCHMARKS,
    ids=[p.name for p in BENCHMARK_CONFIGS_WITH_BENCHMARKS],
)
def test_benchmark_config_import_strings(config_path: Path) -> None:
    """Benchmark configs have well-formed 'module:Class' import strings."""
    data = yaml.safe_load(config_path.read_text())

    for bench in data["benchmarks"]:
        import_path = bench.get("benchmark", "")
        assert ":" in import_path, f"Import string must be 'module:Class', got {import_path!r} in {config_path.name}"
        module, _, cls_name = import_path.partition(":")
        assert module, f"Empty module in {import_path!r}"
        assert cls_name, f"Empty class name in {import_path!r}"
