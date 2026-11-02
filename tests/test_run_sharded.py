"""The launcher passes the same recording location to run and export."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("override, export_status", [(False, 0), (True, 0), (True, 7)])
def test_launcher_resolves_export_db(tmp_path, override, export_status):
    base = tmp_path / "base.yaml"
    base.write_text("output_dir: inherited results\n")
    config = tmp_path / "eval.yaml"
    config.write_text("extends: base.yaml\n")
    executable = tmp_path / "vla-eval"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "name = args[args.index('--shard-id') + 1] if args[0] == 'run' else 'export'\n"
        "pathlib.Path(name + '.json').write_text(json.dumps(args))\n"
        f"sys.exit({export_status} if args[0] == 'export' else 0)\n"
    )
    executable.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/run_sharded.sh"
    command = ["bash", str(script), "-c", str(config), "-n", "2", "-e", "abc"]
    if override:
        command.extend(["-o", "override results"])
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}"},
        capture_output=True,
        timeout=30,
    )
    assert (result.returncode != 0) == (export_status != 0)
    output = "override results" if override else "inherited results"
    assert json.loads((tmp_path / "export.json").read_text()) == ["export", f"{output}/recording-abc.sqlite"]
    for shard in range(2):
        args = json.loads((tmp_path / f"{shard}.json").read_text())
        assert args[args.index("--output-dir") + 1] == output
        assert args[args.index("--eval-id") + 1] == "abc"


@pytest.mark.parametrize(
    "codes, attempts, status",
    [([139, 3, 0], 3, 0), ([1], 1, 1), ([137, 137, 137, 137], 3, 137)],  # crash and env unusable rerun; errors don't
)
def test_launcher_reruns_a_shard_that_exits_3_or_dies_from_a_signal(tmp_path, codes, attempts, status):
    config = tmp_path / "eval.yaml"
    config.write_text("output_dir: out\n")
    executable = tmp_path / "vla-eval"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'export' or args[args.index('--shard-id') + 1] != '0':\n"
        "    sys.exit(0)\n"
        "log = pathlib.Path('attempts')\n"
        "n = len(log.read_text()) if log.exists() else 0\n"
        "log.write_text('x' * (n + 1))\n"
        f"sys.exit({codes!r}[min(n, {len(codes) - 1})])\n"
    )
    executable.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/run_sharded.sh"
    result = subprocess.run(
        ["bash", str(script), "-c", str(config), "-n", "2", "-e", "abc"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "VLA_EVAL_SHARD_RESTARTS": "2",
        },
        capture_output=True,
        timeout=60,
    )
    assert len((tmp_path / "attempts").read_text()) == attempts
    assert (result.returncode != 0) == (status != 0), result.stderr.decode()
