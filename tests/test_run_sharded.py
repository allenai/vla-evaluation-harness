"""The launcher passes the same recording location to run and export."""

import json
import os
import sqlite3
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


# A shard: registers in the recording, claims and finishes its queue items through the real RecordingStore, and
# exits with the next planned code per attempt: "unregistered" exits 1 first; "<rc>" exits rc after one claim;
# "<rc>@end" exits rc after finishing everything (a crash at interpreter exit). Shard 1 starts once shard 0 claimed.
_FAKE_SHARD = """
import json, os, pathlib, sys, time
from vla_eval.recording import RecordingStore
args = sys.argv[1:]
if args[0] == "export":
    sys.exit(0)
shard, n = int(args[args.index("--shard-id") + 1]), int(args[args.index("--num-shards") + 1])
attempts = pathlib.Path(f"attempts-{shard}")
attempt = len(attempts.read_text()) if attempts.exists() else 0
attempts.write_text("x" * (attempt + 1))
plan = json.loads(os.environ["FAKE_PLAN"]).get(str(shard), [])
step = plan[attempt] if attempt < len(plan) else "0"
started = time.monotonic()
while shard and not pathlib.Path("ready").exists() and time.monotonic() - started < 30:
    time.sleep(0.05)
if step == "unregistered":
    pathlib.Path("ready").touch()
    sys.exit(1)
out, eval_id = args[args.index("--output-dir") + 1], args[args.index("--eval-id") + 1]
store, entry = RecordingStore(f"{out}/recording-{eval_id}.sqlite"), f"{eval_id}-bench"
store.set_shard_complete(entry, shard, n, complete=False)
store.seed_queue(entry, [0] * 6)
rc, _, at_end = step.partition("@")
while (item := store.claim(entry, shard, 0, n)) is not None:
    pathlib.Path("ready").touch()
    if int(rc) and not at_end:
        os._exit(int(rc))
    store.finish(entry, item)
store.set_shard_complete(entry, shard, n, complete=True)
os._exit(int(rc))
"""


@pytest.mark.parametrize(
    "plan, max_restarts, attempts, status",
    [
        ({"0": ["139", "3", "0"]}, "10", 3, 0),  # crash, then env unusable, then done
        ({"0": ["unregistered"]}, "10", 1, 1),  # a startup/config error is not retried
        ({"0": ["137"] * 5}, "2", 3, 1),  # bounded
        ({"0": ["139@end"]}, "10", 1, 0),  # every item recorded: the exit code alone is no reason to rerun
    ],
)
def test_launcher_restarts_a_failed_shard_while_the_recording_has_work_left(
    tmp_path, plan, max_restarts, attempts, status
):
    config = tmp_path / "eval.yaml"
    config.write_text("output_dir: out\nbenchmarks:\n  - benchmark: bench\n")
    executable = tmp_path / "vla-eval"
    executable.write_text(f"#!{sys.executable}\n{_FAKE_SHARD}")
    executable.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/run_sharded.sh"
    result = subprocess.run(
        ["bash", str(script), "-c", str(config), "-n", "2", "-e", "abc"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "FAKE_PLAN": json.dumps(plan),
            "VLA_EVAL_SHARD_RESTARTS": max_restarts,
            "VLA_EVAL_RESTART_DELAY": "0",
        },
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == status, result.stderr.decode()
    assert len((tmp_path / "attempts-0").read_text()) == attempts
    assert len((tmp_path / "attempts-1").read_text()) == 1
    if status == 0:
        db = sqlite3.connect(tmp_path / "out/recording-abc.sqlite")
        assert db.execute("SELECT COUNT(*), SUM(done) FROM work_queue").fetchone() == (6, 6)
