"""The extract and refine agents read third-party paper text with permissions bypassed.

They must not be handed shell or network tools: an instruction embedded in a
paper would otherwise run with the operator's privileges.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import extract  # noqa: E402
import refine  # noqa: E402

UNSAFE_TOOLS = {"Bash", "WebFetch", "WebSearch"}


def test_extract_agent_has_no_shell_or_network_tools(tmp_path):
    stdout = json.dumps({"type": "result", "subtype": "success", "is_error": False})
    proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    with patch.object(extract.subprocess, "run", return_value=proc) as run:
        extract._call_claude_cli("s", "u", [], model="m", timeout=5, log_path=tmp_path / "batch.log")
    cmd = run.call_args.args[0]
    assert "bypassPermissions" in cmd
    blocked = set(cmd[cmd.index("--disallowed-tools") + 1].split())
    assert UNSAFE_TOOLS <= blocked
    assert "Agent" in blocked  # sub-agent delegation stays off


def test_refine_agent_has_no_shell_or_network_tools():
    assert UNSAFE_TOOLS <= set(refine.REFINE_DISALLOWED_TOOLS.split())
