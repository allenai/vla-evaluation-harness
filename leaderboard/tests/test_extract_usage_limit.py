"""Unit tests for extract's usage-limit handling.

A usage-limit rejection used to surface as ``LLMError("error: success")``
with no log written (the log write sat after the raise), and then trigger a
per-paper retry storm against a window that cannot recover mid-run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import extract  # noqa: E402


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def _call(stdout: str, log_path: Path, returncode: int = 0) -> int:
    proc = subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")
    with patch.object(extract.subprocess, "run", return_value=proc):
        return extract._call_claude_cli("s", "u", [], model="m", timeout=5, log_path=log_path)


def test_rejected_rate_limit_raises_usage_limit_error(tmp_path):
    stdout = _stream(
        {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 1786398000}},
        {"type": "result", "subtype": "success", "is_error": True},
    )
    log = tmp_path / "batch.log"
    with pytest.raises(extract.UsageLimitError, match="1786398000"):
        _call(stdout, log)
    assert log.read_text() == stdout


def test_error_event_writes_log_before_raising(tmp_path):
    stdout = _stream({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 529})
    log = tmp_path / "batch.log"
    with pytest.raises(extract.LLMError, match="529"):
        _call(stdout, log)
    assert log.read_text() == stdout


def test_successful_stream_unaffected(tmp_path):
    stdout = _stream(
        {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}},
        {"type": "result", "subtype": "success", "is_error": False},
    )
    assert _call(stdout, tmp_path / "batch.log") == 0


def test_usage_limit_skips_batch_and_per_paper_fallback(tmp_path, monkeypatch):
    aids = ["2601.00001", "2601.00002"]
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path / "papers")
    monkeypatch.setattr(extract, "EXTRACTIONS_RAW_DIR", tmp_path / "raw")
    for aid in aids:
        (tmp_path / "papers" / aid).mkdir(parents=True)
        (tmp_path / "papers" / aid / "paper.md").write_text("x")

    calls = []

    def fake_retry(*args, **kwargs):
        calls.append(args)
        raise extract.UsageLimitError("usage limit reached")

    monkeypatch.setattr(extract, "_call_claude_cli_with_retry", fake_retry)
    results = extract.extract_batch(aids, "rules", "m", resume=False)
    assert results == {aid: None for aid in aids}
    assert len(calls) == 1
