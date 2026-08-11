"""Tests for the registry mirror fallback in cli._docker.ensure_image_local."""

from unittest.mock import patch

import pytest

from vla_eval.cli import _docker

IMAGE = "ghcr.io/allenai/vla-evaluation-harness/duobench:0.4.0"
MIRROR = "ghcr.io/worv-ai/vla-evaluation-harness-public/duobench:0.4.0"


def _run(call_results: dict[str, int]) -> list[list[str]]:
    """Run ensure_image_local with canned `docker pull` exit codes; return recorded calls."""
    calls: list[list[str]] = []

    def fake_call(cmd):
        calls.append(cmd)
        if cmd[1] == "pull":
            return call_results[cmd[2]]
        return 0

    with patch.object(_docker, "image_exists_locally", return_value=False):
        with patch.object(_docker.subprocess, "call", side_effect=fake_call):
            _docker.ensure_image_local("docker", IMAGE, auto_yes=True)
    return calls


def test_primary_pull_success_skips_mirror():
    calls = _run({IMAGE: 0})
    assert [c[1] for c in calls] == ["pull"]


def test_fallback_pulls_mirror_and_retags():
    calls = _run({IMAGE: 1, MIRROR: 0})
    assert calls[1] == ["docker", "pull", MIRROR]
    assert calls[2] == ["docker", "tag", MIRROR, IMAGE]


def test_both_registries_failing_exits():
    with pytest.raises(SystemExit):
        _run({IMAGE: 1, MIRROR: 1})


def test_non_allenai_image_has_no_mirror():
    other = "ghcr.io/example/thing:1.0"
    calls: list[list[str]] = []

    def fake_call(cmd):
        calls.append(cmd)
        return 1

    with patch.object(_docker, "image_exists_locally", return_value=False):
        with patch.object(_docker.subprocess, "call", side_effect=fake_call):
            with pytest.raises(SystemExit):
                _docker.ensure_image_local("docker", other, auto_yes=True)
    assert [c[1] for c in calls] == ["pull"]
