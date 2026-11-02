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


DIGEST = "sha256:" + "ab" * 32
PINNED = f"ghcr.io/allenai/vla-evaluation-harness/duobench@{DIGEST}"
PINNED_MIRROR = f"ghcr.io/worv-ai/vla-evaluation-harness-public/duobench@{DIGEST}"


def test_digest_pinned_mirror_fallback_runs_the_mirror_reference():
    """A digest reference cannot be retagged, so the mirror's own reference is what runs."""
    calls: list[list[str]] = []

    def fake_call(cmd):
        calls.append(cmd)
        return 1 if cmd[-1] == PINNED else 0

    with patch.object(_docker, "image_exists_locally", return_value=False):
        with patch.object(_docker.subprocess, "call", side_effect=fake_call):
            ref = _docker.ensure_image_local("docker", PINNED, auto_yes=True)
    assert ref == PINNED_MIRROR
    assert calls == [["docker", "pull", PINNED], ["docker", "pull", PINNED_MIRROR]]  # no `docker tag`


def test_digest_pinned_image_already_pulled_from_the_mirror_is_reused():
    with patch.object(_docker, "image_exists_locally", side_effect=lambda d, ref: ref == PINNED_MIRROR):
        with patch.object(_docker.subprocess, "call", side_effect=AssertionError("no pull")):
            assert _docker.ensure_image_local("docker", PINNED, auto_yes=False) == PINNED_MIRROR
    with patch.object(_docker, "image_exists_locally", return_value=True):
        assert _docker.ensure_image_local("docker", PINNED, auto_yes=False) == PINNED
