"""Run provenance: image references, redaction, script pins, host probe, HELLO and the exported block."""

from __future__ import annotations

import json
import logging
import subprocess
import textwrap
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import websockets

from vla_eval import provenance as prov
from vla_eval.connection import Connection
from vla_eval.orchestrator import Orchestrator
from vla_eval.protocol.messages import Message, MessageType, make_hello_payload, pack_message, unpack_message
from vla_eval.recording import RecordingStore
from vla_eval.results.export import export_db, export_eval

from tests.conftest import EchoModelServer, StubBenchmark, running

DIGEST = "sha256:" + "ab" * 32


@pytest.fixture(autouse=True)
def _no_gpu_probe(monkeypatch):
    prov.host_info.cache_clear()
    monkeypatch.setattr(prov.shutil, "which", lambda name: None)
    yield
    prov.host_info.cache_clear()


@pytest.mark.parametrize(
    ("ref", "parts", "mutable"),
    [
        ("ghcr.io/org/img:latest", ("ghcr.io/org/img", "latest", None), True),
        ("ghcr.io/org/img", ("ghcr.io/org/img", None, None), True),
        ("localhost:5000/img", ("localhost:5000/img", None, None), True),
        ("localhost:5000/img:0.4.0", ("localhost:5000/img", "0.4.0", None), False),
        (f"ghcr.io/org/img@{DIGEST}", ("ghcr.io/org/img", None, DIGEST), False),
        (f"ghcr.io/org/img:latest@{DIGEST}", ("ghcr.io/org/img", "latest", DIGEST), False),
    ],
)
def test_split_image_ref(ref, parts, mutable) -> None:
    assert prov.split_image_ref(ref) == parts
    assert prov.is_mutable_ref(ref) is mutable


def test_mutable_tag_warns_with_the_resolved_pin(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="vla_eval.provenance"):
        prov.warn_if_mutable("ghcr.io/org/img:latest", {"digest": DIGEST})
        prov.warn_if_mutable("ghcr.io/org/img", {})
        prov.warn_if_mutable(f"ghcr.io/org/img@{DIGEST}", {"digest": DIGEST})
        prov.warn_if_mutable("ghcr.io/org/img:0.4.0", {"digest": DIGEST})
    assert len(caplog.records) == 2
    assert f"docker.image: ghcr.io/org/img@{DIGEST}" in caplog.records[0].getMessage()
    assert "<repository>@sha256:<digest>" in caplog.records[1].getMessage()


def test_image_env_round_trips_and_absent_is_null(monkeypatch) -> None:
    identity = {"image": "reg/img:tag", "digest": DIGEST, "id": "sha256:cd", "runtime": "docker", "export": None}
    env = prov.image_env(identity)
    assert env == {
        "VLA_EVAL_IMAGE": "reg/img:tag",
        "VLA_EVAL_IMAGE_DIGEST": DIGEST,
        "VLA_EVAL_IMAGE_ID": "sha256:cd",
        "VLA_EVAL_IMAGE_RUNTIME": "docker",
    }
    assert prov.image_from_env(env) == {k: v for k, v in identity.items() if v}
    assert "VLA_EVAL_IMAGE_DEV" not in prov.image_env({**identity, "dev": False})
    dev = prov.image_env({**identity, "dev": True})
    assert dev["VLA_EVAL_IMAGE_DEV"] == "1" and prov.image_from_env(dev)["dev"] is True
    for var in prov.IMAGE_ENV.values():
        monkeypatch.delenv(var, raising=False)
    block = prov.benchmark_provenance()  # --no-docker: nothing guessed
    assert block["image"] is None and block["digest"] is None and "id" not in block
    assert set(block["host"]) == {"platform", "python"}


class _Mode(str, Enum):
    FAST = "fast"


def test_redaction_is_by_word_and_strips_url_credentials() -> None:
    args = {
        "hf_token": "hf_abc",
        "apiKey": "k",
        "OPENAI_API_KEY": "k",
        "key": "k",
        "access_key": "k",
        "db_password": "p",
        "client_secret": "s",
        "nested": {"auth": {"user": "u"}, "items": [{"token": "t"}]},
        "use_auth_token": True,  # booleans and None hold no secret
        "token": None,
        "tokenizer_name": "t5",
        "max_new_tokens": 8,
        "unnorm_key": "bridge_orig",
        "image_keys": ["agentview"],
        "keyframe": 3,
        "checkpoint": "https://user:pw@hf.co/m",
        "ckpt_path": Path("/ckpt"),
        "mode": _Mode.FAST,
    }
    out = prov.redact(args)
    for k in ("hf_token", "apiKey", "OPENAI_API_KEY", "key", "access_key", "db_password", "client_secret"):
        assert out[k] == prov.REDACTED, k
    assert out["nested"] == {"auth": prov.REDACTED, "items": [{"token": prov.REDACTED}]}
    assert out["use_auth_token"] is True and out["token"] is None
    assert out["tokenizer_name"] == "t5" and out["max_new_tokens"] == 8
    assert out["unnorm_key"] == "bridge_orig" and out["image_keys"] == ["agentview"] and out["keyframe"] == 3
    assert out["checkpoint"] == "https://***@hf.co/m"
    assert out["ckpt_path"] == "/ckpt" and out["mode"] == "fast"
    json.dumps(out)


@pytest.mark.parametrize(
    "name",
    [
        *("wandb_key", "openai_key", "hf_key", "ssh_key", "client_key", "HFToken", "OpenAIKey", "hftoken"),
        *("api_keys", "apiKeys", "hf_tokens", "authorization", "Authorization", "bearer", "pwd", "passphrase"),
        *("db_pass", "sentry_dsn", "x-api-key", "AWS_SECRET_ACCESS_KEY", "privateKey", "HF_TOKEN", "keys"),
    ],
)
def test_secret_names(name) -> None:
    assert prov.is_secret_key(name)


@pytest.mark.parametrize(
    "name",
    [
        *("max_tokens", "max_new_tokens", "num_tokens", "added_tokens", "pad_token", "tokenizer_name"),
        *("unnorm_key", "unnorm_keys", "image_key", "image_keys", "wrist_image_key", "state_key", "camera_keys"),
        *("video_keys", "action_keys", "modality_keys", "policy_key", "language_key", "keyframe", "checkpoint"),
        *("passthrough", "bypass", "compass"),
    ],
)
def test_harmless_names_stay_visible(name) -> None:
    assert not prov.is_secret_key(name)


@pytest.mark.parametrize(
    ("value", "scrubbed"),
    [
        ("Bearer sk-live-123", prov.REDACTED),
        ("basic dXNlcjpwYXNz", prov.REDACTED),
        ("curl -H 'Authorization: Bearer sk-live-123' x", "curl -H 'Authorization: *** x"),
        ("https://user:p@ss@host/x", "https://***@host/x"),  # urllib: up to the authority's last @
        ("https://user:pa/ss@host/x", "https://***@host/x"),
        ("https://tok@host/x", "https://***@host/x"),
        ("git+https://user:tok@github.com/org/repo@v1.0", "git+https://***@github.com/org/repo@v1.0"),
        ("https://host/x?token=sk-1&api_key=abc&page=2", "https://host/x?token=***&api_key=***&page=2"),
        ("https://host/x?max_tokens=8", "https://host/x?max_tokens=8"),
        ("user:sekret@host:5432/db", "***@host:5432/db"),
        ("postgres dsn=user:sekret@db.internal", "postgres dsn=***@db.internal"),
        # not credentials: kept as they are
        ("https://hf.co/org/model@main", "https://hf.co/org/model@main"),
        ("https://github.com:443/org/pkg@v1", "https://github.com:443/org/pkg@v1"),
        ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),
        (f"ubuntu:22.04@{DIGEST}", f"ubuntu:22.04@{DIGEST}"),
        ("Do a basic pick task", "Do a basic pick task"),
        ("basic setup", "basic setup"),
        ("Bearer of bad news", "Bearer of bad news"),
    ],
)
def test_credentials_are_scrubbed_from_values(value, scrubbed) -> None:
    assert prov.redact({"v": value}) == {"v": scrubbed}


def test_auth_header_and_script_sources() -> None:
    out = prov.redact(
        {
            "headers": {"Authorization": "Bearer sk-live-123", "X-Trace": "Bearer sk-live-456"},
            "script": {"sources": {"google-auth": {"git": "https://u:tok@github.com/x/y", "rev": "abc"}}},
        }
    )
    assert out["headers"] == {"Authorization": prov.REDACTED, "X-Trace": prov.REDACTED}
    # sources are keyed by package name: credentials are stripped, names kept
    assert out["script"]["sources"] == {"google-auth": {"git": "https://***@github.com/x/y", "rev": "abc"}}
    assert "sk-live" not in json.dumps(out) and "tok@" not in json.dumps(out)


# Indented and dedented: a `# /// script` block at column 0 would make ty (and uv) treat this file as a script.
SCRIPT = textwrap.dedent(
    """\
    # /// script
    # requires-python = "~=3.11"
    # dependencies = [
    #     "vla-eval",
    #     "cogact",
    #     "torch>=2.2",
    # ]
    #
    # [tool.uv.sources]
    # vla-eval = { path = "../../..", editable = true }
    # cogact = { git = "https://github.com/microsoft/CogACT.git", rev = "b174a1b" }
    #
    # [tool.uv]
    # exclude-newer = "2026-11-02T00:20:00Z"
    # index-url = "https://user:secret@pypi.example/simple"
    # ///
    print("hi")
    """
)


def test_script_metadata_reads_pep723_pins(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    script.write_text(SCRIPT)
    meta = prov.script_metadata(script)
    assert meta["path"] == script.name and len(meta["sha256"]) == 64  # outside the checkout: no home path
    assert meta["requires_python"] == "~=3.11"
    assert meta["dependencies"] == ["vla-eval", "cogact", "torch>=2.2"]
    assert meta["sources"]["cogact"] == {"git": "https://github.com/microsoft/CogACT.git", "rev": "b174a1b"}
    assert meta["exclude_newer"] == "2026-11-02T00:20:00Z"
    assert "secret" not in json.dumps(meta)  # only exclude-newer is taken from [tool.uv]
    plain = tmp_path / "plain.py"
    plain.write_text("x = 1\n")
    assert set(prov.script_metadata(plain)) == {"path", "sha256"}


def test_default_provenance_has_class_script_redacted_args_and_host() -> None:
    server = EchoModelServer()
    assert server.get_provenance()["args"] is None  # not launched by run_server
    server._launch_args = {"checkpoint": "org/model", "hf_token": "hf_abc"}
    out = prov.safe_provenance(server.get_provenance)
    assert out["class"] == "tests.conftest.EchoModelServer"
    assert out["script"]["path"].endswith("conftest.py")
    assert out["args"] == {"checkpoint": "org/model", "hf_token": prov.REDACTED}
    assert out["host"]["python"]


def test_safe_provenance_never_raises() -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("no checkpoint")

    assert prov.safe_provenance(boom) == {"error": "RuntimeError: no checkpoint"}
    assert "error" in prov.safe_provenance(lambda: ["not", "a", "dict"])  # ty: ignore[invalid-argument-type]
    assert "error" in prov.safe_provenance(lambda: {"blob": "x" * prov.MAX_BYTES})
    assert prov.safe_provenance(lambda: {"secret": "s", "p": Path("/x")}) == {"secret": prov.REDACTED, "p": "/x"}


class _FakeSmi:
    """``nvidia-smi`` as Popen sees it; ``stuck`` never exits (uninterruptible sleep), so it must not be waited on."""

    def __init__(self, out: str = "", *, stuck: bool = False) -> None:
        self.out, self.stuck, self.returncode, self.killed, self.stdout = out, stuck, 0, False, None

    def __call__(self, cmd, **kw) -> _FakeSmi:
        return self

    def communicate(self, timeout=None) -> tuple[str, None]:
        assert timeout == prov._NVIDIA_SMI_TIMEOUT_S
        if self.stuck:
            raise subprocess.TimeoutExpired("nvidia-smi", prov._NVIDIA_SMI_TIMEOUT_S)
        return self.out, None

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        raise AssertionError("a stuck nvidia-smi must not be waited on")


def test_host_info_reports_gpus_and_survives_nvidia_smi_failure(monkeypatch) -> None:
    monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    out = "NVIDIA H100 80GB HBM3, 570.86\nNVIDIA H100 80GB HBM3, 570.86\nNVIDIA L4, 570.86\n"
    monkeypatch.setattr(prov.subprocess, "Popen", _FakeSmi(out))
    assert prov.host_info()["gpus"] == [
        {"name": "NVIDIA H100 80GB HBM3", "driver": "570.86", "count": 2},
        {"name": "NVIDIA L4", "driver": "570.86", "count": 1},
    ]
    prov.host_info.cache_clear()
    stuck = _FakeSmi(stuck=True)
    monkeypatch.setattr(prov.subprocess, "Popen", stuck)
    info = prov.host_info()
    assert info["python"] and "TimeoutExpired" in info["gpus"]["error"] and stuck.killed


def test_shards_on_different_images_flag_divergence(tmp_path: Path, caplog) -> None:
    def meta(digest: str, python: str) -> dict[str, Any]:
        bench = {"image": "reg/img:latest", "digest": digest, "host": {"python": python}}
        return {"benchmark": "b", "provenance": {"benchmark": bench}}

    store = RecordingStore(tmp_path / "r.sqlite")
    store.upsert_eval_metadata("e-b", "b", meta(DIGEST, "3.11.9"))
    store.upsert_eval_metadata("e-b", "b", meta(DIGEST, "3.11.7"))  # another node's host: expected
    assert "divergent" not in _stored(store)
    with caplog.at_level(logging.WARNING, logger="vla_eval.recording"):
        store.upsert_eval_metadata("e-b", "b", meta("sha256:" + "cd" * 32, "3.11.9"))  # pulled later
        store.upsert_eval_metadata("e-b", "b", meta(DIGEST, "3.11.9"))
    block = _stored(store)
    store.close()
    assert block["divergent"] is True and block["digest"] == DIGEST  # the first shard's identity is kept
    assert len([r for r in caplog.records if "provenance.benchmark.divergent" in r.getMessage()]) == 1


def _stored(store: RecordingStore) -> dict[str, Any]:
    return _metadata(store)["provenance"]["benchmark"]


def _metadata(store: RecordingStore) -> dict[str, Any]:
    row = store._conn.execute("SELECT metadata FROM eval_metadata").fetchone()
    return json.loads(row[0])


def _cfg_id(n: int) -> str:
    return "sha256:" + f"{n:02d}" * 32


@pytest.mark.parametrize(
    ("first", "later", "divergent"),
    [
        # one digest, two Docker image stores (classic Id = config digest, containerd Id = manifest digest)
        ({"digest": DIGEST, "id": _cfg_id(11)}, {"digest": DIGEST, "id": DIGEST}, False),
        # the same digest pulled from the public mirror: the reference is not compared
        ({"image": "ghcr.io/a/img:1", "digest": DIGEST}, {"image": "ghcr.io/mirror/img:1", "digest": DIGEST}, False),
        # an export made before pulls were recorded has no digest: unknown, not different
        ({"digest": None, "id": _cfg_id(22), "runtime": "charliecloud"}, {"digest": DIGEST, "id": _cfg_id(33)}, False),
        # no digests (local builds): ids compare within one runtime, not across
        ({"digest": None, "id": _cfg_id(11)}, {"digest": None, "id": _cfg_id(22)}, True),
        ({"digest": None, "id": _cfg_id(11)}, {"digest": None, "id": _cfg_id(22), "runtime": "charliecloud"}, False),
        ({"digest": None, "id": None}, {"digest": None, "id": _cfg_id(22)}, False),
    ],
)
def test_image_divergence_rules(tmp_path: Path, first, later, divergent) -> None:
    def meta(identity: dict[str, Any]) -> dict[str, Any]:
        return {"provenance": {"benchmark": {"image": "reg/img:1", "runtime": "docker", **identity}}}

    store = RecordingStore(tmp_path / "r.sqlite")
    store.upsert_eval_metadata("e-b", "b", meta(first))
    store.upsert_eval_metadata("e-b", "b", meta(later))
    assert _stored(store).get("divergent", False) is divergent
    store.close()


def test_model_server_divergence_is_flagged_on_stable_fields(tmp_path: Path, caplog) -> None:
    def meta(**over: Any) -> dict[str, Any]:
        server = {
            "class": "__main__.CogACT",
            "script": {"path": "cogact.py", "sha256": "aa"},
            "args": {"checkpoint": "org/a", "hf_token": prov.REDACTED},
            "host": {"python": "3.11.9", "gpus": [{"name": "H100"}]},
            **over,
        }
        return {"server_info": {"model_server": "CogACT", "provenance": server}}

    store = RecordingStore(tmp_path / "r.sqlite")
    store.upsert_eval_metadata("e-b", "b", meta())
    store.upsert_eval_metadata("e-b", "b", meta(host={"python": "3.11.7"}))  # another node: expected
    store.upsert_eval_metadata("e-b", "b", meta(script={"error": "OSError: gone"}))  # unknown, not different
    store.upsert_eval_metadata("e-b", "b", {"server_info": {"provenance": {"error": "timed out"}}})
    store.upsert_eval_metadata("e-b", "b", {"server_info": {"model_server": "CogACT"}})  # an old server
    assert "divergent" not in _metadata(store)["server_info"]["provenance"]
    with caplog.at_level(logging.WARNING, logger="vla_eval.recording"):
        store.upsert_eval_metadata("e-b", "b", meta(args={"checkpoint": "org/b", "hf_token": prov.REDACTED}))
    assert "server_info.provenance.divergent" in caplog.text
    stored = _metadata(store)["server_info"]["provenance"]
    assert stored["divergent"] is True and stored["args"]["checkpoint"] == "org/a"  # the first shard's is kept
    store.close()
    store = RecordingStore(tmp_path / "s.sqlite")
    store.upsert_eval_metadata("e-b", "b", meta())
    store.upsert_eval_metadata("e-b", "b", meta(script={"path": "cogact.py", "sha256": "bb"}))
    assert _metadata(store)["server_info"]["provenance"]["divergent"] is True
    (aggregate,) = export_db(tmp_path / "s.sqlite", tmp_path)
    assert aggregate["provenance"]["model_server"]["divergent"] is True
    store.close()


class _BrokenProvenance(EchoModelServer):
    def get_provenance(self) -> dict[str, Any]:
        raise RuntimeError("probe failed")


@pytest.mark.anyio
@pytest.mark.parametrize("broken", [False, True])
async def test_hello_carries_provenance(free_port, broken) -> None:
    server = _BrokenProvenance() if broken else EchoModelServer()
    async with running(server, free_port) as url:
        for _ in range(2):  # computed once, sent on every connection
            async with websockets.connect(url) as ws:
                await ws.send(pack_message(Message(MessageType.HELLO, make_hello_payload(), 1)))
                payload = unpack_message(await ws.recv(decode=False)).payload
                assert payload["model_server"] == type(server).__name__
                if broken:
                    assert payload["provenance"] == {"error": "RuntimeError: probe failed"}
                else:
                    assert payload["provenance"]["class"] == "tests.conftest.EchoModelServer"
    assert server._provenance == payload["provenance"]


class _SlowProvenance(EchoModelServer):
    def get_provenance(self) -> dict[str, Any]:
        time.sleep(1.0)  # e.g. resolving a checkpoint revision over the network
        return {**super().get_provenance(), "checkpoint_rev": "abc"}


class _StuckProvenance(EchoModelServer):
    def get_provenance(self) -> dict[str, Any]:
        threading.Event().wait()  # never returns
        return {}


@pytest.mark.anyio
async def test_slow_provenance_is_computed_before_connections_not_in_hello(free_port) -> None:
    async with running(_SlowProvenance(), free_port) as url:  # serving: provenance is already cached
        conn = Connection(url, timeout=0.5)  # shorter than get_provenance takes
        try:
            await conn.connect()
            assert conn.server_info["provenance"]["checkpoint_rev"] == "abc"
        finally:
            await conn.close()


@pytest.mark.anyio
async def test_stuck_provenance_is_capped(free_port, monkeypatch) -> None:
    monkeypatch.setattr(prov, "TIMEOUT_S", 0.2)
    async with running(_StuckProvenance(), free_port) as url:
        conn = Connection(url, timeout=5.0)
        try:
            await conn.connect()
            assert conn.server_info["provenance"] == {"error": "get_provenance did not return within 0.2 s"}
        finally:
            await conn.close()


@pytest.mark.anyio
async def test_exported_aggregate_groups_benchmark_and_model_server_provenance(echo_server, tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_EVAL_IMAGE", "reg/img:latest")
    monkeypatch.setenv("VLA_EVAL_IMAGE_DIGEST", DIGEST)
    monkeypatch.setenv("VLA_EVAL_IMAGE_RUNTIME", "docker")
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "prov",
                "episodes_per_task": 1,
                "max_steps": 5,
                "params": {"done_at_step": 1, "num_tasks": 1},
            }
        ],
    }
    with patch("vla_eval.orchestrator.resolve_import_string", return_value=StubBenchmark):
        await Orchestrator(config, eval_id="ev-prov").run()
    (aggregate,) = export_eval(tmp_path, "ev-prov")
    block = aggregate["provenance"]
    assert block["benchmark"]["image"] == "reg/img:latest"
    assert block["benchmark"]["digest"] == DIGEST and block["benchmark"]["runtime"] == "docker"
    assert block["benchmark"]["host"]["python"]
    assert block["model_server"]["class"] == "tests.conftest.EchoModelServer"
    assert block["model_server"] == aggregate["server_info"]["provenance"]  # additive: server_info unchanged
    on_disk = json.loads((tmp_path / "prov_aggregate.json").read_text())
    assert on_disk["provenance"]["benchmark"]["digest"] == DIGEST
