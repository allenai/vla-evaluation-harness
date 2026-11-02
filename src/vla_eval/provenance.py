"""Run provenance: which benchmark image, model-server code and host produced a result.

The host resolves the benchmark image's identity once it is local (``cli/_docker.py``, ``cli/_charliecloud.py``)
and passes it into the container as ``VLA_EVAL_IMAGE*`` env; the orchestrator records it with the host block in
each benchmark's eval metadata.  The model server reports its own (``ModelServer.get_provenance``) in HELLO.  The
exported aggregate groups both under ``provenance``.  Nothing here raises: a failed probe records ``{"error": ...}``.
Never recorded: environment values, ``docker.env``, and constructor args whose names look secret.
"""

from __future__ import annotations

import enum
import functools
import hashlib
import inspect
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

IMAGE_ENV: Final = {
    "image": "VLA_EVAL_IMAGE",
    "digest": "VLA_EVAL_IMAGE_DIGEST",
    "id": "VLA_EVAL_IMAGE_ID",
    "runtime": "VLA_EVAL_IMAGE_RUNTIME",
    "export": "VLA_EVAL_IMAGE_EXPORT",
    "export_mtime": "VLA_EVAL_IMAGE_EXPORT_MTIME",
    "dev": "VLA_EVAL_IMAGE_DEV",
}
"""Image identity field -> the env var carrying it into the container."""

REDACTED: Final = "<redacted>"
MAX_BYTES: Final = 32 * 1024
"""HELLO's ``provenance`` beyond this (as JSON) is replaced by an error: it rides every connection."""
TIMEOUT_S: Final = 60.0
"""``get_provenance`` still running after this at server start is recorded as an error instead."""

_NVIDIA_SMI_TIMEOUT_S: Final = 5.0


def _guard(fn: Callable[[], Any]) -> Any:
    """``fn()``, or ``{"error": ...}`` if it raised."""
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


@functools.cache
def host_info() -> dict[str, Any]:
    """Platform, Python and GPUs (``nvidia-smi``, when present) of this process's host; cached, never raises."""

    def probe() -> dict[str, Any]:
        info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
        gpus = _guard(_nvidia_gpus)
        if gpus is not None:
            info["gpus"] = gpus
        return info

    return _guard(probe)


def _nvidia_gpus() -> list[dict[str, Any]] | None:
    """``[{"name", "driver", "count"}]`` from ``nvidia-smi``; ``None`` without it.  A stuck ``nvidia-smi`` is killed
    and abandoned after ``_NVIDIA_SMI_TIMEOUT_S``, not waited on (one in uninterruptible sleep never exits)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    cmd = [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        out, _ = proc.communicate(timeout=_NVIDIA_SMI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()
        raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    gpus: list[dict[str, Any]] = []
    for line in filter(str.strip, out.splitlines()):
        name, _, driver = (part.strip() for part in line.rpartition(","))
        for g in gpus:
            if g["name"] == name and g["driver"] == driver:
                g["count"] += 1
                break
        else:
            gpus.append({"name": name, "driver": driver, "count": 1})
    return gpus


# ---------------------------------------------------------------------------
# Benchmark image
# ---------------------------------------------------------------------------


def split_image_ref(ref: str) -> tuple[str, str | None, str | None]:
    """``(repository, tag, digest)`` of an image reference; a ``:port`` in the registry is not a tag."""
    name, _, digest = ref.partition("@")
    colon = name.rfind(":")
    if colon > name.rfind("/"):
        return name[:colon], name[colon + 1 :], digest or None
    return name, None, digest or None


def is_mutable_ref(ref: str) -> bool:
    """``:latest`` or no tag, and no ``@sha256:`` pin."""
    _, tag, digest = split_image_ref(ref)
    return digest is None and tag in (None, "latest")


def warn_if_mutable(image: str, identity: Mapping[str, Any]) -> None:
    """One warning for a mutable reference, with the digest pin this run resolved (when known)."""
    if not is_mutable_ref(image):
        return
    digest = identity.get("digest")
    pin = f"{split_image_ref(image)[0]}@{digest}" if digest else "<repository>@sha256:<digest>"
    logger.warning(
        "docker.image %s is a mutable tag; reruns may get a different image. Pin it to reproduce this run: "
        "docker.image: %s",
        image,
        pin,
    )


def image_env(identity: Mapping[str, Any]) -> dict[str, str]:
    """The ``VLA_EVAL_IMAGE*`` env carrying ``identity`` into the container (unset or false fields omitted)."""
    return {
        var: "1" if identity[k] is True else str(identity[k])
        for k, var in IMAGE_ENV.items()
        if identity.get(k) not in (None, "", False)
    }


def image_from_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The image identity the host passed in (empty when run outside a vla-eval-launched container)."""
    env = os.environ if environ is None else environ
    out: dict[str, Any] = {k: env[var] for k, var in IMAGE_ENV.items() if env.get(var)}
    if "dev" in out:
        out["dev"] = out["dev"] == "1"
    return out


def benchmark_provenance() -> dict[str, Any]:
    """The benchmark side's block: image identity (``null`` when unknown, e.g. ``--no-docker``), ``dev: true`` when
    ``--dev`` mounted host source over the image's (its digest then does not describe the code that ran), and
    host.  May run ``nvidia-smi``: call it off the event loop."""
    return {"image": None, "digest": None, **image_from_env(), "host": host_info()}


# ---------------------------------------------------------------------------
# Model server
# ---------------------------------------------------------------------------

_PEP723 = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")


def script_metadata(path: str | Path) -> dict[str, Any]:
    """``path`` and ``sha256`` of a script, plus its PEP 723 ``script`` block's pins when it has one:
    ``requires_python``, ``dependencies``, ``sources`` (``[tool.uv.sources]``, e.g. git revs) and
    ``exclude_newer``."""
    p = Path(path)
    data = p.read_bytes()
    out: dict[str, Any] = {"path": _portable_path(p), "sha256": hashlib.sha256(data).hexdigest()}
    block = next((m for m in _PEP723.finditer(data.decode("utf-8", "replace")) if m["type"] == "script"), None)
    if block is None:
        return out
    toml = "".join(line[2:] if line.startswith("# ") else line[1:] for line in block["content"].splitlines(True))
    meta = tomllib.loads(toml)
    uv = meta.get("tool", {}).get("uv", {})
    pins = {
        "requires_python": meta.get("requires-python"),
        "dependencies": meta.get("dependencies"),
        "sources": uv.get("sources"),
        "exclude_newer": uv.get("exclude-newer"),
    }
    return {**out, **{k: v for k, v in pins.items() if v is not None}}


def _portable_path(p: Path) -> str:
    """``p`` relative to the harness checkout (``src/vla_eval/model_servers/x.py``), else its file name: results
    are shared, and an absolute path names the user's home."""
    resolved = p.resolve()
    root = Path(__file__).resolve().parents[2]  # <checkout>/src/vla_eval/provenance.py
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


_SECRET_WORDS: Final = frozenset(
    {
        *("token", "secret", "secrets", "password", "passwords", "passwd", "pass", "pwd", "passphrase"),
        *("apikey", "apikeys", "credential", "credentials", "auth", "authorization", "bearer", "cookie", "dsn"),
    }
)
_HARMLESS_QUALIFIERS: Final = {
    "key": frozenset(
        {"image", "state", "unnorm", "camera", "video", "action", "modality", "policy", "language", "bench", "obs"}
    ),
    "token": frozenset({"max", "new", "num", "added", "special", "pad", "eos", "bos"}),
}
"""Words before a trailing ``key(s)``/``token(s)`` that name data, not a secret: those the shipped model servers and
configs use (``unnorm_key``, ``image_keys``, ``wrist_image_key``, ``max_new_tokens``, ...)."""

_URL_USERINFO = re.compile(
    r"(?<=://)(?:[^\s/?#]*@"  # as urllib splits it: up to the authority's last @ (user:p@ss@host)
    r"|[^\s/?#@:]*:(?!\d*(?:[/?#\s]|$))[^\s@]*@)"  # user:pa/ss@host: an authority whose ':x' is no port
)
_DSN_USERINFO = re.compile(r"(?:^|(?<=[\s=]))[\w.~%+-]+:[^\s@/:]+@(?!sha\d+:)(?=[\w.-]+(?::\d+)?(?:[/?\s]|$))")
"""Scheme-less ``user:pass@host[:port][/db]``; not ``git@host:org/repo`` nor ``name:tag@sha256:...``."""
_AUTH_VALUE = re.compile(r"(?i)^\s*(?:bearer\s+[\w.~+/-]{8,}=*|basic\s+[A-Za-z0-9+/]{8,}={0,2})\s*$")
"""A whole value that is an auth header's (``Bearer <token>``, ``Basic <base64>``), not prose like ``basic setup``."""
_AUTH_HEADER = re.compile(r"(?i)\b((?:proxy-)?authorization)\s*[:=]\s*\S+(?:\s+\S+)?")
_QUERY_PARAM = re.compile(r"(?<=[?&])([^=&#\s]+)=([^&#\s]*)")
_PACKAGE_KEYED: Final = frozenset({("script", "sources")})
"""Mappings keyed by package name, not by what the value holds: ``google-auth`` is no secret."""


def _words(name: object) -> list[str]:
    """``name`` as lowercase words, split at non-alphanumerics, camelCase and acronyms (``HFToken``: hf, token)."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(name))
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower())


def is_secret_key(name: object) -> bool:
    """Whether an arg named ``name`` may hold a secret: a word in ``_SECRET_WORDS`` (``token``, ``password``,
    ``pass``, ``pwd``, ``auth``, ``authorization``, ``bearer``, ``dsn``, ...), or a trailing ``key``/``keys``/
    ``token``/``tokens`` (``hf_key``, ``api_keys``, ``HFToken``, ``key``) unless the word before it is in
    ``_HARMLESS_QUALIFIERS`` (``unnorm_key``, ``image_keys``, ``max_new_tokens``).  Words, not substrings:
    ``tokenizer_name`` and ``keyframe`` do not match."""
    words = _words(name)
    if not words:
        return False
    last = words[-1].removesuffix("s")
    if last in ("key", "token"):
        return len(words) == 1 or words[-2] not in _HARMLESS_QUALIFIERS[last]
    return any(w in _SECRET_WORDS for w in words) or last.endswith("token")  # hftoken, accesstoken


def _scrub(text: str) -> str:
    """``text`` without credentials: an auth header value (``Bearer ...``), URL or DSN userinfo, and secret-named
    query parameters (``?token=...``)."""
    if _AUTH_VALUE.match(text):
        return REDACTED
    text = _AUTH_HEADER.sub(r"\1: ***", text)
    text = _DSN_USERINFO.sub("***@", _URL_USERINFO.sub("***@", text))
    return _QUERY_PARAM.sub(lambda m: f"{m[1]}=***" if is_secret_key(m[1]) else m[0], text)


def redact(value: Any) -> Any:
    """``value`` made JSON-safe with secret-named entries (``is_secret_key``) replaced by ``REDACTED`` and
    credentials scrubbed from strings (``_scrub``).  ``None`` and booleans are kept: they hold no secret.  The
    keys of ``script.sources`` are package names, so only their values are scrubbed."""
    return _redact(value, ())


def _redact(value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        by_name = path not in _PACKAGE_KEYED
        out: dict[str, Any] = {}
        for k, v in value.items():
            secret = by_name and is_secret_key(k) and v is not None and not isinstance(v, bool)
            out[str(k)] = REDACTED if secret else _redact(v, (*path, str(k)))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact(v, path) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return _redact(value.value, path)
    return _scrub(str(value))


def model_server_provenance(server: Any) -> dict[str, Any]:
    """Default ``ModelServer.get_provenance``: class, script (path, sha256, PEP 723 pins), constructor args as
    parsed by ``run_server`` (``None`` otherwise; secrets redacted) and host.  Each part fails on its own."""
    cls = type(server)
    args = getattr(server, "_launch_args", None)
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "script": _guard(lambda: script_metadata(inspect.getfile(cls))),
        "args": _guard(lambda: redact(args)) if args is not None else None,
        "host": host_info(),
    }


def safe_provenance(get: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    """``get()`` as a small JSON-safe dict for HELLO; any failure, or more than ``MAX_BYTES``, becomes
    ``{"error": ...}``."""

    def build() -> dict[str, Any]:
        prov = get()
        if not isinstance(prov, Mapping):
            raise TypeError(f"get_provenance returned {type(prov).__name__}, not a dict")
        body = json.loads(json.dumps(redact(prov), default=str))
        size = len(json.dumps(body))
        if size > MAX_BYTES:
            return {"error": f"provenance is {size} bytes (> {MAX_BYTES}); keep get_provenance small"}
        return body

    return _guard(build)


def bounded_provenance(get: Callable[[], Mapping[str, Any]], timeout: float | None = None) -> dict[str, Any]:
    """``safe_provenance(get)``, or ``{"error": ...}`` once it has run ``timeout`` (default ``TIMEOUT_S``) seconds: a
    stuck ``get`` is left on its daemon thread, not waited on.  Blocks: call it off the event loop."""
    timeout = TIMEOUT_S if timeout is None else timeout
    box: list[dict[str, Any]] = []
    thread = threading.Thread(target=lambda: box.append(safe_provenance(get)), name="vla-eval-provenance", daemon=True)
    thread.start()
    thread.join(timeout)
    if box:
        return box[0]
    if thread.is_alive():
        return {"error": f"get_provenance did not return within {timeout:g} s"}
    return {"error": "get_provenance did not return"}  # a BaseException ended the thread
