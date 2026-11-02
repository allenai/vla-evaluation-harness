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
}
"""Image identity field -> the env var carrying it into the container."""

REDACTED: Final = "<redacted>"
MAX_BYTES: Final = 32 * 1024
"""HELLO's ``provenance`` beyond this (as JSON) is replaced by an error: it rides every connection."""

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
    """``[{"name", "driver", "count"}]`` from ``nvidia-smi``; ``None`` without it."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    out = subprocess.run(
        [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=_NVIDIA_SMI_TIMEOUT_S,
        check=True,
    ).stdout
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
    """The ``VLA_EVAL_IMAGE*`` env carrying ``identity`` into the container (unset fields omitted)."""
    return {var: str(identity[k]) for k, var in IMAGE_ENV.items() if identity.get(k) not in (None, "")}


def image_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The image identity the host passed in (empty when run outside a vla-eval-launched container)."""
    env = os.environ if environ is None else environ
    return {k: env[var] for k, var in IMAGE_ENV.items() if env.get(var)}


def benchmark_provenance() -> dict[str, Any]:
    """The benchmark side's block: image identity (``null`` when unknown, e.g. ``--no-docker``) and host."""
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
    {"token", "secret", "secrets", "password", "passwd", "apikey", "credential", "credentials", "auth", "cookie"}
)
_SECRET_KEY_QUALIFIERS: Final = frozenset({"api", "access", "private", "secret", "auth", "signing", "encryption"})
_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


def is_secret_key(name: object) -> bool:
    """Whether an arg named ``name`` may hold a secret: a word ``token``/``secret``/``password``/``apikey``/
    ``credential``/``auth``/``cookie``, a name ending ``api_key``/``access_key``/``private_key``/..., or ``key``
    alone.  Words, not substrings: ``hf_token``/``apiKey`` match; ``tokenizer_name``/``max_tokens``/
    ``unnorm_key``/``image_keys`` do not."""
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower())
    if any(w in _SECRET_WORDS for w in words):
        return True
    return words == ["key"] or (len(words) >= 2 and words[-1] == "key" and words[-2] in _SECRET_KEY_QUALIFIERS)


def redact(value: Any) -> Any:
    """``value`` made JSON-safe with secret-named entries (``is_secret_key``) replaced by ``REDACTED`` and URL
    credentials (``scheme://user:pass@``) stripped.  ``None`` and booleans are kept: they hold no secret."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            secret = is_secret_key(k) and v is not None and not isinstance(v, bool)
            out[str(k)] = REDACTED if secret else redact(v)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return redact(value.value)
    return _USERINFO.sub("***@", str(value))


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
