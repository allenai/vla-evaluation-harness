"""Host-side cache directory resolver and runtime licence helper.

Mirrors HuggingFace's ``HF_HOME`` / ``HF_ASSETS_CACHE`` precedence shape so consumers (benchmarks,
model servers) put state in one canonical place.  See PR #58 for the full layout discussion.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ACCEPTED_LICENSES_ENV = "VLA_EVAL_ACCEPTED_LICENSES"


def home() -> Path:
    """``$VLA_EVAL_HOME > $XDG_CACHE_HOME/vla-eval > ~/.cache/vla-eval``."""
    override = os.environ.get("VLA_EVAL_HOME")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "vla-eval"


def assets_cache(subdir: str | None = None) -> Path:
    """``$VLA_EVAL_ASSETS_CACHE > home()/assets`` (+ optional ``subdir``)."""
    override = os.environ.get("VLA_EVAL_ASSETS_CACHE")
    base = Path(override).expanduser() if override else home() / "assets"
    return base / subdir if subdir else base


def ensure_git_clone(name: str, repo: str, rev: str, *, shallow: bool = False) -> Path:
    """Lazy clone ``repo`` at ``rev`` into ``assets_cache(name)``.  Idempotent."""
    target = assets_cache(name)
    if (target / ".git").exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s @ %s -> %s", repo, rev, target)
    if shallow:
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", rev, repo, str(target)])
    else:
        # Full clone for arbitrary commit SHAs (GitHub rejects shallow-fetch by SHA).
        subprocess.check_call(["git", "clone", repo, str(target)])
        subprocess.check_call(["git", "-C", str(target), "checkout", rev])
    return target


_LICENCE_BANNER = "=" * 70


def ensure_license(license_id: str, *, url: str, description: str) -> None:
    """Ensure the user accepted ``license_id``; raise ``SystemExit`` on rejection.

    Bypass via ``$VLA_EVAL_ACCEPTED_LICENSES`` (comma-separated); else interactive stdin prompt;
    else exits with a hint about ``--accept-license`` / the env var.
    """
    accepted = {item.strip() for item in os.environ.get(ACCEPTED_LICENSES_ENV, "").split(",") if item.strip()}
    if license_id in accepted:
        return

    banner = (
        f"\n{_LICENCE_BANNER}\n"
        f"[vla-eval] Licence required: {description}\n"
        f"  ID:  {license_id}\n"
        f"  URL: {url}\n"
        f"{_LICENCE_BANNER}\n"
    )
    sys.stderr.write(banner)

    if not sys.stdin.isatty():
        sys.stderr.write(
            "Non-interactive context (no TTY).  To proceed, re-run with one of:\n"
            f"  vla-eval run ... --accept-license {license_id}\n"
            f"  {ACCEPTED_LICENSES_ENV}={license_id} vla-eval run ...\n"
        )
        raise SystemExit(1)

    sys.stderr.write("Accept this licence? [y/N] ")
    sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer in ("y", "yes"):
        return
    sys.stderr.write("Licence rejected; aborting.\n")
    raise SystemExit(1)
