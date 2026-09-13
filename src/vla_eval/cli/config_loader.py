"""YAML config loader with ``extends`` inheritance support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file, resolving ``extends`` chains and
    ``${oc.env:VAR,default}`` interpolations.

    If the YAML contains ``extends: relative/path.yaml``, the base config is
    loaded first (recursively) and the child is merged on top via OmegaConf.
    """
    from omegaconf import OmegaConf

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is not None:
        base_path = str(Path(path).resolve().parent / extends)
        base = load_config(base_path)
        merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(raw))
    else:
        merged = OmegaConf.create(raw)

    # ``resolve=True`` expands OmegaConf interpolations (``${oc.env:VAR}``,
    # ``${oc.env:VAR,default}``) so configs can pick up host-side state
    # like ``$VLA_EVAL_DATA_DIR`` without requiring a pre-pass.  This
    # runs uniformly for both ``extends``-based and standalone configs.
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"expected dict from OmegaConf.to_container, got {type(container).__name__}")
    config = cast("dict[str, Any]", container)
    _resolve_build_paths(config, Path(path).resolve().parent)
    return config


def _resolve_build_paths(config: dict[str, Any], base: Path) -> None:
    """``docker.build.context`` is relative to the YAML that declares it; ``dockerfile`` stays relative to it."""
    docker = config.get("docker")
    if not isinstance(docker, dict) or not docker.get("build"):
        return
    build = docker["build"]
    if isinstance(build, str):
        build = {"context": build}
    if not isinstance(build, dict) or not build.get("context"):
        return
    docker["build"] = {**build, "context": str((base / build["context"]).resolve())}
