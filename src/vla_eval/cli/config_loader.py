"""YAML config loader with ``extends`` inheritance support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file, resolving ``extends`` chains.

    If the YAML contains ``extends: relative/path.yaml``, the base config is
    loaded first (recursively) and the child is merged on top via OmegaConf.
    The result is always returned as a plain ``dict[str, Any]``.

    Configs without ``extends`` are loaded identically to ``yaml.safe_load``.
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
    # OmegaConf's return type is dict[Unknown, Unknown]; YAML mappings
    # are dict[str, Any] in practice. Cast so the public signature
    # holds.
    return cast(dict[str, Any], container)
