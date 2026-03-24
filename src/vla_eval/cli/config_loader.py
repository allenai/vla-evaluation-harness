"""YAML config loader with ``extends`` inheritance support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file, resolving ``extends`` chains.

    If the YAML contains ``extends: relative/path.yaml``, the base config is
    loaded first (recursively) and the child is merged on top via OmegaConf.
    The result is always returned as a plain ``dict[str, Any]``.

    Configs without ``extends`` are loaded identically to ``yaml.safe_load``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    from omegaconf import OmegaConf

    base_path = str(Path(path).resolve().parent / extends)
    base = load_config(base_path)
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(raw))
    result: dict[str, Any] = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]
    return result
