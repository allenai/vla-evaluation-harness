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
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    from omegaconf import OmegaConf

    base_path = str(Path(path).resolve().parent / extends)
    base = load_config(base_path)
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(raw))
    # OmegaConf.to_container returns Union[dict, list, None, str]; a
    # merge of two DictConfigs always yields a dict. Assert narrows
    # the type for the checker and catches genuinely unexpected shape
    # at runtime (not just when the caller indexes the result).
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"expected dict from OmegaConf.to_container, got {type(container).__name__}")
    # OmegaConf's return type is dict[Unknown, Unknown]; merging two
    # DictConfigs gives us string keys in practice. Cast so the public
    # signature's dict[str, Any] holds.
    return cast(dict[str, Any], container)
