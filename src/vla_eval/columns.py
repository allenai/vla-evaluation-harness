"""Column batches: a (nested) dict of columns, each an array with a leading row axis or a list with an item per
row; row ``i`` of every column belongs to one episode."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from vla_eval.types import Batch


def num_rows(batch: Batch) -> int | None:
    """The batch's row count (``None`` for an empty batch); raises if its columns disagree."""
    lengths = {len(v) for v in _leaves(batch)}
    if len(lengths) > 1:
        raise ValueError(f"columns of different lengths {sorted(lengths)}")
    return lengths.pop() if lengths else None


def take(batch: Batch, rows: Sequence[int] | np.ndarray) -> Batch:
    """These ``rows`` of ``batch``, in this order."""
    idx = np.asarray(rows, np.int64)
    return _map(batch, lambda v: v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx.tolist()])


def concat(parts: Sequence[Batch]) -> Batch:
    """The rows of every part, part after part; raises if their keys differ."""
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else {}
    if any(p.keys() != parts[0].keys() for p in parts):
        raise ValueError(f"batches with different keys: {sorted({k for p in parts for k in p})}")
    out: Batch = {}
    for key, first in parts[0].items():
        values = [p[key] for p in parts]
        if isinstance(first, dict):
            out[key] = concat(values)
        elif all(isinstance(v, np.ndarray) for v in values) and len({v.shape[1:] for v in values}) == 1:
            out[key] = np.concatenate(values)
        else:  # rows of different shapes: a list column
            out[key] = [x for v in values for x in v]
    return out


def split(batch: Batch, n: int) -> list[dict[str, Any]]:
    """Row ``i``'s values, as a (nested) dict per row."""
    return [_map(batch, lambda v, i=i: v[i]) for i in range(n)]


def stack(items: Sequence[dict[str, Any]]) -> Batch:
    """The column batch of per-row dicts with the same keys: arrays of one shape and dtype stack, the rest are lists."""
    if not items:
        return {}
    out: Batch = {}
    for key, first in items[0].items():
        values = [item[key] for item in items]
        if isinstance(first, dict):
            out[key] = stack(values)
        elif (
            all(isinstance(v, (np.ndarray, np.generic)) for v in values)
            and len({(np.shape(v), np.asarray(v).dtype) for v in values}) == 1
        ):
            out[key] = np.stack(values)
        else:
            out[key] = values
    return out


def _leaves(batch: Batch) -> list[Any]:
    out: list[Any] = []
    for v in batch.values():
        out.extend(_leaves(v) if isinstance(v, dict) else [v])
    return out


def _map(batch: Batch, fn: Any) -> Batch:
    return {k: _map(v, fn) if isinstance(v, dict) else fn(v) for k, v in batch.items()}
