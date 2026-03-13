"""Tests for action chunk buffer and ensemble functions."""

from __future__ import annotations

import numpy as np

from vla_eval.model_servers.chunking import ActionChunkBuffer, get_ensemble_fn


def test_chunk_buffer_basic():
    fn = get_ensemble_fn("newest")
    buf = ActionChunkBuffer(chunk_size=4, ensemble_fn=fn)
    chunk = np.arange(28, dtype=np.float32).reshape(4, 7)
    buf.push_chunk(chunk)
    for i in range(4):
        action = buf.pop()
        np.testing.assert_array_equal(action, chunk[i])
    assert buf.empty


def test_chunk_buffer_ema_ensemble():
    fn = get_ensemble_fn("ema", ema_alpha=0.5)
    buf = ActionChunkBuffer(chunk_size=2, ensemble_fn=fn)
    chunk1 = np.array([[1.0, 0.0], [2.0, 0.0]])
    chunk2 = np.array([[3.0, 0.0], [4.0, 0.0]])
    buf.push_chunk(chunk1)
    _ = buf.pop()  # consume first action
    # Now 1 action remains: [2.0, 0.0]
    buf.push_chunk(chunk2)
    # Overlap of 1: ema(old=[2,0], new=[3,0], alpha=0.5) = [2.5, 0]
    action = buf.pop()
    assert action is not None
    np.testing.assert_array_almost_equal(action, [2.5, 0.0])


def test_callable_ensemble():
    def my_fn(old, new):
        return old * 0.3 + new * 0.7

    fn = get_ensemble_fn(my_fn)
    assert fn is my_fn
