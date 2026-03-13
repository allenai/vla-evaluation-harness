"""Tests for protocol message packing and unpacking."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
from vla_eval.protocol.numpy_codec import decode_ndarray


def test_pack_unpack_simple():
    msg = Message(type=MessageType.OBSERVATION, payload={"key": "value"}, seq=1)
    data = pack_message(msg)
    restored = unpack_message(data)
    assert restored.type == MessageType.OBSERVATION
    assert restored.payload["key"] == "value"
    assert restored.seq == 1


def test_pack_unpack_numpy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    msg = Message(type=MessageType.ACTION, payload={"actions": arr}, seq=5)
    data = pack_message(msg)
    restored = unpack_message(data)
    np.testing.assert_array_equal(restored.payload["actions"], arr)
    assert restored.payload["actions"].dtype == np.float32


def test_numpy_codec_rejects_object_dtype():
    bad = {"__ndarray__": True, "data": b"\x00", "dtype": "O", "shape": [1]}
    with pytest.raises(ValueError, match="Disallowed numpy dtype"):
        decode_ndarray(bad)


def test_numpy_2d_roundtrip():
    arr = np.random.rand(16, 7).astype(np.float32)
    msg = Message(type=MessageType.ACTION, payload={"chunk": arr}, seq=0)
    restored = unpack_message(pack_message(msg))
    np.testing.assert_array_almost_equal(restored.payload["chunk"], arr)
