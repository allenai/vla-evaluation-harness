"""Message types and serialization for the WebSocket protocol."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack

from vla_eval.protocol.numpy_codec import decode_ndarray, encode_ndarray


class MessageType(str, enum.Enum):
    HELLO = "hello"
    OBSERVATION = "observation"
    ACTION = "action"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    ERROR = "error"


PROTOCOL_VERSION = 1


def make_hello_payload(**extra: Any) -> dict[str, Any]:
    """Build the common HELLO payload fields. Callers add role-specific keys via *extra*."""
    from vla_eval import __version__

    return {"harness_version": __version__, "protocol_version": PROTOCOL_VERSION, **extra}


@dataclass
class Message:
    type: MessageType
    payload: dict[str, Any]
    seq: int = 0
    timestamp: float = field(default_factory=time.time)


def pack_message(msg: Message) -> bytes:
    """Serialize a Message to msgpack bytes."""
    raw = {
        "type": msg.type.value,
        "payload": msg.payload,
        "seq": msg.seq,
        "timestamp": msg.timestamp,
    }
    return msgpack.packb(raw, default=encode_ndarray, use_bin_type=True)


def unpack_message(data: bytes) -> Message:
    """Deserialize msgpack bytes to a Message.

    Raises:
        ValueError: If the message is malformed or missing required fields.
    """
    try:
        raw = msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)
    except Exception as exc:
        raise ValueError(f"Failed to decode msgpack data ({len(data)} bytes): {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Expected msgpack dict, got {type(raw).__name__}")

    _REQUIRED = ("type", "payload", "seq", "timestamp")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"Message missing required fields: {missing}. Got keys: {list(raw.keys())}")

    try:
        msg_type = MessageType(raw["type"])
    except ValueError:
        raise ValueError(f"Unknown message type: {raw['type']!r}. Valid types: {[t.value for t in MessageType]}")

    return Message(
        type=msg_type,
        payload=raw["payload"],
        seq=raw["seq"],
        timestamp=raw["timestamp"],
    )
