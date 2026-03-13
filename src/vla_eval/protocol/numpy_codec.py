"""Numpy codec for msgpack serialization.

Supports an optional ``image_format`` parameter to compress HWC uint8 image
arrays as JPEG or PNG instead of raw bytes.  Non-image arrays always use the
raw numpy encoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.protocol.image_codec import (
    ImageFormat,
    _is_image_array,
    decode_image,
    encode_image,
    is_encoded_image,
)

# Allowed dtype kinds: bool (b), signed int (i), unsigned int (u), float (f)
_ALLOWED_DTYPE_KINDS = frozenset("biuf")
_NDARRAY_KEY = "__ndarray__"

# Module-level default; overridden via ``set_image_format()``.
_image_format: ImageFormat = "png"


def set_image_format(fmt: ImageFormat) -> None:
    """Set the image encoding format for subsequent ``encode_ndarray`` calls."""
    global _image_format
    _image_format = fmt


def get_image_format() -> ImageFormat:
    """Return the current image encoding format."""
    return _image_format


def encode_ndarray(obj: Any) -> Any:
    """msgpack default hook: convert numpy arrays to serializable dicts.

    HWC uint8 image arrays are encoded using the configured image format
    (raw/jpeg/png).  All other arrays use the standard numpy encoding.
    """
    if isinstance(obj, np.ndarray):
        if _image_format != "raw" and _is_image_array(obj):
            return encode_image(obj, _image_format)
        return {
            _NDARRAY_KEY: True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
        }
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def decode_ndarray(obj: Any) -> Any:
    """msgpack object_hook: reconstruct numpy arrays from dicts."""
    if not isinstance(obj, dict):
        return obj
    if is_encoded_image(obj):
        return decode_image(obj)
    if _NDARRAY_KEY not in obj:
        return obj
    dtype = np.dtype(obj["dtype"])
    if dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValueError(f"Disallowed numpy dtype: {dtype}. Only bool/int/uint/float are permitted.")
    return np.frombuffer(obj["data"], dtype=dtype).reshape(obj["shape"]).copy()
