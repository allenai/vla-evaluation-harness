"""Image codec for efficient image transport over WebSocket.

Supports three encoding modes:
- ``raw``: numpy array bytes (lossless, ~196KB per 256×256×3 uint8).
- ``jpeg``: JPEG compression (Q95, lossy, roughly ~10× smaller). Only supports
  RGB (3-channel) images; raises ``ValueError`` for other modes.
- ``png``: PNG compression (default, lossless, roughly ~2× smaller).

Usage with the numpy codec:
    The image codec hooks into ``encode_ndarray`` / ``decode_ndarray`` by
    detecting HWC uint8 arrays (ndim==3, dtype==uint8, channels in {1,3,4}).
    Non-image arrays pass through unchanged.
"""

from __future__ import annotations

import io
from typing import Any, Literal

import numpy as np

_IMAGE_KEY = "__image__"

ImageFormat = Literal["raw", "jpeg", "png"]


def _is_image_array(arr: np.ndarray) -> bool:
    """Heuristic: HWC uint8 array with 1, 3, or 4 channels."""
    return arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[2] in (1, 3, 4)


def encode_image(img: np.ndarray, fmt: ImageFormat = "raw") -> dict[str, Any]:
    """Encode a numpy image array into a transport-friendly dict.

    Returns a dict with ``__image__`` marker for the decoder to recognize.
    """
    if fmt == "raw":
        return {
            _IMAGE_KEY: True,
            "format": "raw",
            "data": img.tobytes(),
            "dtype": img.dtype.str,
            "shape": list(img.shape),
        }

    from PIL import Image

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()

    if fmt == "jpeg":
        if pil_img.mode != "RGB":
            raise ValueError(
                f"JPEG encoding requires RGB images, got mode={pil_img.mode!r}. "
                f"Use fmt='png' or fmt='raw' for non-RGB images."
            )
        pil_img.save(buf, format="JPEG", quality=95)
    elif fmt == "png":
        pil_img.save(buf, format="PNG")
    else:
        raise ValueError(f"Unknown image format: {fmt!r}")

    return {
        _IMAGE_KEY: True,
        "format": fmt,
        "data": buf.getvalue(),
        "shape": list(img.shape),
    }


def decode_image(obj: dict[str, Any]) -> np.ndarray:
    """Decode a transport dict back into a numpy image array."""
    fmt = obj["format"]

    if fmt == "raw":
        dtype = np.dtype(obj["dtype"])
        return np.frombuffer(obj["data"], dtype=dtype).reshape(obj["shape"]).copy()

    from PIL import Image

    buf = io.BytesIO(obj["data"])
    pil_img = Image.open(buf)
    arr = np.array(pil_img)

    return arr


def is_encoded_image(obj: Any) -> bool:
    """Check if a dict is an encoded image from this codec."""
    return isinstance(obj, dict) and _IMAGE_KEY in obj
