"""LIBERO image preprocessing utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert floating point image to uint8."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method=Image.Resampling.BILINEAR) -> np.ndarray:
    """Resize image with padding to target size, preserving aspect ratio."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize images with padding. Handles batched or single images."""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def preprocess_libero_image(img: np.ndarray, resolution: int = 256) -> np.ndarray:
    """Flip a LIBERO camera image (both axes) and ensure uint8.

    LIBERO's MuJoCo cameras produce images that are flipped on both axes
    relative to the convention expected by VLA models.
    """
    img = np.ascontiguousarray(img[::-1, ::-1])
    return convert_to_uint8(img)
