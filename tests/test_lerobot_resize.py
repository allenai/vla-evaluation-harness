"""LeRobot bridge image resizing: letterbox geometry and passthrough for self-letterboxing policies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_eval.model_servers import lerobot as lr

KEY = "observation.images.front"


def _nearest(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    rows = (np.arange(h) * img.shape[0]) // h
    cols = (np.arange(w) * img.shape[1]) // w
    return img[rows[:, None], cols[None, :]]


def _server(monkeypatch, policy_type: str = "act", declared=(3, 256, 256), **config):
    """``LeRobotModelServer`` without ``__init__``: only what ``_resize_to_declared`` reads."""
    server = lr.LeRobotModelServer.__new__(lr.LeRobotModelServer)
    server.policy_type = policy_type
    features = {KEY: SimpleNamespace(shape=declared)} if declared else {}
    server._policy = SimpleNamespace(config=SimpleNamespace(input_features=features, **config))
    monkeypatch.setattr(server, "_scale_bilinear", _nearest)
    return server


@pytest.mark.parametrize(
    ("src", "dst", "expected"),
    [
        ((480, 640), (256, 256), (192, 256, 32, 0)),  # 4:3 into a square: bars top and bottom
        ((640, 480), (256, 256), (256, 192, 0, 32)),  # 3:4: bars left and right
        ((480, 640), (240, 320), (240, 320, 0, 0)),  # same aspect: fills the target
        ((300, 400), (225, 300), (225, 300, 0, 0)),  # exact match even where float division would truncate
    ],
)
def test_letterbox_geometry(src, dst, expected) -> None:
    assert lr._letterbox_geometry(src, dst) == expected


def test_4_3_frame_is_letterboxed_into_square(monkeypatch) -> None:
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    out = _server(monkeypatch)._resize_to_declared(img, KEY)

    assert out.shape == (256, 256, 3) and out.dtype == np.uint8
    assert not out[:32].any() and not out[224:].any()  # black bars, not stretched content
    np.testing.assert_array_equal(out[32:224], _nearest(img, (192, 256)))


@pytest.mark.parametrize(
    ("policy_type", "config"),
    [
        ("smolvla", {"resize_imgs_with_padding": (512, 512)}),
        ("xvla", {"resize_imgs_with_padding": (224, 224)}),
        ("pi0", {"image_resolution": (224, 224)}),
        ("pi05", {"image_resolution": (224, 224)}),
        ("pi0_fast", {"image_resolution": (224, 224)}),
    ],
)
def test_policy_with_its_own_letterbox_gets_the_raw_frame(monkeypatch, policy_type, config) -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _server(monkeypatch, policy_type, **config)._resize_to_declared(img, KEY) is img


@pytest.mark.parametrize(
    ("policy_type", "config"),
    [
        ("act", {}),
        ("xvla", {"resize_imgs_with_padding": None}),
        ("fastwam", {"image_resolution": (224, 224)}),  # only the pi family letterboxes on image_resolution
    ],
)
def test_policy_without_its_own_letterbox_is_resized(monkeypatch, policy_type, config) -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _server(monkeypatch, policy_type, **config)._resize_to_declared(img, KEY).shape == (256, 256, 3)
