"""Image resizing in the LeRobot bridge: letterbox to the declared shape, passthrough for policies
that letterbox themselves.

Pure numpy: the bilinear scaler is swapped for a nearest-neighbour one (the dev env has no torch);
the real torch scaler is exercised too whenever torch is importable.
"""

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


def _frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Grey 4:3 frame with a white 200x200 square: the square must stay square after resizing."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    img[100:300, 220:420] = 255
    return img


def _server(monkeypatch, policy_type: str = "act", declared=(3, 256, 256), scaler=_nearest, **config):
    """A ``LeRobotModelServer`` without ``__init__`` (no policy download): just the attributes
    ``_resize_to_declared`` reads."""
    server = lr.LeRobotModelServer.__new__(lr.LeRobotModelServer)
    server.policy_type = policy_type
    features = {KEY: SimpleNamespace(shape=declared)} if declared else {}
    server._policy = SimpleNamespace(config=SimpleNamespace(input_features=features, **config))
    if scaler is not None:
        monkeypatch.setattr(server, "_scale_bilinear", scaler)
    return server


def _square_bbox(out: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(out[..., 0] > 128)
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


@pytest.mark.parametrize(
    ("src", "dst", "expected"),
    [
        ((480, 640), (256, 256), (192, 256, 32, 0)),  # 4:3 into a square: bars top and bottom
        ((640, 480), (256, 256), (256, 192, 0, 32)),  # 3:4: bars left and right
        ((480, 640), (224, 224), (168, 224, 28, 0)),
        ((480, 640), (240, 320), (240, 320, 0, 0)),  # same aspect: fills the target
        ((300, 400), (225, 300), (225, 300, 0, 0)),  # exact match even where float division would truncate
    ],
)
def test_letterbox_geometry(src, dst, expected) -> None:
    assert lr._letterbox_geometry(src, dst) == expected


def test_same_size_passes_through(monkeypatch) -> None:
    server = _server(monkeypatch, scaler=lambda img, hw: pytest.fail("must not scale"))
    img = _frame(256, 256)
    assert server._resize_to_declared(img, KEY) is img


def test_undeclared_shape_passes_through(monkeypatch) -> None:
    server = _server(monkeypatch, declared=None, scaler=lambda img, hw: pytest.fail("must not scale"))
    img = _frame()
    assert server._resize_to_declared(img, KEY) is img


def test_4_3_frame_is_letterboxed_into_square(monkeypatch) -> None:
    server = _server(monkeypatch)
    out = server._resize_to_declared(_frame(), KEY)

    assert out.shape == (256, 256, 3) and out.dtype == np.uint8
    assert not out[:32].any() and not out[224:].any()  # black bars, not stretched content
    assert (out[32:224] >= 40).all()  # the whole frame width is inside the content band
    np.testing.assert_array_equal(out[32:224], _nearest(_frame(), (192, 256)))
    assert _square_bbox(out) == (80, 80)  # 200 px at a uniform 1/2.5 scale, on both axes


def test_same_aspect_scales_without_padding(monkeypatch) -> None:
    calls = []

    def scaler(img, hw):
        calls.append(hw)
        return _nearest(img, hw)

    server = _server(monkeypatch, declared=(3, 240, 320), scaler=scaler)
    out = server._resize_to_declared(_frame(), KEY)

    assert calls == [(240, 320)]
    np.testing.assert_array_equal(out, _nearest(_frame(), (240, 320)))


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
    server = _server(monkeypatch, policy_type, scaler=lambda img, hw: pytest.fail("must not scale"), **config)
    img = _frame()
    assert server._resize_to_declared(img, KEY) is img


@pytest.mark.parametrize(
    ("policy_type", "config"),
    [
        ("act", {}),
        ("xvla", {"resize_imgs_with_padding": None}),
        ("fastwam", {"image_resolution": (224, 224)}),  # only the pi family letterboxes on image_resolution
    ],
)
def test_policy_without_its_own_letterbox_is_resized(monkeypatch, policy_type, config) -> None:
    server = _server(monkeypatch, policy_type, **config)
    assert server._resize_to_declared(_frame(), KEY).shape == (256, 256, 3)


def test_torch_scaler_letterboxes(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    server = _server(monkeypatch, scaler=None)
    server._torch = torch
    out = server._resize_to_declared(_frame(), KEY)

    assert out.shape == (256, 256, 3) and out.dtype == np.uint8
    assert not out[:32].any() and not out[224:].any()
    h, w = _square_bbox(out)
    assert abs(h - w) <= 1 and abs(h - 80) <= 1
