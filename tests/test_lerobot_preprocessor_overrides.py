"""``preprocessor_overrides``: per-step LeRobot preprocessor config from YAML, merged over the
bridge's device entry and forwarded to ``make_pre_post_processors``.

The dev env has neither torch nor lerobot; minimal stand-ins in ``sys.modules`` let the real
``LeRobotModelServer.__init__`` run and capture what reaches ``make_pre_post_processors``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from vla_eval.model_servers import lerobot as lr
from vla_eval.model_servers.serve import run_server

TOKENIZER = {"tokenizer_processor": {"tokenizer_name": "/local/tokenizer"}}


def test_merge_without_overrides_is_the_device_entry() -> None:
    assert lr._merge_preprocessor_overrides("cuda:1", None) == {"device_processor": {"device": "cuda:1"}}
    assert lr._merge_preprocessor_overrides("cpu", {}) == {"device_processor": {"device": "cpu"}}


def test_merge_adds_user_steps_next_to_the_device_entry() -> None:
    assert lr._merge_preprocessor_overrides("cuda", TOKENIZER) == {"device_processor": {"device": "cuda"}, **TOKENIZER}


def test_user_device_processor_entry_replaces_the_bridge_one() -> None:
    user = {"device_processor": {"device": "cpu", "float_dtype": "float16"}}
    assert lr._merge_preprocessor_overrides("cuda", user) == user


@pytest.fixture
def fake_lerobot(monkeypatch) -> dict[str, Any]:
    """Stand-ins for ``torch`` and the two ``lerobot`` modules ``__init__`` imports; returns the
    kwargs captured from ``make_pre_post_processors``."""
    captured: dict[str, Any] = {}

    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.front": None}, input_features={}, n_obs_steps=1, n_action_steps=10
        )
    )
    policy.to = lambda device: policy
    policy.eval = lambda: policy

    def make_pre_post_processors(config, source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return (lambda frame: frame), (lambda chunk: chunk)

    fakes = {
        "torch": SimpleNamespace(device=lambda name: name, cuda=SimpleNamespace(is_available=lambda: False)),
        "lerobot": SimpleNamespace(),
        "lerobot.configs": SimpleNamespace(),
        "lerobot.configs.policies": SimpleNamespace(
            PreTrainedConfig=SimpleNamespace(
                from_pretrained=lambda checkpoint: SimpleNamespace(device=None, compile_model=False)
            )
        ),
        "lerobot.policies": SimpleNamespace(
            get_policy_class=lambda policy_type: SimpleNamespace(
                from_pretrained=lambda checkpoint, config=None: policy
            ),
            make_pre_post_processors=make_pre_post_processors,
        ),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured


def test_init_forwards_the_merged_overrides(fake_lerobot) -> None:
    lr.LeRobotModelServer("smolvla", "org/ckpt", device="cpu", preprocessor_overrides=TOKENIZER)

    assert fake_lerobot["source"] == "org/ckpt"
    assert fake_lerobot["preprocessor_overrides"] == {"device_processor": {"device": "cpu"}, **TOKENIZER}


def test_init_without_overrides_sends_only_the_device_entry(fake_lerobot) -> None:
    lr.LeRobotModelServer("smolvla", "org/ckpt", device="cpu")

    assert fake_lerobot["preprocessor_overrides"] == {"device_processor": {"device": "cpu"}}


def test_yaml_config_reaches_make_pre_post_processors(fake_lerobot, monkeypatch, tmp_path) -> None:
    """The documented tokenizer example, parsed by ``run_server`` from a ``vla-eval serve`` yaml."""
    config = tmp_path / "server.yaml"
    args = {"policy_type": "smolvla", "checkpoint": "org/ckpt", "device": "cpu", "preprocessor_overrides": TOKENIZER}
    config.write_text(yaml.safe_dump({"script": "/unused/lerobot.py", "args": args}))
    monkeypatch.setattr("vla_eval.model_servers.serve.serve", lambda server, host, port: sys.exit(0))
    monkeypatch.setattr(sys, "argv", ["serve.py", "--config", str(config)])

    with pytest.raises(SystemExit) as exc:
        run_server(lr.LeRobotModelServer)

    assert exc.value.code == 0
    assert fake_lerobot["preprocessor_overrides"] == {"device_processor": {"device": "cpu"}, **TOKENIZER}
