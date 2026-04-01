from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.cowvla import (
    CoWVLAModelServer,
    _PreparedRequest,
    _SessionBuffers,
    _normalize_optional_str,
    _select_image,
    _unnormalize_actions,
)


def test_normalize_optional_str_handles_disabled_values():
    assert _normalize_optional_str(None) is None
    assert _normalize_optional_str("none") is None
    assert _normalize_optional_str(" Null ") is None
    assert _normalize_optional_str("wrist") == "wrist"


def test_select_image_prefers_named_key_then_falls_back_to_order():
    primary = np.ones((2, 2, 3), dtype=np.uint8)
    wrist = np.zeros((2, 2, 3), dtype=np.uint8)
    images = {"agentview": primary, "wrist": wrist}

    np.testing.assert_array_equal(_select_image(images, "wrist"), wrist)
    np.testing.assert_array_equal(_select_image(images, "missing", fallback_index=0), primary)
    np.testing.assert_array_equal(_select_image(images, "missing", fallback_index=1), wrist)


def test_unnormalize_actions_uses_q01_q99_range():
    actions = np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32)
    stats = {"q01": [0.0, 10.0, 20.0], "q99": [10.0, 20.0, 30.0]}

    expected = np.asarray([[0.0, 15.0, 30.0]], dtype=np.float32)
    np.testing.assert_allclose(_unnormalize_actions(actions, stats), expected)


def _ctx(session_id: str) -> SessionContext:
    return SessionContext(session_id=session_id, episode_id=f"{session_id}-ep")


def _torch():
    return pytest.importorskip("torch")


def _make_request(
    session_id: str, input_ids: list[int], *, token_type_ids: list[int] | None = None
) -> _PreparedRequest:
    torch = _torch()
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    token_tensor = torch.tensor([token_type_ids or [1] * len(input_ids)], dtype=torch.long)
    attention_tensor = torch.ones_like(input_tensor)
    return _PreparedRequest(
        ctx=_ctx(session_id),
        buffers=_SessionBuffers(image_inputs=deque(), action_tokens=deque()),
        final_inputs={
            "input_ids": input_tensor,
            "token_type_ids": token_tensor,
            "attention_mask": attention_tensor,
        },
        prompt_len=len(input_ids),
    )


def test_pad_prepared_requests_left_pads_decoder_inputs():
    _torch()
    server = CoWVLAModelServer()
    server._tokenizer = SimpleNamespace(pad_token_id=99)

    req_long = _make_request("a", [10, 11, 12], token_type_ids=[1, 1, 1])
    req_short = _make_request("b", [20, 21], token_type_ids=[7, 8])

    batch, max_prompt_len = server._pad_prepared_requests([req_long, req_short])

    assert max_prompt_len == 3
    assert batch["input_ids"].tolist() == [[10, 11, 12], [99, 20, 21]]
    assert batch["token_type_ids"].tolist() == [[1, 1, 1], [0, 7, 8]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]


def test_predict_batch_preserves_input_order(monkeypatch: pytest.MonkeyPatch):
    torch = _torch()
    server = CoWVLAModelServer()
    order = []

    monkeypatch.setattr(server, "_load_model", lambda: None)

    def _prepare_request(obs, ctx, buffers):
        order.append((obs["task_description"], ctx.session_id, buffers))
        value = len(order)
        return _PreparedRequest(
            ctx=ctx,
            buffers=buffers,
            final_inputs={
                "input_ids": torch.tensor([[value]], dtype=torch.long),
                "token_type_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            },
            prompt_len=1,
        )

    monkeypatch.setattr(server, "_prepare_request", _prepare_request)
    monkeypatch.setattr(
        server,
        "_predict_prepared_requests",
        lambda prepared: [{"actions": np.asarray([idx], dtype=np.float32)} for idx, _ in enumerate(prepared)],
    )

    obs_batch = [
        {"task_description": "first"},
        {"task_description": "second"},
        {"task_description": "third"},
    ]
    ctx_batch = [_ctx("s1"), _ctx("s2"), _ctx("s3")]

    results = server.predict_batch(obs_batch, ctx_batch)

    assert [int(r["actions"][0]) for r in results] == [0, 1, 2]
    assert [item[0] for item in order] == ["first", "second", "third"]


def test_predict_batch_reuses_session_buffers_for_same_session(monkeypatch: pytest.MonkeyPatch):
    torch = _torch()
    server = CoWVLAModelServer()
    seen_buffers = []

    monkeypatch.setattr(server, "_load_model", lambda: None)

    def _prepare_request(obs, ctx, buffers):
        seen_buffers.append((ctx.session_id, id(buffers)))
        return _PreparedRequest(
            ctx=ctx,
            buffers=buffers,
            final_inputs={
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "token_type_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            },
            prompt_len=1,
        )

    monkeypatch.setattr(server, "_prepare_request", _prepare_request)
    monkeypatch.setattr(
        server,
        "_predict_prepared_requests",
        lambda prepared: [{"actions": np.zeros(1, dtype=np.float32)} for _ in prepared],
    )

    obs_batch = [{"task_description": "a"}, {"task_description": "b"}]
    ctx_batch = [_ctx("shared"), _ctx("shared")]

    server.predict_batch(obs_batch, ctx_batch)

    assert seen_buffers[0][0] == "shared"
    assert seen_buffers[0][1] == seen_buffers[1][1]


def test_predict_batch_splits_duplicate_session_ids_into_separate_subbatches(monkeypatch: pytest.MonkeyPatch):
    torch = _torch()
    server = CoWVLAModelServer()
    calls: list[list[str]] = []

    monkeypatch.setattr(server, "_load_model", lambda: None)

    def _prepare_request(obs, ctx, buffers):
        value = len(calls) + len(obs["task_description"])
        return _PreparedRequest(
            ctx=ctx,
            buffers=buffers,
            final_inputs={
                "input_ids": torch.tensor([[value]], dtype=torch.long),
                "token_type_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            },
            prompt_len=1,
        )

    def _predict_prepared_requests(prepared):
        calls.append([request.ctx.session_id for request in prepared])
        return [{"actions": np.asarray([idx], dtype=np.float32)} for idx, _ in enumerate(prepared)]

    monkeypatch.setattr(server, "_prepare_request", _prepare_request)
    monkeypatch.setattr(server, "_predict_prepared_requests", _predict_prepared_requests)

    outputs = server.predict_batch(
        [
            {"task_description": "a"},
            {"task_description": "b"},
            {"task_description": "c"},
        ],
        [_ctx("shared"), _ctx("other"), _ctx("shared")],
    )

    assert calls == [["shared", "other"], ["shared"]]
    assert len(outputs) == 3


def test_prepare_request_keeps_ctx_on_prepared_request(monkeypatch: pytest.MonkeyPatch):
    torch = _torch()
    pytest.importorskip("transformers")

    server = CoWVLAModelServer(window_size=1)
    server._tokenizer = SimpleNamespace(bos_token="<bos>")
    server._processor = SimpleNamespace(
        tokenizer=lambda _: {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "token_type_ids": torch.tensor([[0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        video_process=lambda **_: {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "token_type_ids": torch.tensor([[0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
    )
    monkeypatch.setattr(server, "_encode_image", lambda image: torch.tensor([[9, 9]], dtype=torch.long))

    ctx = _ctx("real")
    buffers = server._make_session_buffers()
    prepared = server._prepare_request({"images": {"agentview": np.zeros((2, 2, 3), dtype=np.uint8)}}, ctx, buffers)

    assert prepared.ctx is ctx
    assert prepared.buffers is buffers


def test_predict_prepared_requests_nonlatent_slices_after_padded_prompt():
    torch = _torch()
    server = CoWVLAModelServer(chunk_size=2, action_dim=3)
    server._action_tokenizer = SimpleNamespace(vocab_size=4)
    server._generation_config = object()
    server._tokenizer = SimpleNamespace(pad_token_id=20)
    server._device = torch.device("cpu")
    server._use_latent_action = False
    server.eoa_token_id = 17

    seen: dict[str, list[list[int]]] = {}

    def _decode(tokens, *, time_horizon, action_dim):
        seen["tokens"] = [list(token) for token in tokens]
        assert time_horizon == 2
        assert action_dim == 3
        batch = len(tokens)
        return np.arange(batch * time_horizon * action_dim, dtype=np.float32).reshape(batch, time_horizon, action_dim)

    server._action_tokenizer.decode = _decode

    generated = torch.tensor(
        [
            [50, 51, 52, 7, 17, 20],
            [20, 60, 61, 10, 11, 17],
        ],
        dtype=torch.long,
    )
    server._model = SimpleNamespace(generate=lambda **_: generated)

    req_a = _make_request("a", [50, 51, 52])
    req_b = _make_request("b", [60, 61])

    outputs = server._predict_prepared_requests([req_a, req_b])

    assert seen["tokens"] == [[12], [9, 8]]
    assert req_a.buffers.action_tokens[0].tolist() == [[7, 17]]
    assert req_b.buffers.action_tokens[0].tolist() == [[10, 11, 17]]
    assert len(outputs) == 2
    assert outputs[0]["actions"].shape == (2, 3)


def test_predict_prepared_requests_latent_action_uses_generated_tokens_directly():
    torch = _torch()
    server = CoWVLAModelServer(chunk_size=2, action_dim=2, window_size=1)
    server._action_tokenizer = SimpleNamespace(vocab_size=4)
    server._generation_config = object()
    server._tokenizer = SimpleNamespace(pad_token_id=20)
    server._device = torch.device("cpu")
    server._use_latent_action = True
    server.eoa_token_id = 17

    seen: dict[str, list[list[int]]] = {}

    def _decode(tokens, *, time_horizon, action_dim):
        seen["tokens"] = [list(token) for token in tokens]
        return np.zeros((tokens.shape[0], time_horizon, action_dim), dtype=np.float32)

    server._action_tokenizer.decode = _decode
    server._model = SimpleNamespace(
        generate=lambda **_: torch.tensor([[7, 17, 20], [10, 11, 17]], dtype=torch.long),
        model=SimpleNamespace(embed_tokens=lambda input_ids: torch.zeros((*input_ids.shape, 4), dtype=torch.float32)),
        latent_action_query=torch.zeros((1, 1, 4), dtype=torch.float32),
        img_end_token_id=99,
    )

    req_a = _make_request("a", [1, 99, 5])
    req_b = _make_request("b", [2, 99, 6])

    outputs = server._predict_prepared_requests([req_a, req_b])

    assert seen["tokens"] == [[12], [9, 8]]
    assert len(outputs) == 2
    assert outputs[1]["actions"].shape == (2, 2)


@pytest.mark.anyio
async def test_episode_lifecycle_clears_session_buffers():
    server = CoWVLAModelServer()
    ctx = _ctx("episode")
    server._session_buffers["episode"] = server._make_session_buffers()

    await server.on_episode_end({}, ctx)
    assert "episode" not in server._session_buffers

    await server.on_episode_start({}, ctx)
    assert "episode" in server._session_buffers
