---
name: add-model-server
description: "Add a new VLA model server to the evaluation harness. Use this skill whenever the user wants to integrate, create, or add a new model — e.g. 'add OpenVLA server', 'integrate RT-2', 'hook up my model', 'write a model server'. Also use when they ask how model servers work or want to understand the server interface."
---

# Add Model Server

Integrate a new VLA model into vla-eval. Model servers are standalone uv scripts that run a WebSocket server, receiving observations and returning actions.

## 1. Gather requirements

Ask the user for (if not already provided):
- **Model name** (e.g. `openvla`)
- **Framework/library** (e.g. HuggingFace Transformers, custom repo)
- **Python dependencies** (torch version, model-specific packages)
- **Checkpoint source** (HuggingFace Hub model ID or local path)
- **Action output format** (dimension, chunk_size, continuous vs discrete)
- **Input requirements** (single image vs multi-view, needs proprioceptive state?)

## 2. Create the model server script

Create `src/vla_eval/model_servers/<name>.py` as a **uv script** with PEP 723 inline metadata:

```python
# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "<model-package>",
#     "torch>=2.0",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# <model-package> = { git = "https://github.com/org/repo.git", rev = "<commit-sha>" }
#
# [tool.uv]
# exclude-newer = "<YYYY-MM-DD>T00:00:00Z"
# ///

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class MyModelServer(PredictModelServer):
    def __init__(self, checkpoint: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self._model = None

    def _load_model(self) -> None:
        """Lazily load model on first predict() call.

        Always use this pattern — never load in __init__(), because the
        server process may fork or the model may not be needed immediately.
        """
        if self._model is not None:
            return
        import torch
        # Load model here...
        self._model = ...

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        """Single-observation inference. Blocking call.

        Args:
            obs: {"images": {"cam_name": np.ndarray HWC uint8},
                  "task_description": str,
                  "state": np.ndarray (optional)}
            ctx: Session context (session_id, episode_id, step, is_first)

        Returns:
            {"actions": np.ndarray} with shape:
              - (action_dim,) for single actions
              - (chunk_size, action_dim) for action chunks
        """
        self._load_model()
        # Extract image and task description
        images = obs.get("images", {})
        img_array = next(iter(images.values()))
        pil_image = PILImage.fromarray(img_array).convert("RGB")
        text = obs.get("task_description", "")
        # Run inference...
        actions = ...
        return {"actions": np.array(actions, dtype=np.float32)}

    def get_action_spec(self) -> dict[str, DimSpec]:
        # Declare the action format this server produces.
        # The orchestrator compares this against the benchmark's spec
        # and warns on mismatches before wasting GPU hours.
        ...

    def get_observation_spec(self) -> dict[str, DimSpec]:
        # Declare what observations this server expects.
        ...


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(MyModelServer)
```

### `run_server()` — standard CLI entrypoint

`run_server(MyModelServer)` auto-generates argparse from the `__init__` signature, sets up logging, pre-loads the model (via `_load_model()` if present), and starts the WebSocket server. **Always use this instead of manual argparse.** It auto-discovers:
- All `__init__` parameters as `--flag` CLI arguments (bool → `--flag/--no-flag`, list → JSON string)
- Standard flags: `--host`, `--port`, `--address`, `--verbose`

### PEP 723 metadata conventions

- `vla-eval` source must use `editable = true`
- Pin a git `rev` (commit SHA) for reproducibility, not `branch`
- Set `exclude-newer` to the date dependencies were last verified

## PredictModelServer features

`PredictModelServer.__init__` accepts these keyword arguments:

| Parameter | Default | Purpose |
|---|---|---|
| `chunk_size` | `None` | Actions per inference call. `None` = no chunking (raw output used as-is). |
| `action_ensemble` | `"newest"` | Blending for overlapping chunks: `"newest"`, `"average"`, `"ema"`, or custom callable. |
| `ema_alpha` | `0.5` | Blend ratio for `"ema"` ensemble. |
| `max_batch_size` | `1` | Max observations per batch. `> 1` requires overriding `predict_batch()`. |
| `max_wait_time` | `0.01` | Seconds to wait for a full batch before dispatching partial. |

### Action chunking

When `chunk_size` is set and `predict()` returns a 2-D array `(chunk_size, action_dim)`, the framework buffers actions and serves one per step, only re-calling `predict()` when the buffer empties. If `predict()` returns 1-D `(action_dim,)`, chunking is bypassed.

### Batched inference

Override `predict_batch()` and set `max_batch_size > 1` for GPU-batched multi-shard evaluation:

```python
def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
    # Batch inference across concurrent sessions
    ...
```

### Per-episode chunk size

Override `on_episode_start()` to set per-session chunk sizes (e.g. different chunk sizes per benchmark suite):

```python
async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
    suite = config.get("params", {}).get("suite", "")
    self._session_chunk_sizes[ctx.session_id] = self.chunk_size_map.get(suite, 1)
    await super().on_episode_start(config, ctx)
```

### Observation params

Override `get_observation_params()` to tell the benchmark what observations the model needs (e.g. wrist camera, proprioceptive state). These are sent in the HELLO response and auto-merged into benchmark params:

```python
def get_observation_params(self) -> dict[str, Any]:
    return {"include_wrist_image": True, "include_state": True}
```

## Server hierarchy

```
ModelServer (ABC)                ← Advanced: async on_observation()
    └── PredictModelServer       ← Most models: blocking predict()
```

Use `PredictModelServer` for standard request-response models (95% of cases). Use `ModelServer` directly only for async streaming or custom message handling.

## 3. Create config YAML

Create configs in a subdirectory `configs/model_servers/<name>/`:

```yaml
# configs/model_servers/<name>/<name>.yaml
# <Model Name> model server — <benchmark> checkpoint
# Weight: <HuggingFace model ID>

script: "src/vla_eval/model_servers/<name>.py"
args:
  checkpoint: <org/model-id>
  chunk_size: 1
  port: 8000
```

For multiple benchmark-specific configs, use `extends` to inherit from a shared base:

```yaml
# configs/model_servers/<name>/_base.yaml
script: "src/vla_eval/model_servers/<name>.py"
args:
  port: 8000

# configs/model_servers/<name>/libero.yaml
extends: _base.yaml
args:
  checkpoint: org/model-libero
  chunk_size: 16
```

The `extends` key deep-merges `args` from the base config. The CLI runs this via `vla-eval serve -c configs/model_servers/<name>/<name>.yaml`.

## 4. Verify

```bash
make check                                            # lint + format + type check
make test                                             # existing tests still pass
vla-eval test -c configs/model_servers/<name>.yaml    # smoke-test (starts server, sends dummy obs, checks response — requires uv + GPU + model weights)
```

**Don't add `tests/test_<name>_server.py` with mocked model libraries.**
`tests/` is for harness mechanics, not per-model integration.  Fake
`transformers` / `torch.nn` / custom inference libs drift from upstream
each release and miss the real bugs (tokenizer versions,
checkpoint-format drift, action denormalisation).  Verify via the
smoke test above.

## Reference implementations

| Model | File | Key patterns |
|---|---|---|
| CogACT | `model_servers/dexbotic/cogact.py` | Diffusion action head, `chunk_size_map` per suite, batched inference |
| starVLA | `model_servers/starvla.py` | Auto-detecting framework, HF checkpoint download, monkey-patches for upstream compat |
