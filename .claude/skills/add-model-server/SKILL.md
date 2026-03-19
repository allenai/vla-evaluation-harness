# Skill: add-model-server

Add a new VLA model server to the evaluation harness.

## Trigger

User asks to add/integrate a new model (e.g. "add OpenVLA server", "integrate RT-2").

## Steps

### 1. Gather Requirements

Ask the user (if not already provided):
- **Model name** (e.g. `openvla`)
- **Framework/library** (e.g. HuggingFace Transformers, custom repo)
- **Python dependencies** (torch version, model-specific packages)
- **Checkpoint source** (HuggingFace Hub model ID or local path)
- **Action output format** (dimension, chunk_size, continuous vs discrete)
- **Input requirements** (single image vs multi-view, needs proprioceptive state?)

### 2. Create Model Server Script

Create `src/vla_eval/model_servers/<name>.py` as a **uv script** (standalone, inline deps).

The file MUST start with a PEP 723 inline script metadata block:

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
# vla-eval = { path = "../../.." }
# <model-package> = { git = "https://github.com/org/repo.git", branch = "main" }
# ///
```

Subclass `PredictModelServer` (most models) or `ModelServer` (advanced async):

```python
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve


class MyModelServer(PredictModelServer):
    def __init__(self, checkpoint: str, *, chunk_size: int = 1, action_ensemble: str = "newest", **kwargs):
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self._model = None

    def _load_model(self) -> None:
        """Lazily load model on first predict() call."""
        if self._model is not None:
            return
        import torch
        # Load model here...
        self._model = ...

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        """Single-observation inference. Blocking call.

        Args:
            obs: {"images": {"cam_name": np.ndarray HWC uint8},
                  "task_description": str,
                  "states": np.ndarray (optional)}
            ctx: Session context (session_id, episode_id, step, is_first)

        Returns:
            {"actions": np.ndarray} with shape:
              - (action_dim,) if chunk_size == 1
              - (chunk_size, action_dim) if chunk_size > 1
        """
        self._load_model()
        # Run inference...
        return {"actions": np.array(actions, dtype=np.float32)}
```

### Key Patterns (from existing implementations)

**PredictModelServer features (inherited automatically):**
- **Action chunking**: When `chunk_size > 1`, return `(chunk_size, action_dim)` array. Framework auto-buffers and serves one action per step, re-inferring only when buffer empties.
- **Action ensemble**: `"newest"` (default), `"average"`, `"ema"` — blends overlapping chunks. Set via `action_ensemble=` in `__init__`.
- **Batched inference**: Override `predict_batch()` + set `max_batch_size > 1` for GPU-batched multi-shard eval.
- **Per-suite chunk_size**: Override `on_episode_start()` to set `self._session_chunk_sizes[ctx.session_id] = N` (see CogACT example).
- **CI/LAAS**: Set `continuous_inference=True` for continuous inference mode (DRAFT).

**Image handling:**
```python
from PIL import Image as PILImage
images = obs.get("images", {})
img_array = next(iter(images.values()))  # first camera
pil_image = PILImage.fromarray(img_array).convert("RGB")
```

**Task description:**
```python
text = obs.get("task_description", "")
```

**Lazy model loading**: Always use a `_load_model()` pattern. Do NOT load in `__init__`.

### 3. Add `if __name__ == "__main__"` Entry Point

The script must be runnable via `uv run`:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="<Model> server (uv script)")
    parser.add_argument("--checkpoint", required=True, help="HF model ID or local path")
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    server = MyModelServer(args.checkpoint)
    server.chunk_size = args.chunk_size
    server.action_ensemble = args.action_ensemble

    logger.info("Pre-loading model...")
    server._load_model()
    logger.info("Model ready, starting server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
```

### 4. Create Config YAML

Create `configs/model_servers/<name>.yaml`:

```yaml
# <Model Name> model server — <benchmark> checkpoint
# Weight: <HuggingFace model ID>
# Benchmark: <target benchmark>

script: "src/vla_eval/model_servers/<name>.py"
args:
  checkpoint: <org/model-id>
  chunk_size: 1
  port: 8000
```

The CLI runs this via: `vla-eval serve --config configs/model_servers/<name>.yaml`
which translates to: `uv run <script> --checkpoint <value> --chunk_size <value> --port <value>`

### 5. Verify

1. Run `make check` — lint + format + type check
2. Run `make test` — ensure existing tests still pass
3. Suggest user test: `vla-eval test -c configs/model_servers/<name>.yaml`
   (starts server, sends dummy observations from a StubBenchmark, checks for valid action response — requires `uv` + GPU + model weights)

### Reference Implementations

- **CogACT** (`model_servers/dexbotic/cogact.py`): Diffusion action head, chunk_size_map per suite, batched inference, text template option
- **starVLA** (`model_servers/starvla.py`): Auto-detecting framework, HuggingFace checkpoint download, monkey-patches for upstream compat

### Server Hierarchy

```
ModelServer (ABC)                    ← Advanced: async on_observation()
    └── PredictModelServer           ← Most models: blocking predict()
```

- Use `PredictModelServer` for standard request-response models (95% of cases)
- Use `ModelServer` only if you need async streaming or custom message handling

