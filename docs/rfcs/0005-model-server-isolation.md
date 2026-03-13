# RFC-0005: Model Server Dependency Isolation via uv Scripts

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2025-02-22
- **Requires:** RFC-0003
- **Superseded-By:** —

## Summary

Model servers should use PEP 723 inline script metadata ("uv scripts") for dependency isolation. This complements Docker isolation for benchmarks — benchmarks need strong OS-level isolation, while model servers need lightweight Python-level isolation.

## Motivation

### The Problem
Different VLA models require different dependencies:
- CogACT: dexbotic + transformers 4.40 + diffusers
- OpenVLA: transformers 4.38 + specific tokenizer
- Pi0: jax + flax (or torch + custom layers)

Currently, model servers run on the host with no isolation.

### Why Not Docker?
- GPU passthrough is cumbersome (nvidia-container-toolkit, --gpus all)
- CUDA/cuDNN pinned in image — host mismatch causes failures
- Model checkpoint/HuggingFace cache mounting adds complexity
- Disproportionate overhead for "one Python script"

### Why uv Scripts Fit
- Model servers are fundamentally "one Python script"
- GPU uses host drivers directly — most natural path
- Python-level isolation sufficient (no C library conflicts between models)
- uv creates venv + installs deps automatically — zero user friction
- PEP 723 is a Python standard

## Design

### Script Format

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vla-eval",
#     "dexbotic>=0.2",
#     "torch>=2.0",
#     "transformers>=4.40",
# ]
# ///

from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

class CogACTModelServer(PredictModelServer):
    chunk_size = 12
    action_ensemble = "newest"

    def predict(self, obs, ctx):
        ...

if __name__ == "__main__":
    serve(CogACTModelServer(...), port=8000)
```

### Execution
```bash
vla-eval serve --config configs/model_servers/dexbotic_cogact_libero.yaml
```

### Asymmetric Isolation Strategy

| | Benchmarks | Model Servers |
|---|---|---|
| Isolation reason | Simulator conflicts (robosuite vs SAPIEN) | Model dep differences (transformers versions) |
| GPU needed | Often (rendering) | Yes (inference) |
| Conflict severity | Strong — OS libs, C bindings | Weak — Python packages |
| Right tool | Docker | uv script |

### External Contribution Model
- Researcher creates serve_my_model.py in their own repo
- Script declares vla-eval as dependency
- No fork/modify needed
- `uv run serve_my_model.py` just works

## Open Questions
- How to handle model servers needing system-level deps? Fall back to Docker?
- Cache management for overlapping deps across scripts?

## Implementation Status
- ✅ PEP 723 inline script metadata — CogACT reference (`model_servers/cogact.py`)
- ✅ `vla-eval serve` CLI wrapping `uv run` (`cli/main.py`)
- ✅ Model server YAML config format (`configs/model_servers/`)
- ✅ Documentation (`CONTRIBUTING.md`)

