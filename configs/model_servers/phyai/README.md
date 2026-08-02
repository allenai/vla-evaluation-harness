---
smoke_config: null
---

# PhyAI model server

Serves a converted PhyAI pi0.5 checkpoint for LIBERO through the
vla-evaluation-harness WebSocket protocol.

The server pins PhyAI commit
[`c5a9044`](https://github.com/rebecca26358/phyai/commit/c5a904493b74b29a9126a11caaa72ebd51385169),
which is tracked by [PhyAI PR #21](https://github.com/mingti-org/phyai/pull/21).

## Prerequisites

- Python 3.12
- A CUDA GPU supported by PhyAI
- A converted PhyAI pi0.5 LIBERO checkpoint
- The `google/paligemma-3b-pt-224` tokenizer files or Hugging Face access

Set the checkpoint and, when using local tokenizer files, the tokenizer path:

```bash
export PHYAI_CHECKPOINT_PATH=/data/models/pi05_libero_phyai_converted
export PHYAI_TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
```

## Usage

Start the model server:

```bash
vla-eval serve --config configs/model_servers/phyai/libero.yaml
```

In another terminal, run the LIBERO smoke test:

```bash
vla-eval run --config configs/benchmarks/libero/smoke_test.yaml
```

Run a full suite by selecting its benchmark config, for example:

```bash
vla-eval run --config configs/benchmarks/libero/spatial.yaml
```

The model server requests the agent-view image, wrist image, proprioceptive
state, and task description. It runs native PhyAI inference with a 10-action
chunk and returns 7-dimensional LIBERO actions.
