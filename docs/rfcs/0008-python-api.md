# RFC-0008: Python API for Training-Time Evaluation

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2026-09-13
- **Requires:** RFC-0002, RFC-0003
- **Superseded-By:** —

## Summary

Expose the CLI's two halves as functions: `serve_background()` hosts a `ModelServer` on a
thread of the calling process, `run()` executes an eval config, `evaluate()` composes them.
No protocol change; the container is still the WebSocket client and the model is still the
server.

## Motivation

Every external adopter found so far uses the harness after training, never during it. The
one lab that tried to embed it imported the runner, monkeypatched it, and rewrote the
batched loop. Reading their patches, the blocker was not the protocol direction but the
absence of an entry point: `vla-eval run`'s body lived in an argparse handler and the server
could only be started as a foreground process. A training script had to orchestrate three
processes by hand.

## Design

- `vla_eval.api.run(config, ...)` is `cmd_run` minus argparse and `sys.exit`. Docker
  execution and render validation moved out of `cli/main.py` (`cli/_docker.py`,
  `render.py`) so the library and the CLI share one implementation.
- `serve_async` gained a `ready(port)` callback so `serve_background` can bind port 0 and
  learn the port. The server runs on a fresh asyncio loop in a daemon thread; `close()` stops
  the loop.
- `evaluate(model_server, config, **kw)` is the composition. The model server is a
  `PredictModelServer` subclass holding a reference to the live model, so no weights are
  copied and specs are declared like any other server.
- Library defaults differ from the CLI where a CLI default would hurt a host process: the
  stall watchdog (`os._exit`) is off unless asked for, image pulls need `pull=True`,
  `no_save` is refused for Docker runs (results come back via the recording).

## Alternatives considered

- **Env-server mode** (container exposes `reset`/`step`, training loop owns the episode):
  needed to plug into LeRobot's `env_eval_freq`. Requires a second protocol, a client-side
  benchmark proxy, and live-mode semantics over RPC. Deferred until demand is demonstrated.
- **Example script only, no API:** two adopters already wrote this glue; owning it is cheaper
  than letting each copy rot.

## Verification

`tests/test_api.py` (real WebSocket, stub benchmark) and `tests/test_pusht_benchmark.py`
(real gym-pusht env). CI runs `examples/pusht_train_eval` (a standalone uv project) end to end
with a 20-step budget.
