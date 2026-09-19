# Python API

`vla_eval.evaluate`, `vla_eval.run`, and `vla_eval.serve_background` expose what the CLI does
as functions, so an evaluation can be called from a training script. See
[RFC-0008](rfcs/0008-python-api.md) for the design.

```python
import vla_eval
from vla_eval.model_servers.predict import PredictModelServer

class MyServer(PredictModelServer):
    def predict(self, obs, ctx): ...
    def get_action_spec(self): ...
    def get_observation_spec(self): ...

results = vla_eval.evaluate(MyServer(model), "configs/benchmarks/libero/spatial.yaml")
print(results[0]["mean_success"])
```

## `evaluate(model_server, config, **run_kwargs)`

Serves `model_server` in-process and runs `config` against it. Blocking. Equivalent to:

```python
with vla_eval.serve_background(model_server) as handle:
    results = vla_eval.run(config, server_url=handle.url, **run_kwargs)
```

## `run(config, *, server_url, output_dir, eval_id, no_save, docker, pull, benchmark_overrides, watchdog_timeout_s)`

The body of `vla-eval run`. Returns one `BenchmarkResult` dict per `benchmarks[]` entry.

| Argument | Meaning |
|---|---|
| `config` | Path to an eval YAML (`extends`, `${oc.env:...}` resolved) or a config dict |
| `docker` | `None` (default): container when `docker.image` is set, in-process otherwise. `False` forces in-process, `True` requires an image |
| `pull` | Allow pulling a missing image without a prompt. Off by default: images are tens of GB |
| `no_save` | In-memory results only. In-process runs only; Docker runs report through the recording |
| `benchmark_overrides` | Applied to every benchmark entry, e.g. `{"episodes_per_task": 10, "max_tasks": 1, "params": {"seed": 3}}`. `params` merges, other keys replace. Same rule, same code as `vla-eval run --benchmark-field max_steps=200` |
| `watchdog_timeout_s` | Arms the stall watchdog. It `os._exit`s the process, so leave it unset inside a training loop |

Docker runs use `--network host`, so the in-process server on `127.0.0.1` is reachable from the
container. Sharding is not wrapped; launch `run` once per shard with the same `eval_id` and call
`vla_eval.results.export.export_eval` afterwards, as `scripts/run_sharded.sh` does.

## `serve_background(model_server, *, host="127.0.0.1", port=0)`

Starts the WebSocket server on a daemon thread and returns a `ServerHandle` (`url`, `port`,
`close()`, context manager) once the socket is listening. `port=0` picks a free port.

## What is deliberately not here

- No tracking run is created unless the config's `tracking:` asks for one; log the returned
  dicts to the run you already own.
- No distributed awareness: call from rank 0 and barrier.
- No env-style `reset`/`step` interface. The harness owns the episode loop; a training
  framework that owns its own loop (LeRobot's `env_eval_freq`) cannot host it without a
  protocol change, which is out of scope for this API.
