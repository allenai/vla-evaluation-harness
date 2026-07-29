# Render Backends

Benchmarks render on the GPU by default. `--render cpu` moves the simulator to
software rendering and starts its container with no GPU attached, leaving the
device to the model server.

```bash
vla-eval run -c configs/benchmarks/libero/spatial.yaml --render cpu
scripts/run_sharded.sh -c configs/benchmarks/libero/spatial.yaml -n 16 --render cpu
```

Or in the config, which `--render` overrides:

```yaml
render: cpu   # top-level, alongside output_dir / server / docker
```

## When to use it

On a single-GPU host, simulator rendering and model inference land on the same
device. CUDA compute and GPU rendering readback on one device abort
nondeterministically in the NVIDIA driver — a controlled A/B measured 6/6 stable
runs when split across two devices against 4/4 aborting when co-tenant. CPU
rendering removes the contention entirely.

Software rendering is substantially slower per frame, so this trades simulator
throughput for a free GPU. It pays off when inference is the bottleneck, or when
GPU contention is crashing runs outright.

## Supported benchmarks

LIBERO (with Pro / Plus / Mem), RoboCasa, RoboCasa365, RoboCerebra, DuoBench and
VLABench — all MuJoCo, rendering through OSMesa.

Anything else fails fast, before the image is pulled, with the benchmark named:

```
ERROR: render: cpu is not supported by:
  MIKASABenchmark does not support render: cpu (declares render_backends: gpu)
```

Failing is deliberate. Falling back to the GPU would reinstate the crash the flag
exists to avoid, and a backend that is declared but doesn't engage is worse than
one that isn't offered.

RoboMME is the current example of that bar. Its lavapipe software-Vulkan path is
implemented, but it stays GPU-only until two things are fixed: its configs
bind-mount `/usr/share/vulkan/icd.d` over the image's, hiding the ICD that
`mesa-vulkan-drivers` installs, and the image never creates the alternate
`/opt/lavapipe/lvp_icd.json`. Its `ROBOMME_USE_LAVAPIPE` env var still works as
the broken-host workaround on the GPU path.

## Scope and interaction with `docker.gpus`

The setting is run-level, not per benchmark entry: the renderer binds at the first
simulator import and cannot be rebound in-process.

`render: cpu` implies no GPU device for the container. Where that meets an
explicit `docker.gpus`, the rules are:

| | outcome |
|---|---|
| `--render cpu` + `gpus:` in the config | CLI wins, forced to no device |
| `render: cpu` + `gpus:` in the same config | rejected — the config contradicts itself |
| `--render cpu` + `--gpus` | rejected — same precedence, neither outranks |
| `gpus: none` + `render: gpu` | rejected — the simulator needs a device |

Rejecting rather than guessing keeps a typed flag from being silently swallowed.

## Provenance

The requested mode and the env actually applied are recorded in the run metadata
and flow into the merge aggregate. The two can differ: RoboMME's
`ROBOMME_USE_LAVAPIPE=auto` probes the host and may pick either path.

Under sharding every process writes metadata under one `eval_id`, and the store is
first-writer-wins, so a shard that resolved differently would otherwise be
invisible. A disagreeing write sets `render.divergent` and logs both values, so the
record cannot claim one shard's answer for every episode.

## Adding support to a benchmark

Two lines plus a hook — see the Render Backends section of
[CONTRIBUTING.md](../CONTRIBUTING.md#render-backends).
