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

## Support by benchmark

Every entry below was measured on the published image with no GPU attached.

| Benchmark | `cpu` | Renderer | Notes |
|---|:--:|---|---|
| LIBERO (+ Pro / Plus / Mem) | ✅ | MuJoCo → OSMesa | |
| RoboCasa | ✅ | MuJoCo → OSMesa | |
| RoboCasa365 | ✅ | MuJoCo → OSMesa | |
| RoboCerebra | ✅ | MuJoCo → OSMesa | |
| DuoBench | ✅ | MuJoCo → OSMesa | |
| VLABench | ✅ | dm_control / MuJoCo → OSMesa | |
| MolmoSpaces-Bench | ✅ | MuJoCo → OSMesa | AI2-THOR lineage is in the assets, not the renderer |
| Kinetix | ✅ | JAX | No GL: frames are computed, so `JAX_PLATFORMS=cpu` is the whole switch |
| RoboMME | ✅ | SAPIEN 3.0.3 → lavapipe | Its configs mount the host's `/usr/share/vulkan/icd.d` over the image's, so a host without Mesa needs `ROBOMME_LAVAPIPE_ICD`. See below |
| CALVIN | ❌ | PyBullet → EGL | Env instantiation fails under software GL; passes on GPU |
| SimplerEnv | ❌ | SAPIEN 2.2.2 | lavapipe: `vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent` |
| ManiSkill2 | ❌ | SAPIEN 2.2.2 | same |
| RoboTwin | ❌ | SAPIEN 3.0.0b1 | same |
| MIKASA-Robo | ❌ | SAPIEN 3.0.0b1 | same |
| BEHAVIOR-1K | ❌ | Isaac / OmniGibson | Not measured — licence-gated image |
| RoboDojo | ❌ | Isaac Lab | Not measured — needs the RTX renderer |
| RLBench | ❌ | CoppeliaSim | Not measured — licence-gated image |

Anything not declared fails fast, before the image is pulled:

```
ERROR: render: cpu is not supported by:
  MIKASABenchmark does not support render: cpu (declares render_backends: gpu)
```

Failing is deliberate. Falling back to the GPU would reinstate the crash the flag
exists to avoid, and a backend that is declared but doesn't engage is worse than
one that isn't offered.

SAPIEN splits by version rather than by family: 3.0.3 (RoboMME) renders through
lavapipe with no GPU, while the 2.2.2 and 3.0.0b1 images fail against the same ICD.
Those four are not waiting on harness work — they need newer SAPIEN builds.

RoboMME carries one caveat. Its configs bind-mount `/usr/share/vulkan/icd.d` over
the image's copy, so on a host that ships no Mesa the ICD `mesa-vulkan-drivers`
installed is hidden and lavapipe cannot be resolved. That fails loudly at startup,
naming the reason; point `ROBOMME_LAVAPIPE_ICD` at an ICD to resolve it. Its
separate `ROBOMME_USE_LAVAPIPE` env var remains the broken-host workaround on the
GPU path.

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
