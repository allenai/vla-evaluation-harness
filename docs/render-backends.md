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

A simulator is not a renderer, so the table separates the two. MuJoCo draws with its
own built-in OpenGL renderer; what a backend chooses is the **GL context provider** —
EGL on an NVIDIA device for `gpu`, OSMesa (pure-software GL) or an EGL software device
for `cpu`. SAPIEN draws through Vulkan, where lavapipe is Mesa's software driver.
PyBullet ships two rasterizers of its own (hardware OpenGL via its EGL plugin, and the
software TinyRenderer). Kinetix computes frames as JAX arrays and has no GL at all.

The `cpu` column was measured on each benchmark's image with no GPU attached; nothing
is inferred from the simulator's name. The `gpu` column is the image's shipped NVIDIA
path.

| Benchmark | Simulator | `gpu` | `cpu` | `cpu` render path | Notes |
|---|---|:--:|:--:|---|---|
| LIBERO (+ Pro / Plus / Mem) | MuJoCo | ✅ | ✅ | OSMesa | |
| RoboCasa | MuJoCo | ✅ | ✅ | OSMesa | |
| RoboCasa365 | MuJoCo | ✅ | ✅ | OSMesa | |
| RoboCerebra | MuJoCo | ✅ | ✅ | OSMesa | |
| DuoBench | MuJoCo + rcs | ✅ | ✅ | EGL on Mesa's software device | Not OSMesa: its rcs camera stack bootstraps a real EGL context regardless of `MUJOCO_GL`, so cpu keeps EGL and pins `MUJOCO_EGL_DEVICE_ID` to the only (software) device |
| VLABench | dm_control (MuJoCo) | ✅ | ✅ | OSMesa | |
| MolmoSpaces-Bench | MuJoCo | ✅ | ✅ | OSMesa | AI2-THOR lineage is in the assets, not the renderer. The adapter stubs the macOS-only `mujoco.cgl` module that molmo_spaces calls on the GPU-less path |
| RoboMME | SAPIEN 3.0.3 | ✅ | ✅ | lavapipe (software Vulkan) | Its configs mount the host's `/usr/share/vulkan/icd.d` over the image's, so a host without Mesa needs `ROBOMME_LAVAPIPE_ICD`. See below |
| CALVIN | PyBullet | ✅ | ✅ | TinyRenderer | The EGL plugin aborts the whole process with no GPU, so cpu swaps it for PyBullet's built-in rasterizer — frames are close to, but not pixel-identical with, the GPU path's |
| Kinetix | JAX (no GL) | ✅ | ✅ | `JAX_PLATFORMS=cpu` | No GL: frames are computed as JAX arrays, so the device switch is the whole backend |
| SimplerEnv | SAPIEN 2.2.2 | ✅ | ❌ | — | SAPIEN requires the Vulkan extension `VK_KHR_external_semaphore_fd` at device creation; lavapipe does not implement it (verified on Mesa 23.2 and 25.0) |
| ManiSkill2 | SAPIEN 2.2.2 | ✅ | ❌ | — | same |
| RoboTwin | SAPIEN 3.0.0b1 | ✅ | ❌ | — | same |
| MIKASA-Robo | SAPIEN 3.0.0b1 | ✅ | ❌ | — | same |
| BEHAVIOR-1K | OmniGibson (Isaac Sim) | ✅ | ❌ | — | Isaac Sim dumps core during extension startup with no GPU |
| RoboDojo | Isaac Lab | ✅ | ❌ | — | Isaac reports `ERROR_INCOMPATIBLE_DRIVER` / "Failed to create any GPU devices" with no GPU; the RTX renderer has no software path |
| RLBench | CoppeliaSim | — | — | — | The shipped image cannot render on either backend: all-black frames with and without a GPU, and the pinned RLBench 1.1.0 predates the adapter's imports. Its Xvfb pipeline is Mesa software GL even when a GPU is attached |

**AMD GPUs**: the harness can attach ROCm devices — on a ROCm runtime, `docker.gpus`
mounts `/dev/kfd` and `/dev/dri` instead of passing `--gpus` — but no benchmark's
*rendering* has been measured on ROCm. The `gpu` column makes no claim about AMD.

Anything not declared fails fast, before the image is pulled:

```
ERROR: render: cpu is not supported by:
  MIKASABenchmark does not support render: cpu (declares render_backends: gpu)
```

Failing is deliberate. Falling back to the GPU would reinstate the crash the flag
exists to avoid, and a backend that is declared but doesn't engage is worse than
one that isn't offered.

SAPIEN splits by version rather than by family: 3.0.3 (RoboMME) renders through
lavapipe with no GPU, while 2.2.2 and 3.0.0b1 demand `VK_KHR_external_semaphore_fd`
at `vkCreateDevice` — before any shader or scene configuration — and lavapipe does
not implement that extension. A newer Mesa does not close the gap (the same failure
reproduces against Mesa 25.0 lavapipe); 3.0.3 dropped the hard requirement. Those
four are not waiting on harness work — they need newer SAPIEN builds.

RoboMME carries one caveat. Its configs bind-mount `/usr/share/vulkan/icd.d` over
the image's copy, so on a host that ships no Mesa, the ICD file the image's
`mesa-vulkan-drivers` installed is hidden and lavapipe cannot be resolved. That
fails loudly at startup, naming the reason; point `ROBOMME_LAVAPIPE_ICD` at an ICD
to resolve it. Its separate `ROBOMME_USE_LAVAPIPE` env var remains the broken-host
workaround on the GPU path.

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
