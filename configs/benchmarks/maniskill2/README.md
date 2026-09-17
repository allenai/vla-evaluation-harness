---
smoke_config: eval.yaml
---

# ManiSkill2

Generalizable manipulation benchmark (SAPIEN).
[Paper](https://arxiv.org/abs/2302.04659) | [GitHub](https://github.com/haosulab/ManiSkill2)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/maniskill2:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Full ManiSkill2 evaluation | 20 | 100 |

## Rendering without a GPU

SAPIEN rasterizes through Vulkan. `--render cpu` moves that to Mesa lavapipe and starts the
container with no device attached, which is the way to run this benchmark on a host whose GPU
Vulkan path is unusable -- on some H100 nodes rendering hangs inside the first capture or raises
`vk::Device::waitForFences: ErrorDeviceLost`, while the same image renders normally on A100s.

```bash
vla-eval run --render cpu -c configs/benchmarks/maniskill2/eval.yaml
```

SAPIEN 2.2.2 needs a lavapipe from Mesa 24.3 or newer, which is older than what Ubuntu 22.04
ships, so the image installs one from conda-forge into `/opt/lavapipe-env` and stashes its ICD at
`/opt/lavapipe/lvp_icd.json`. Nothing else in the image sees that Mesa. If the ICD is missing the
run fails at startup naming the reason rather than falling back to the GPU; `MANISKILL2_LAVAPIPE_ICD` points at
a manifest elsewhere.

### What it costs

Measured on one DGX-H100 node with no GPU attached, 256x256 cameras, one environment in
one process, against a random-action server. Three episodes across two config entries,
over two runs.

| | |
|---|---|
| Policy steps/s, one shard (`PickCube-v0`) | 18 to 20 |
| Frames black (RGB mean <= 1 of 255) | 0 of 125 |
| Frame RGB mean | 55.6 to 57.6 |

Every episode stepped its full budget with no errors, and each entry's aggregate records
`render.requested: cpu` with `applied_env` naming `/opt/lavapipe/lvp_icd.json` -- which is
what proves the backend engaged rather than something else working by accident. The second
entry reports the same env as the first: the renderer binds once per process, and later
entries replay what that call applied rather than claiming an empty (GPU) env.

This is a smoke measurement, not a throughput study: the GPU path's rate on the same host
was not measured, and neither was scaling across shards. `LP_NUM_THREADS` is lavapipe's
rasterizer thread count and defaults to 4 here.

See [docs/render-backends.md](../../../docs/render-backends.md) for the mode's interaction with
`docker.gpus`, and for every benchmark's support.
