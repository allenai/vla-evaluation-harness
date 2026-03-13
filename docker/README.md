# Docker Images

Benchmark environments for `vla-eval`. Each benchmark runs in its own container with CPU-only PyTorch; GPU model servers run on the host.

## Image Hierarchy

`Dockerfile.base` provides the common layer (CUDA 12.1, EGL/Vulkan, Miniforge, uv). Each `Dockerfile.<bench>` extends it with benchmark-specific dependencies. See `ls docker/Dockerfile.*` for the full list.

## Build & Push

```bash
# Build all images (base first, then benchmarks)
docker/build.sh

# Build a single benchmark
docker/build.sh libero

# Build with a version tag
docker/build.sh --tag 0.2.0

# Push all images (requires: docker login ghcr.io)
docker/push.sh --tag 0.2.0

# Push a single image
docker/push.sh --tag 0.2.0 libero
```

Images are published to `ghcr.io/allenai/vla-evaluation-harness/<name>:<tag>`.

## Adding a New Benchmark

1. Create `Dockerfile.<name>` — use `ARG BASE_IMAGE` and install benchmark-specific deps.
2. Add `<name>` to the `BENCHMARKS` array in `build.sh` and `IMAGES` array in `push.sh`.
