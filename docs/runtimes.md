# Container runtimes

The benchmark side runs from a pinned OCI image under one of two runtimes. Selection:
`--runtime` > `$VLA_EVAL_RUNTIME` > `docker.runtime:` in the YAML > `docker`.

| | Docker (default) | Charliecloud |
|---|---|---|
| Needs | Docker daemon, `docker` group, NVIDIA Container Toolkit for GPUs | `ch-*` tools on `PATH` (`pixi global install charliecloud`, >= 0.45.1) and unprivileged user namespaces (`unshare -Ur true` must succeed) |
| OS | Linux, macOS, Windows (Linux VM) | Linux only |
| Runs as | image default (root) unless `docker.user` | the calling user; `docker.user` ignored |
| GPU | `--gpus` | `ch-fromhost --nvidia` injects the host driver into the export (needs `nvidia-container-cli`); `docker.gpus` maps to `CUDA_VISIBLE_DEVICES`, unset/`all` inherits the job's mask |
| CPU pinning (`docker.cpus`) | `--cpuset-cpus` | not applied |
| `docker.build` | `docker build -t <image> -f <dockerfile> <context>` when the image is missing, or with `--build`; `image` defaults to `<parent>-<dir>:vla-eval` | `ch-image build` when not in storage, then export; `--build` rebuilds and drops the image's other cached exports, so do not force it while another run of the same image is active |
| Image cache | Docker's store | `~/.cache/vla-eval/charliecloud/<image>[+nvidia-<driver>]` (`VLA_EVAL_HOME` overrides); delete to re-pull |
| Image format | n/a | `docker.charliecloud.image_format`: `auto` (default), `dir`, or `squashfs`; `$VLA_EVAL_CH_IMAGE_FORMAT` overrides |

```bash
pixi global install charliecloud
vla-eval run --runtime charliecloud --yes -c configs/benchmarks/libero/smoke_test.yaml
```

First use pulls with `ch-image` (public-mirror fallback like Docker), exports a directory with `ch-convert`,
injects the driver, and renames it into place under a per-directory lock. GPU exports are separate per host
driver version and never modified afterwards. Image preparation is serialized per export and shared
`ch-image` storage. `ch-run` starts from the image's own environment and binds the same paths as the Docker path.

Storage locking uses `<storage>.vla-eval-lock`, owned by the current user and opened read/write.
The storage parent must allow creating it, or an administrator must provision it with those permissions.
Keep this file in place; locking failures stop image preparation. Direct `ch-image` invocations do not use this lock.

## SquashFS exports

The default `auto` prefers SquashFS when the packer, mount tools and `/dev/fuse` are available.
It reuses cached SquashFS images without requiring the packer, and falls back to a directory export
with a logged reason if prerequisites are missing or mounting fails. Set
`docker.charliecloud.image_format: dir` or `squashfs` to force a format; `VLA_EVAL_CH_IMAGE_FORMAT` overrides.

SquashFS stores each export in `charliecloud/.squashfs/<image>[+nvidia-<driver>].sqfs`, reducing inode use and
metadata traffic on shared filesystems. Preparation still needs a temporary directory for
`ch-convert` and GPU driver injection before packing with `mksquashfs` (`squashfs-tools`).

Each run mounts its own read-only view using `squashfuse_ll` or `squashfuse`, then unmounts it on exit.
This works even when `ch-run` lacks built-in SquashFUSE support. Compute nodes need the mount tool,
`fusermount3` or `fusermount`, and usable `/dev/fuse`. Forced SquashFS reports errors instead of falling back.
Both formats are cached side by side; `--build` invalidates the image's other cached exports.

Sites that disable unprivileged user namespaces (`user.max_user_namespaces=0`, Debian's
`kernel.unprivileged_userns_clone=0`, Ubuntu 24.04's AppArmor restriction) can only run containers through a
setuid Apptainer, which vla-eval does not drive.

Not covered: `vla-eval test --benchmark` still launches Docker directly.
