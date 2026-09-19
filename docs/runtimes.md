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

```bash
pixi global install charliecloud
vla-eval run --runtime charliecloud --yes -c configs/benchmarks/libero/smoke_test.yaml
```

First use pulls with `ch-image` (public-mirror fallback like Docker), exports a directory with `ch-convert`,
injects the driver, and renames it into place under a per-directory lock. GPU exports are separate per host
driver version and never modified afterwards. `ch-run` starts from the image's own environment and binds the
same paths as the Docker path.

Sites that disable unprivileged user namespaces (`user.max_user_namespaces=0`, Debian's
`kernel.unprivileged_userns_clone=0`, Ubuntu 24.04's AppArmor restriction) can only run containers through a
setuid Apptainer, which vla-eval does not drive.

Not covered: `vla-eval test --benchmark` still launches Docker directly.
