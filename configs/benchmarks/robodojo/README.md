# RoboDojo

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) is an Isaac Lab
benchmark for evaluating manipulation policies across tasks and robot
embodiments ([paper](https://arxiv.org/abs/2607.04434)). This integration uses
RoboDojo's native scene, reward, reset, and observation code and vla-eval's
model-server protocol.

## Requirements

- NVIDIA GPU and Container Toolkit
- acceptance of the NVIDIA Isaac Sim EULA
- a local RoboDojo asset checkout (about 90 GiB in the current distribution)

Set the asset path before running:

```bash
export ROBODOJO_ASSETS=/absolute/path/to/RoboDojo/Assets
```

Clone the public release and build RoboDojo's official image first, then build
the thin vla-eval layer:

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git ~/repo/RoboDojo-release
git -C ~/repo/RoboDojo-release checkout e9ef978fd5d78845bc0812ea0a1e7229f274051f
docker build -t robodojo:cuda12.8 ~/repo/RoboDojo-release
docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo
```

Run the one-episode integration check:

```bash
vla-eval test -c configs/benchmarks/robodojo/smoke_test.yaml
```

Run the initial three-task dual-X5 protocol:

```bash
vla-eval run -c configs/benchmarks/robodojo/eval.yaml
```

## Interface

The default `arx_x5` config is RoboDojo's dual-X5 embodiment. Its action is a
14-D absolute joint vector:

```text
left arm (6), left gripper (1), right arm (6), right gripper (1)
```

Observations contain an `images` camera dictionary, a natural-language
`task_description`, and concatenated joint `state`.  Success comes directly
from RoboDojo's native reward manager.

The `seed` parameter selects an
`Assets/Eval_Layout/RoboDojo/arx_x5/<seed>` layout group. Episodes traverse
the task's numbered JSON layouts in that group.

The initial config is a three-task integration slice. Expanding it to the full
published 42-task protocol requires separate configs for tasks that switch to
the mixed X5/Franka embodiment and for the task-specific 25/50 episode counts.
