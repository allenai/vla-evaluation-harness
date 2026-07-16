---
smoke_config: smoke_test.yaml
---

# RoboDojo

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) is an Isaac Lab benchmark
evaluating bimanual manipulation policies on 42 simulation tasks across five capability
dimensions ([paper](https://arxiv.org/abs/2607.04434)). This integration drives RoboDojo's
native scene/reward/reset/observation code through vla-eval's model-server protocol.

## Requirements

- NVIDIA GPU + Container Toolkit, acceptance of the NVIDIA Isaac Sim EULA
- the RoboDojo eval assets (~90 GiB), mounted at runtime (licence-restricted, never baked in)

## Build the image

The upstream repo publishes the complete Isaac Sim 5.1 image recipe; vla-eval only adds a
thin layer. Submodules (IsaacLab, cuRobo, XPolicyLab) are COPY'd into the build context, so
they must be present in the clone:

```bash
git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
git -C RoboDojo checkout --recurse-submodules e9ef978fd5d78845bc0812ea0a1e7229f274051f
docker build -t robodojo:cuda12.8 RoboDojo
docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo
```

The image is NO_REDIST (bundles Isaac Sim): build locally, never push to a public registry.

## Download the assets

Upstream's downloader fetches the assets from the Hugging Face dataset
`RoboDojo-Benchmark/RoboDojo` (sparse git + LFS):

```bash
cd RoboDojo && bash scripts/init_assets.sh
export ROBODOJO_ASSETS=$(pwd)/Assets
```

Without `ROBODOJO_ASSETS`, configs fall back to
`~/.cache/vla-eval/assets/robodojo/Assets`.

## Run

```bash
vla-eval test -c configs/benchmarks/robodojo/smoke_test.yaml   # one-episode integration check
scripts/run_robodojo_protocol.sh                               # full 42-task published protocol
```

Isaac's simulation context is process-global, so each container run evaluates exactly
one task (upstream's `eval_policy.sh` has the same shape). The protocol driver reads
`eval.yaml` and launches one `vla-eval run` per task; parallelise by running several
instances with disjoint `--filter` regexes (one lane per GPU).

## Interface

- **Action (14-D absolute joints):** left arm (6), left gripper (1), right arm (6),
  right gripper (1). All 42 tasks use the dual ARX-X5 policy action space; the Franka arm
  in the three `dual_x5_and_franka_competition` tasks is scripted, not policy-controlled.
- **Observation:** `images` (RoboDojo camera dict), `task_description` (the task's language
  instruction), and optional concatenated joint `state`.
- **Metrics:** `success` (binary, from RoboDojo's native reward manager) and `score`
  (paper metric: 1.0 on success, else partial task progress). The paper reports
  score/success-rate scaled by 100 and averages the five dimension means.

## Protocol notes

- `seed` selects a published `Assets/Eval_Layout/RoboDojo/arx_x5/<seed>` layout group;
  episode indices address the deterministic numbered layouts inside it, matching upstream's
  layout ids one-to-one.
- Episode counts follow the published `_task.yml`: 50 per task, except the 12
  Generalization tasks (25 standard + 25 via their `_random` variants). Each task enforces
  its own `step_lim` (300–1900); the harness cap only backstops it.
- Bad layouts (scene instability, broken scene builds) are skipped and replaced from the
  group's spares (55–65 layouts per task), matching the official protocol; each episode
  result records the `layout_id` it actually ran.
