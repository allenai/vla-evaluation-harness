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
- the RoboDojo eval assets (~38.5 GiB; `init_assets.sh` also keeps a copy in its git-LFS cache),
  mounted at runtime (licence-restricted, never baked in)

## Build the image

The upstream repo publishes the complete Isaac Sim 5.1 image recipe; vla-eval only adds a
thin layer. Submodules (IsaacLab, cuRobo, XPolicyLab) are COPY'd into the build context, so
they must be present in the clone:

```bash
git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
git -C RoboDojo checkout --recurse-submodules ee67a1468510da7624a089164402359f2afc72c8
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

A process runs its tasks one after another (the next task closes the last one's EvalEnv first). Run `eval.yaml`
directly, drive it with the protocol script (one `vla-eval run` and aggregate per task; disjoint `--filter`
regexes per GPU), or shard it with `scripts/run_sharded.sh` (shards take whole tasks). Resuming after a crash or
exit 3 needs the shared queue, so rerun through `run_sharded.sh` (also with `-n 1`) and the same eval id.

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

The adapter follows upstream `src/eval_client/main.py` (see the module docstring for line references):

- `seed` selects a published `Assets/Eval_Layout/RoboDojo/arx_x5/<seed>` group. A task's layouts form a queue
  in id order; each round takes the next `min(num_envs, episodes still to count)` ids on a fresh scene.
- `num_envs` (default 1) must match the policy's upstream setting: 1 when its XPolicyLab `deploy.yml` has
  `eval_batch: false`, else 10 (`robodojo_pi05` asks for 10). Layout stability depends on the round's
  composition, slot order included, so counted sets match upstream only at equal `num_envs`.
- Episode counts follow `_task.yml` (`episodes_per_task` may lower them, as upstream's `EVAL_NUM`): 50, or 25
  for the 12 Generalization tasks and their `_random` variants (at most 5 envs). Each task enforces its own
  `step_lim` (300–1900); the harness cap only backstops it.
- Layouts that do not settle, envs a task flags unstable mid-episode (the Franka competition tasks) and rounds
  that raise are recorded as `skipped`: not counted, not retried, left out of aggregates. PhysX-broken envs
  abandon their layout and the round reruns with refills, as upstream's PhysX monitor does for tasks with
  articulations. A task whose queue runs dry counts fewer episodes, as upstream.
- A round's episodes end together and are scored then (`score`: 1.0 on success, else process score / 100), so
  an episode's `steps` is its round's; its step rows stop where it ended.
- Five failing rounds in a row or a dead PhysX context raise `EnvUnusableError`: the shard
  returns its unfinished items and exits 3. A rerun with the same eval id resumes past every counted and
  skipped layout (upstream's own resume retries the skipped ones).
