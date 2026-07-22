---
smoke_config: smoke.yaml
---

# RoboCasa365

Official 50-task multi-task benchmark using RoboCasa's Panda-Omron Gymnasium wrapper.
[Paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf) ·
[Code](https://github.com/robocasa/robocasa) ·
[Protocol](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html) ·
[Leaderboard](https://robocasa.ai/leaderboard.html)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robocasa:latest`

## Configs

| File | Description | Tasks | Episodes/task |
| --- | --- | ---: | ---: |
| `rc365.yaml` | Official multi-task protocol | 50 | 50 |
| `rc365_s2b_qualification.yaml` | Base config for one S2B qualification shard | 1 | 1 |
| `smoke.yaml` | Contract smoke test | 2 | 1 |

The adapter reads task membership and task-specific horizons from RoboCasa's registry.
It evaluates the target50 tasks in pretraining kitchens (`split: pretrain`), matching the official multi-task leaderboard protocol.
Success is sampled and accumulated at the end of each 16-action chunk, while every episode continues to its task-specific horizon.
The wire contract preserves all 12 Panda-Omron dimensions in the official dataset order instead of padding a 7-D arm action.

## Qualification-slice parity

The parity script compares the isolated RC365 adapter with the read-only
`rc365_s2b.environments.OfficialRoboCasaEnv` for the six System 1
qualification tasks at seed 0. It checks repeated-reset determinism, raw
observation keys, shapes, dtypes, values, and simulator state across the same
recorded 32-step action sequence.

```bash
SIM_PYTHON=/path/to/system1/.venv/bin/python
REFERENCE_SRC=/path/to/projects/rc365-s2b/src
MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 "$SIM_PYTHON" \
  scripts/check_rc365_adapter_parity.py \
  --reference-src "$REFERENCE_SRC"
```

The output is `results/rc365_adapter_qualification_parity.json`. The script
redirects RoboCasa's generated temporary XML files to `/tmp`; it does not
modify the simulator environment or the reference source tree. Full GR00T or
MLLM inference is outside adapter parity and remains GPU or Slurm-deferred.

## Implementation boundary

- Reused upstream: the registered RoboCasa Gym environment, task registry, pretrain split, per-task horizon, observation schema, and strict success predicate.
- Implemented here: canonical observation mapping and lossless 12-D named-action decoding.
- Dependency boundary: Docker verifies RoboCasa `1.0.1` and robosuite `1.5.2` at tested patch revisions because those upstream fixes have no new semantic tags.

## CPU rendering

Set `VLA_EVAL_RENDER=cpu` to select Mesa llvmpipe through OSMesa. Set
`docker.gpus: none`, or pass `--gpus none`, so the benchmark container receives
no GPU device. The model server remains on the host GPU. GPU rendering through
the EGL device platform remains the default.

```bash
VLA_EVAL_RENDER=cpu vla-eval run \
  --config configs/benchmarks/robocasa/smoke.yaml \
  --gpus none
```

CPU mode sets `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`, forces
software GL, and clears `EGL_PLATFORM` and `MUJOCO_EGL_DEVICE_ID` before
MuJoCo is imported.

Container verification requires a Docker host:

```bash
docker build -f docker/Dockerfile.base -t vla-eval-base:rc365-cpu .
docker build \
  --build-arg BASE_IMAGE=vla-eval-base:rc365-cpu \
  -f docker/Dockerfile.robocasa \
  -t vla-eval-robocasa:rc365-cpu .
docker run --rm \
  -e VLA_EVAL_RENDER=cpu \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e CUDA_VISIBLE_DEVICES=-1 \
  --entrypoint /opt/conda/bin/conda \
  vla-eval-robocasa:rc365-cpu \
  run --no-capture-output -n robocasa \
  python /workspace/scripts/check_robocasa_cpu_render.py
```

## RC365 S2B qualification

The qualification runner expands the reference seed manifest in stable task
and seed order, then interleaves `gold-s2`, `global-s1`, and `random-valid`.
Each invocation owns exactly one task, seed, and condition. Qualification-only
success checks and 16-step chunk telemetry live in
`RoboCasaS2BQualificationBenchmark`, leaving the upstream adapters unchanged.

```bash
export RC365_S2B_ROOT=/path/to/projects/rc365-s2b
export RC365_PHASE_MANIFEST=/path/to/phase_manifest.jsonl
export ROBOCASA_GR00T_N15_CKPT=/path/to/checkpoint-120000
export ROBOCASA_MODALITY_JSON=/path/to/modality.json
export VLA_EVAL_ROBOCASA_IMAGE=vla-eval-robocasa:rc365-cpu

uv run python -m vla_eval.rc365_s2b_qualification resolve \
  --rung dev --array-index 0 \
  --seed-manifest "$RC365_S2B_ROOT/config/qualification_seeds.json"

uv run python -m vla_eval.rc365_s2b_qualification run \
  --rung dev --array-index 0 \
  --seed-manifest "$RC365_S2B_ROOT/config/qualification_seeds.json" \
  --phase-manifest "$RC365_PHASE_MANIFEST" \
  --render cpu --dev
```

Outputs are one-record JSONL files under
`results/rc365_s2b_qualification/<rung>/<condition>/`. Records use schema
`rc365-s2b-exec-episode-v1` and include strict success, termination, steps,
chunks, System 2 calls, run configuration, and provenance.

```bash
rc365-s2b score-qualification \
  results/rc365_s2b_qualification/dev/gold-s2 \
  results/rc365_s2b_qualification/dev/global-s1 \
  results/rc365_s2b_qualification/dev/random-valid
```

Submit the Slurm template with the rung's array bound. CPU rendering requests
one GPU for the policy. GPU rendering needs two GPUs.

```bash
# dev: 0-17, training-dev: 0-89, frozen: 0-449
sbatch --array=0-17 \
  --export=ALL,QUALIFICATION_RUNG=dev,VLA_EVAL_RENDER=cpu,VLA_EVAL_ROBOCASA_IMAGE=vla-eval-robocasa:rc365-cpu \
  scripts/run_rc365_s2b_qualification.sbatch
```
