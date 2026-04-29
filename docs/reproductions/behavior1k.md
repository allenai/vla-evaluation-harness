# BEHAVIOR-1K — Reproduction Status

[Challenge site](https://behavior.stanford.edu/challenge/) |
[Leaderboard](https://behavior.stanford.edu/challenge/leaderboard.html) |
[Paper (2025 challenge report)](https://arxiv.org/abs/2512.06951) |
50 long-horizon household tasks on R1Pro / OmniGibson

## Status

**Integration:** ✅ Benchmark + config + Docker recipe + unit tests + zero-action model server landed.
**End-to-end run:** ✅ Real Isaac Sim simulation, real BDDL goal evaluation, real result JSON written.
**Trained-VLA reproduction:** ⬜ Pending a R1Pro-compatible VLA model server (e.g. Pi0.5 from the RLC fork).

## End-to-end Results

### Demo Replay (succeeding trajectory)

The strongest possible integration check: take a recorded human
teleoperation that the official Stanford collection labels as a
successful demonstration of `turning_on_radio` (instance 1, episode
00000010 in `behavior-1k/2025-challenge-demos`), play back the recorded
23-D action sequence through our env via a tiny replay model server,
and check whether the official BehaviorTask predicate evaluator returns
`success=True`.  If our env diverges from the recording (action
encoding, instance state, physics determinism), the replay would fail.

| Setting | Value |
|---|---|
| Task | `turning_on_radio` (B10) |
| Instance id | 1 (loaded via ported `load_task_instance`) |
| Robot | R1Pro |
| Policy | [`Behavior1KDemoReplayModelServer`](../../src/vla_eval/model_servers/behavior1k_demo_replay.py) playing back `episode_00000010.parquet` (1956 recorded steps) |
| Episodes × steps | 1 × **1364** (env terminated early on success) |
| Wall clock | 2933.8 s (~49 min including ~9 min sim+scene boot and 25-step physics settle for the TRO state load) |
| **Success rate** | **100.0%** (1 / 1, success=`true`) |

Raw JSON: [`data/behavior1k_demo_replay_turning_on_radio_inst1.json`](data/behavior1k_demo_replay_turning_on_radio_inst1.json).

A `True` from the BDDL goal-predicate evaluator on a recorded human
trajectory closes every link in the integration: scene assets load, the
TRO instance state is applied correctly, the 23-D R1Pro absolute-joint
action format reaches the `og.Environment.step` call faithfully, the
30 Hz physics is deterministic enough for replay, and the success
detector lights up.  `1364 < 1956` recorded steps means the env
terminated on the BDDL goal exactly when the human had pressed the
radio button — the rest of the recording (placing the radio back) was
not strictly required for goal satisfaction.

### Zero-action baseline (sanity floor)

A trivially-small companion run to prove the harness itself works
without any policy: the 23-D zero-action `Behavior1KBaselineModelServer`
mirrors the official `LocalPolicy(action_dim=23)` shipped in
`OmniGibson/omnigibson/learning/policies.py`.

| Setting | Value |
|---|---|
| Task | `turning_on_radio` (instance 0) |
| Policy | Zero-action 23-D vector |
| Episodes × steps | 1 × 100 |
| Wall clock | 754.1 s |
| **Success rate** | **0.0%** (0 / 1, success=`false`) |

Raw JSON: [`data/behavior1k_baseline_zero_action_turning_on_radio.json`](data/behavior1k_baseline_zero_action_turning_on_radio.json).

A 0% success rate is the expected outcome — zero joint commands keep
the robot motionless, so no BDDL goal predicate is ever satisfied.

### Trained-policy reproduction

Comparing against published results (e.g. Robot Learning Collective's
26.0% q-score, 1st place at the 2025 Challenge) is the natural next
step but requires integrating an R1Pro-compatible model server (Pi0.5
fork from the RLC submission, or the official challenge baselines).
That work is tracked in *What Trained-VLA Reproduction Still Needs*
below.

## Published Reference Scores (50-task private test set)

Q-score is the primary ranking metric: fraction of satisfied BDDL goal
predicates (with partial credit) averaged across 50 tasks.  task_sr
requires every goal predicate of a task to be satisfied.

| Rank | Team | task_sr | q_score | Source |
|------|------|:-------:|:-------:|--------|
| 1 | Robot Learning Collective | 12.4% | **26.0%** | [report](https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html), [code](https://github.com/IliaLarchenko/behavior-1k-solution) |
| 2 | Comet (NVIDIA Research) | 11.4% | 25.1% | [report](https://arxiv.org/html/2512.10071v1) |
| 3 | SimpleAI Robot | 10.8% | 15.9% | challenge leaderboard |

The official baselines (π₀.₅, OpenVLA-OFT) are provided as starting
points in [`OmniGibson/learning/`](https://github.com/StanfordVL/BEHAVIOR-1K/tree/main/OmniGibson/omnigibson/learning)
but no q_score / task_sr numbers are published for them on the private
test set.

## Integration Notes

- **Robot:** R1Pro only (the BEHAVIOR Challenge 2025 standard track).
- **Action:** 23-D absolute joint positions, layout matches
  `omnigibson.learning.utils.eval_utils.ACTION_QPOS_INDICES["R1Pro"]`:
  `base[0:3] + torso[3:7] + left_arm[7:14] + left_gripper[14:15] +
  right_arm[15:22] + right_gripper[22:23]`.
- **Cameras:** head 720×720, left_wrist 480×480, right_wrist 480×480.
  OmniGibson `VisionSensor` returns RGBA uint8 — the benchmark drops the
  alpha channel before sending the image to the model server.
- **Success:** binary `info["done"]["success"]`.  Partial-credit q_score
  scoring lives in `omnigibson.learning.utils.score_utils.compute_final_q_score`
  and is reported by the official AgentMetric/TaskMetric callbacks; the
  harness currently surfaces only the binary flag (the q_score path is a
  follow-up if needed).
- **Max steps:** 5000 default (or 2× human demo length when configured;
  see `learning/eval.py` for the dataset-driven path).

## How to Reproduce (zero-action baseline, 1 task, 100 steps)

```bash
# 1. Build the image (heavy: ~17 min, 23.5 GB).
docker/build.sh behavior1k

# 2. Download the dataset (~35 GiB).  Mount-target inside the image
#    is /app/BEHAVIOR-1K/datasets — that's where gm.DATA_PATH points.
mkdir -p /path/to/og_data
docker run --rm --gpus all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -v /path/to/og_data:/app/BEHAVIOR-1K/datasets \
  --entrypoint conda \
  ghcr.io/allenai/vla-evaluation-harness/behavior1k:latest \
  run --no-capture-output -n behavior python -c "
from omnigibson.utils.asset_utils import (
    download_omnigibson_robot_assets,
    download_behavior_1k_assets,
    download_2025_challenge_task_instances,
)
download_omnigibson_robot_assets()
download_behavior_1k_assets(accept_license=True)
download_2025_challenge_task_instances()
"

# 3. Start the zero-action baseline server.
uv run --script src/vla_eval/model_servers/behavior1k_baseline.py \
    --port 8765 --host 0.0.0.0 &

# 4. Run.  --gpus 0 pins the container to a single A100; multi-GPU
#    triggers Isaac Sim's "Multiple ICDs" instability.
uv run vla-eval run -c configs/behavior1k_eval.yaml \
    --server-url ws://127.0.0.1:8765 \
    --output-dir results/behavior1k_baseline \
    --gpus 0 --yes
```

Edit `configs/behavior1k_eval.yaml` `volumes` to point at your dataset path.

## What Trained-VLA Reproduction Still Needs

1. A R1Pro-compatible model server in `src/vla_eval/model_servers/`.
   Natural starting point: the
   [Robot Learning Collective Pi0.5 fork](https://github.com/IliaLarchenko/behavior-1k-solution)
   (1st place, 26.0% q-score) or the official π₀.₅ baseline shipped in
   `OmniGibson/omnigibson/learning/policies.py`.
2. Drop `max_steps` from `params:` (or raise to 5000) so the BehaviorTask
   has enough time to be solved.
3. Run all 50 tasks × 10 instances:
   `vla-eval run -c configs/behavior1k_eval.yaml`.
4. Score the output JSONs through
   `omnigibson.learning.utils.score_utils.compute_final_q_score`.

## Configuration

| | |
|---|---|
| **Benchmark config** | [`configs/behavior1k_eval.yaml`](../../configs/behavior1k_eval.yaml) |
| **Server config (zero-action)** | [`configs/model_servers/behavior1k/baseline.yaml`](../../configs/model_servers/behavior1k/baseline.yaml) |
| **Docker image** | `ghcr.io/allenai/vla-evaluation-harness/behavior1k:latest` (Dockerfile.behavior1k) |
| **Results** | [`data/behavior1k_baseline_zero_action_turning_on_radio.json`](data/behavior1k_baseline_zero_action_turning_on_radio.json) |

## Verification Done at Integration Time

1. Static: `make check` (ruff + ty) passes on `behavior1k/`.
2. Mocked integration: [`tests/test_behavior1k_benchmark.py`](../../tests/test_behavior1k_benchmark.py)
   injects fake `omnigibson` / `gello.robots.sim_robot` / `hydra` modules
   and runs `get_tasks → reset → step (×3) → make_obs → get_step_result`.
   **7/7 tests pass.**  Verifies (a) the v3.7.2 import paths
   (`gello.robots.sim_robot.og_teleop_utils`,
   `omnigibson.envs.env_wrapper.EnvironmentWrapper`,
   `omnigibson.learning.utils.eval_utils.{generate_basic_environment_config,flatten_obs_dict,PROPRIOCEPTION_INDICES}`),
   (b) the RGBA → RGB alpha-drop, (c) `info["done"]["success"]`
   detection, and (d) that `DISABLED_TRANSITION_RULES[*].ENABLED = False`
   is applied during reset.
3. Config validation: `vla-eval test --validate` reports **63/63 configs
   valid.**
4. **Docker image builds end-to-end** (`docker/Dockerfile.behavior1k`,
   ~17 min, 22.8 GB).  Layers: `numpy<2 setuptools<=79` → torch 2.6.0
   cu124 → isaacsim 4.5.0 + extscache → BEHAVIOR-1K v3.7.2 (bddl3,
   OmniGibson[eval], joylo) → cffi 1.17.1 force-reinstall → harness.
5. **Inside the built image, every import the benchmark depends on
   resolves**: `omnigibson.macros`, `omnigibson.envs.env_wrapper`,
   `omnigibson.learning.utils.eval_utils`,
   `gello.robots.sim_robot.og_teleop_{utils,cfg}`, `hydra.utils`,
   `omegaconf`, `torch`, `vla_eval.benchmarks.behavior1k.benchmark`.
   `TASK_NAMES_TO_INDICES` has 50 tasks; `ROBOT_CAMERA_NAMES["R1Pro"]`
   matches the hardcoded `R1PRO_CAMERAS` in the benchmark byte-for-byte;
   `DISABLED_TRANSITION_RULES` has 3 rule classes.
6. **End-to-end smoke** (`vla-eval test -c configs/behavior1k_eval.yaml`):
   **passed** in 30.4 s.  EchoModelServer starts on a free port, the
   container connects, HELLO is exchanged.  Without the dataset mounted
   the benchmark cannot finish an episode (`og.Environment(configs=cfg)`
   needs scene assets), so no per-episode result file is written, but
   the harness/Docker/protocol path is verified.

## Outstanding for Full Score Reproduction

- Mount BEHAVIOR-1K dataset (`2025-challenge-task-instances/` plus the
  per-scene OmniGibson assets) at `/data/og_data` — requires accepting
  the NVIDIA Isaac Sim EULA and the BEHAVIOR Dataset ToS.
- Integrate a R1Pro-compatible model server into the harness (no
  existing server in `configs/model_servers/` targets R1Pro 23-D
  absolute-joint actions).  Natural starting points: the official
  `OmniGibson/learning/policies.py` Pi0.5 baseline, or the
  [Robot Learning Collective Pi0.5 fork](https://github.com/IliaLarchenko/behavior-1k-solution)
  that won the 2025 challenge.
