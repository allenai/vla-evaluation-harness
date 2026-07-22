#!/usr/bin/env bash
#SBATCH --job-name=rc365-s2b-qual
#SBATCH --partition=h100
#SBATCH --exclude=A100-Zebra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=rc365-s2b-%A_%a.out

set -euo pipefail

# Array bounds: dev 0-17, training-dev 0-89, frozen 0-449.
# CPU rendering uses the one GPU requested above for the policy.
# GPU rendering must be submitted with --gres=gpu:2. The policy then uses cuda:1.

: "${RC365_S2B_ROOT:?set RC365_S2B_ROOT to the read-only reference project}"
: "${RC365_PHASE_MANIFEST:?set RC365_PHASE_MANIFEST to the audited phase manifest}"
: "${ROBOCASA_GR00T_N15_CKPT:?set ROBOCASA_GR00T_N15_CKPT}"
: "${ROBOCASA_MODALITY_JSON:?set ROBOCASA_MODALITY_JSON}"

QUALIFICATION_RUNG="${QUALIFICATION_RUNG:-dev}"
QUALIFICATION_OUTPUT="${QUALIFICATION_OUTPUT:-results/rc365_s2b_qualification}"
RENDER_BACKEND="${VLA_EVAL_RENDER:-cpu}"
BENCHMARK_GPU_ARGS=()
BENCHMARK_IMAGE_ARGS=()
REPOSITORY_ROOT="$(realpath "$(dirname "$0")/..")"
cd "$REPOSITORY_ROOT"

if [[ -n "${VLA_EVAL_ROBOCASA_IMAGE:-}" ]]; then
  BENCHMARK_IMAGE_ARGS=(--image "$VLA_EVAL_ROBOCASA_IMAGE")
fi

if [[ "$RENDER_BACKEND" == "gpu" ]]; then
  if [[ "${SLURM_GPUS_ON_NODE:-0}" -lt 2 ]]; then
    echo "GPU rendering requires two allocated GPUs. Submit with --gres=gpu:2." >&2
    exit 2
  fi
  ALLOCATED_GPU_LIST="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
  FIRST_GPU="${ALLOCATED_GPU_LIST%%,*}"
  if [[ -z "$FIRST_GPU" ]]; then
    echo "Cannot resolve the first allocated GPU for Docker." >&2
    exit 2
  fi
  BENCHMARK_GPU_UUID="$(nvidia-smi -i "$FIRST_GPU" --query-gpu=uuid --format=csv,noheader | head -n 1)"
  BENCHMARK_GPU_ARGS=(--benchmark-gpu "$BENCHMARK_GPU_UUID")
fi

mkdir -p "$QUALIFICATION_OUTPUT"

uv run python -m vla_eval.rc365_s2b_qualification run \
  --rung "$QUALIFICATION_RUNG" \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --seed-manifest "$RC365_S2B_ROOT/config/qualification_seeds.json" \
  --reference-root "$RC365_S2B_ROOT" \
  --phase-manifest "$RC365_PHASE_MANIFEST" \
  --checkpoint "$ROBOCASA_GR00T_N15_CKPT" \
  --modality-json "$ROBOCASA_MODALITY_JSON" \
  --output-dir "$QUALIFICATION_OUTPUT" \
  --render "$RENDER_BACKEND" \
  --dev \
  "${BENCHMARK_IMAGE_ARGS[@]}" \
  "${BENCHMARK_GPU_ARGS[@]}"
