#!/usr/bin/env bash
# Build Docker images locally.
# Usage:
#   docker/build.sh                                 # build all (gated images skipped without opt-in)
#   docker/build.sh libero                          # build a single benchmark image
#   docker/build.sh --tag 0.1.0                     # build all with a specific tag
#   docker/build.sh behavior1k --accept-license behavior1k
#                                                   # opt in to a gated image's licence
#   docker/build.sh --accept-license behavior1k --accept-license rlbench
#                                                   # build all + opt in to multiple gated images
set -euo pipefail

TAG="latest"
BASE_IMAGE=""
TARGET=""
ACCEPTED_LICENSES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)              TAG="$2"; shift 2 ;;
    --base-image)       BASE_IMAGE="$2"; shift 2 ;;
    --accept-license)   ACCEPTED_LICENSES+=("$2"); shift 2 ;;
    -*)                 echo "Unknown flag: $1"; exit 1 ;;
    *)                  TARGET="$1"; shift ;;
  esac
done

BENCHMARKS=(simpler libero libero_pro libero_plus libero_mem robocerebra maniskill2 calvin mikasa_robo vlabench rlbench robotwin robocasa kinetix robomme molmospaces behavior1k)

# Derived images that extend a benchmark image instead of base
DERIVED_BENCHMARKS=(simpler_groot simpler_xvla)

# Images whose Dockerfile gates the build behind an ``ARG ACCEPT_*=YES``
# build-arg.  Map: image-name → "<arg-name> <licence-url>".  Adding a new
# gated image means one line here — no CLI flag changes required.
declare -A EULA_GATED=(
  [rlbench]="ACCEPT_RLBENCH_LICENCE https://github.com/stepjam/RLBench/blob/master/LICENSE"
  [behavior1k]="ACCEPT_NVIDIA_EULA https://docs.omniverse.nvidia.com/eula/"
)

REGISTRY="ghcr.io/allenai/vla-evaluation-harness"

# Default BASE_IMAGE follows TAG unless explicitly overridden
BASE_IMAGE="${BASE_IMAGE:-${REGISTRY}/base:${TAG}}"

# Derive harness version via hatch-vcs (PEP 440 compliant)
HARNESS_VERSION="$(uvx hatch version 2>/dev/null || echo "0.0.0")"

is_license_accepted() {
  local n="$1"
  for a in "${ACCEPTED_LICENSES[@]+"${ACCEPTED_LICENSES[@]}"}"; do
    [[ "$a" == "$n" ]] && return 0
  done
  return 1
}

is_derived() {
  local n="$1"
  for d in "${DERIVED_BENCHMARKS[@]}"; do
    [[ "$d" == "$n" ]] && return 0
  done
  return 1
}

build_image() {
  local name="$1"
  local image_name="${name//_/-}"
  local dockerfile="docker/Dockerfile.${name}"
  local image_tag="${REGISTRY}/${image_name}:${TAG}"
  local build_args=()

  if is_derived "$name"; then
    # Derived images use the Dockerfile's default BASE_IMAGE (e.g. simpler:latest)
    build_args=(--build-arg "HARNESS_VERSION=${HARNESS_VERSION}")
  elif [[ "$name" != "base" ]]; then
    build_args=(--build-arg "BASE_IMAGE=${BASE_IMAGE}" --build-arg "HARNESS_VERSION=${HARNESS_VERSION}")
  fi

  if [[ -n "${EULA_GATED[$name]:-}" ]]; then
    read -r arg_name url <<< "${EULA_GATED[$name]}"
    if ! is_license_accepted "$name"; then
      echo "Skipping ${image_tag}: pass --accept-license ${name} to build it"
      echo "  See ${url}"
      return 0
    fi
    build_args+=(--build-arg "${arg_name}=YES")
  fi

  echo "========================================="
  echo "Building: ${image_tag}"
  echo "========================================="
  docker build -t "${image_tag}" -f "${dockerfile}" "${build_args[@]+"${build_args[@]}"}" .
}

if [[ -n "$TARGET" ]]; then
  if [[ "$TARGET" != "base" ]]; then
    found=false
    target_is_derived=false
    for b in "${BENCHMARKS[@]}"; do
      [[ "$b" == "$TARGET" ]] && found=true && break
    done
    for b in "${DERIVED_BENCHMARKS[@]}"; do
      [[ "$b" == "$TARGET" ]] && found=true && target_is_derived=true && break
    done
    if ! $found; then
      echo "ERROR: Unknown image '${TARGET}'. Available: base ${BENCHMARKS[*]} ${DERIVED_BENCHMARKS[*]}"
      exit 1
    fi
    build_image base
    if $target_is_derived; then
      parent="${TARGET%%_*}"
      build_image "$parent"
    fi
  fi
  build_image "$TARGET"
else
  build_image base
  for b in "${BENCHMARKS[@]}"; do
    build_image "$b"
  done
  for b in "${DERIVED_BENCHMARKS[@]}"; do
    build_image "$b"
  done
fi

