#!/usr/bin/env bash
# Build Docker images locally.
# Usage:
#   docker/build.sh              # build all (base first, then benchmarks)
#   docker/build.sh libero       # build a single benchmark image
#   docker/build.sh --tag 0.1.0  # build all with a specific tag
set -euo pipefail

TAG="latest"
BASE_IMAGE=""
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)        TAG="$2"; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    -*)           echo "Unknown flag: $1"; exit 1 ;;
    *)            TARGET="$1"; shift ;;
  esac
done

BENCHMARKS=(simpler libero libero_pro libero_mem robocerebra maniskill2 calvin mikasa_robo vlabench rlbench robotwin robocasa kinetix robomme)
REGISTRY="ghcr.io/allenai/vla-evaluation-harness"

# Default BASE_IMAGE follows TAG unless explicitly overridden
BASE_IMAGE="${BASE_IMAGE:-${REGISTRY}/base:${TAG}}"

build_image() {
  local name="$1"
  local dockerfile="docker/Dockerfile.${name}"

  if [[ ! -f "$dockerfile" ]]; then
    echo "SKIP: $dockerfile not found"
    return
  fi

  local tag="${REGISTRY}/${name}:${TAG}"

  echo "Building ${tag} ..."
  docker build \
    -f "$dockerfile" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg HARNESS_VERSION="${TAG}" \
    -t "$tag" \
    .
}

if [[ -n "$TARGET" ]]; then
  if [[ "$TARGET" == "base" ]]; then
    build_image base
  else
    build_image "$TARGET"
  fi
else
  # Build base first, then all benchmarks
  build_image base
  for name in "${BENCHMARKS[@]}"; do
    build_image "$name"
  done
fi
