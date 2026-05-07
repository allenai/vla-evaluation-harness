#!/usr/bin/env bash
# Push Docker images to ghcr.io.
# Requires: docker login ghcr.io
# Usage:
#   docker/push.sh --tag 0.1.0          # push :0.1.0 and update :latest
#   docker/push.sh --tag 0.1.0 libero   # push a single image
#   docker/push.sh --tag 0.1.0 --no-latest  # push version tag only
#   docker/push.sh                      # push :latest only (with confirmation)
set -euo pipefail

TAG="latest"
TARGET=""
REGISTRY="ghcr.io/allenai/vla-evaluation-harness"
FORCE=false
UPDATE_LATEST=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)       TAG="$2"; shift 2 ;;
    --no-latest) UPDATE_LATEST=false; shift ;;
    -y)          FORCE=true; shift ;;
    -*)          echo "Unknown flag: $1"; exit 1 ;;
    *)           TARGET="$1"; shift ;;
  esac
done

if [[ "$TAG" == "latest" && "$FORCE" != true ]]; then
  read -rp "WARNING: Pushing ':latest' without a version tag. Continue? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
  UPDATE_LATEST=false  # already pushing as latest, no need to double-tag
fi

IMAGES=(base simpler simpler_groot simpler_xvla libero libero_pro libero_plus libero_mem robocerebra maniskill2 calvin mikasa_robo vlabench rlbench robotwin robocasa kinetix robomme molmospaces behavior1k)
# Images excluded from registry pushes — build locally only.
NO_REDIST=(rlbench behavior1k)

is_no_redist() {
  local n="$1"
  for g in "${NO_REDIST[@]}"; do
    [[ "$g" == "$n" ]] && return 0
  done
  return 1
}

push_image() {
  local name="$1"
  local image_name="${name//_/-}"
  local versioned="${REGISTRY}/${image_name}:${TAG}"

  if is_no_redist "$name"; then
    echo "Refusing to push ${versioned}: image bundles proprietary-licensed"
    echo "binaries that may not be redistributed to a public registry."
    echo "(See docs/reproductions/${name}.md for the license rationale.)"
    return 0
  fi

  echo "Pushing: ${versioned}"
  if ! docker push "${versioned}"; then
    echo "ERROR: Push failed. Make sure you are logged in:"
    echo "  docker login ghcr.io"
    exit 1
  fi

  if [[ "$UPDATE_LATEST" == true ]]; then
    local latest="${REGISTRY}/${image_name}:latest"
    echo "Tagging: ${versioned} -> ${latest}"
    docker tag "${versioned}" "${latest}"
    docker push "${latest}"
  fi
}

if [[ -n "$TARGET" ]]; then
  push_image "$TARGET"
else
  for img in "${IMAGES[@]}"; do
    push_image "$img"
  done
fi

