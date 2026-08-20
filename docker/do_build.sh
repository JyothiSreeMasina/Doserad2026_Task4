#!/usr/bin/env bash
#
# do_build.sh
#
# Builds the Task 1 (photon/CT) submission image. Build context is the
# project root (not docker/), since app.py/process.py need to COPY in
# src/ and configs/ -- see Dockerfile's own comment on this. The
# Dockerfile itself lives at the project root, not in docker/ --
# required by Grand Challenge's GitHub-linked build, which looks for
# ./Dockerfile at repo root with no configurable path.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

DOCKER_IMAGE_TAG="doserad2026_task1_photon_ct"

docker build \
  --file "${PROJECT_ROOT}/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$PROJECT_ROOT"

echo "Built ${DOCKER_IMAGE_TAG}"
