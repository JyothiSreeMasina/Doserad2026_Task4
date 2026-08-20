#!/usr/bin/env bash
#
# do_build.sh
#
# Builds the Task 2 (photon/MR) submission image. Build context is the
# project root (not docker_task2_photon_mr/), since app.py/process.py need
# to COPY in src/ and configs/ -- see Dockerfile's own comment on this.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

DOCKER_IMAGE_TAG="doserad2026_task2_photon_mr"

docker build \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$PROJECT_ROOT"

echo "Built ${DOCKER_IMAGE_TAG}"
