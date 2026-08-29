#!/usr/bin/env bash
#
# do_test_run.sh
#
# Rebuilds the Task 4 (proton/MR) submission image, boots it as an HTTP
# server implementing Grand Challenge's "invoke" API, stages
# docker_task4_proton_mr/test-data/input (patient 1ABB006's real MR plus
# the same beamlets used in Task 3's fixture -- same patient, same proton
# plan, just the MR image instead of CT -- and 1x1x1 placeholders in the
# other 9 image slots, matching the batched contract), calls /invoke once,
# and collects output into docker_task4_proton_mr/test-data/output.
#
# Run this after any change to app.py/process.py/transforms.py/the model
# checkpoint to confirm the container still behaves correctly end-to-end
# before packaging it for upload (see ./do_save.sh).
#
# NOTE: the metadata JSON's inner beam/ray/beamlet field names are an
# inferred-by-analogy hypothesis for this task specifically (no Task 4
# submission has run yet) -- see process.py's module docstring. This test
# proves the container's own logic works; it does NOT prove the field names
# match the real Grand Challenge interface's actual JSON.
#
# Pinned to a specific GPU (not --gpus all) since this dev box runs other
# unattended training jobs on other GPUs concurrently -- check `nvidia-smi`
# for a free index before running if this fails to start.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

DOCKER_IMAGE_TAG="doserad2026_task4_proton_mr"
CONTAINER_NAME="doserad2026_task4_proton_mr_test"
CONTAINER_PORT=4743  # app.py hardcodes this internally, same as Task 1/2/3
HOST_PORT=4747  # Task 1=4743, Task 2=4745, Task 3=4746 -- avoid colliding with a leftover test container
GPU_DEVICE="${GPU_DEVICE:-2}"

MODEL_DIR="${SCRIPT_DIR}/model"
INPUT_DIR="${SCRIPT_DIR}/test-data/input"
OUTPUT_DIR="${SCRIPT_DIR}/test-data/output"

HEALTH_CHECK_MAX_ATTEMPTS=40
HEALTH_CHECK_DELAY_SECONDS=3
INVOKE_TIMEOUT_SECONDS=600

log() { printf "> %s\n" "$1"; }

cleanup() {
  log "Cleanup..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "(Re)building the image"
source "${SCRIPT_DIR}/do_build.sh"

log "Verifying container labels"
api_method=$(docker inspect --format='{{index .Config.Labels "org.grand-challenge.api-method"}}' "$DOCKER_IMAGE_TAG" 2>/dev/null || echo "")
if [ "$api_method" != "invoke" ]; then
  log "ERROR: missing required LABEL org.grand-challenge.api-method=\"invoke\" in Dockerfile"
  exit 1
fi

chmod -R -f a+rX "$MODEL_DIR" "$INPUT_DIR" || true
rm -rf "${OUTPUT_DIR:?}"/* 2>/dev/null || true
mkdir -p "$OUTPUT_DIR"
# Container runs as a non-root, non-host user -- world-writable so it can
# create images/stacked-radiation-dose-map-N/ and write output.mha there.
chmod -R 777 "$OUTPUT_DIR"

log "Starting container on GPU ${GPU_DEVICE}"
docker run --detach --name "$CONTAINER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  --volume "$MODEL_DIR":/opt/ml/model:ro \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "$DOCKER_IMAGE_TAG" >/dev/null

BASE_URL="http://localhost:${HOST_PORT}"

log "Waiting for health endpoint..."
status="000"
for ((i = 1; i <= HEALTH_CHECK_MAX_ATTEMPTS; i++)); do
  status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${BASE_URL}/health" || echo "000")
  log "Health check attempt ${i}/${HEALTH_CHECK_MAX_ATTEMPTS} returned ${status}"
  [ "$status" = "200" ] && break
  sleep "$HEALTH_CHECK_DELAY_SECONDS"
done
if [ "$status" != "200" ]; then
  log "ERROR: health endpoint never returned 200"
  docker logs "$CONTAINER_NAME" 2>&1 | tail -50
  exit 1
fi

log "Calling /invoke..."
start_ts=$(date +%s.%N)
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST --max-time "$INVOKE_TIMEOUT_SECONDS" "${BASE_URL}/invoke")
end_ts=$(date +%s.%N)
elapsed=$(awk -v a="$start_ts" -v b="$end_ts" 'BEGIN{printf "%.2f", b-a}')

if [ "$status" != "201" ]; then
  log "ERROR: /invoke returned ${status} (expected 201)"
  docker logs "$CONTAINER_NAME" 2>&1 | tail -50
  exit 1
fi
log "/invoke succeeded in ${elapsed}s"

docker logs "$CONTAINER_NAME" 2>&1 | tail -20

log "Output written to ${OUTPUT_DIR}"
find "$OUTPUT_DIR" -name "output.mha" -exec du -h {} \;

log "Done. Container will be removed on exit (see cleanup)."
