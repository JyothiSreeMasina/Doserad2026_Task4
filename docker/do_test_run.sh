#!/usr/bin/env bash
#
# do_test_run.sh
#
# Rebuilds the Task 1 submission image, boots it as an HTTP server
# implementing Grand Challenge's "invoke" API, stages docker/test-data/input
# (one real patient CT + 4 real control points, plus 1x1x1 placeholders in
# the other 9 image slots -- matches the real batched contract), calls
# /invoke once, and collects output into docker/test-data/output.
#
# Run this after any change to app.py/process.py/transforms.py/the model
# checkpoint to confirm the container still behaves correctly end-to-end
# before packaging it for upload (see ./do_save.sh).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

DOCKER_IMAGE_TAG="doserad2026_task1_photon_ct"
CONTAINER_NAME="doserad2026_task1_photon_ct_test"
CONTAINER_PORT=4743

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

log "Starting container"
docker run --detach --name "$CONTAINER_NAME" \
  --gpus all \
  --volume "$MODEL_DIR":/opt/ml/model:ro \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  -p "${CONTAINER_PORT}:${CONTAINER_PORT}" \
  "$DOCKER_IMAGE_TAG" >/dev/null

BASE_URL="http://localhost:${CONTAINER_PORT}"

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
