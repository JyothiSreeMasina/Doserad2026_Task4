#!/usr/bin/env bash
#
# do_save.sh
#
# Packages the Task 4 submission for upload to Grand Challenge: rebuilds
# the image, saves it as a gzipped tarball, and separately tars up
# model/ (the checkpoint) as model.tar.gz -- Grand Challenge wants the
# trained weights uploaded as a distinct "Model" from the algorithm
# container itself. See docker/do_save.sh (Task 1) for the same pattern.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DOCKER_IMAGE_TAG="doserad2026_task4_proton_mr"

echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo

build_timestamp=$(docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
  echo "Error: failed to retrieve build info for ${DOCKER_IMAGE_TAG}"
  exit 1
fi
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/${output_filename}"

echo "= STEP 2 = Saving the image"
echo "This can take a while."
docker save "$DOCKER_IMAGE_TAG" | gzip -c >"$output_path"
printf "Saved as: \033[32m%s\033[0m\n" "$output_filename"
echo "==== Done"
echo

echo "= STEP 3 = Packing the model"
output_tarball_name="${SCRIPT_DIR}/model.tar.gz"
tar -czf "$output_tarball_name" -C "${SCRIPT_DIR}/model" .
printf "Saved as: \033[32mmodel.tar.gz\033[0m\n"
echo "==== Done"
echo

printf "\033[31mIMPORTANT: upload model.tar.gz as a separate Model attached to your Algorithm.\033[0m\n"
