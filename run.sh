#!/bin/bash
set -e

# ----- Configuration -----
IMAGE_NAME="nn2fpga-container-image"
CONTAINER_NAME="nn2fpga-container"
WORKSPACE_DIR="$(pwd)"
HISTORY_FILE="${WORKSPACE_DIR}/.docker_bash_history"
USERNAME="$(whoami)"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"

# ----- Ensure bash history file exists -----
touch "${HISTORY_FILE}"

# ----- Build image if not already built -----
if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
    echo "Building Docker image '${IMAGE_NAME}'..."
    docker build \
        --build-arg USERNAME="${USERNAME}" \
        --build-arg USER_ID="${USER_ID}" \
        --build-arg GROUP_ID="${GROUP_ID}" \
        -t "${IMAGE_NAME}" .
fi

# ----- Run the container -----
docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    -v "${WORKSPACE_DIR}:/workspace" \
    -v "${HISTORY_FILE}:/workspace/.docker_bash_history" \
    -v /tools:/tools \
    -v /opt:/opt \
    -v /home-ssd/datasets:/home-ssd/datasets \
    --network=host \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    "${IMAGE_NAME}" \
    bash
