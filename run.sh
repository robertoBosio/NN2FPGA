#!/bin/bash
set -e

# ----- Configuration -----
IMAGE_NAME="nn2fpga-container-image"
CONTAINER_NAME="nn2fpga-container"
WORKSPACE_DIR="$(pwd)/.."
HISTORY_FILE="${WORKSPACE_DIR}/NN2FPGA/.docker_bash_history"
USERNAME="$(whoami)"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
XILINX_DIR="/tools"
XRT_DIR="/opt/xilinx/xrt"
XILINX_VERSION="2024.2"
DATASET_DIR="/home-ssd/datasets"
WORKSPACE_ROOT_DIR="/workspace"

# ----- Ensure bash history file exists -----
touch "${HISTORY_FILE}"

# ----- Build image if not already built -----
if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
    echo "Building Docker image '${IMAGE_NAME}'..."
    docker build \
        --file docker/Dockerfile \
        --build-arg USERNAME="${USERNAME}" \
        --build-arg USER_ID="${USER_ID}" \
        --build-arg GROUP_ID="${GROUP_ID}" \
        --build-arg NN2FPGA_ROOT_DIR="${WORKSPACE_ROOT_DIR}/NN2FPGA" \
        -t "${IMAGE_NAME}" .
fi

# ----- Run the container -----
docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    -v "${WORKSPACE_DIR}:${WORKSPACE_ROOT_DIR}" \
    -v "${HISTORY_FILE}:${WORKSPACE_ROOT_DIR}/NN2FPGA/.docker_bash_history" \
    -v "${XILINX_DIR}:${XILINX_DIR}" \
    -v "${XRT_DIR}:${XRT_DIR}" \
    -v "${DATASET_DIR}:/home/datasets" \
    --network=host \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env NN2FPGA_ROOT_DIR="${WORKSPACE_ROOT_DIR}/NN2FPGA" \
    --env XILINX_DIR="${XILINX_DIR}" \
    --env XRT_DIR="${XRT_DIR}" \
    --env XILINX_VERSION="${XILINX_VERSION}" \
    "${IMAGE_NAME}" \
    bash
