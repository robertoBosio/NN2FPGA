#!/bin/bash
set -e

# Read project root from environment, with fallback
SILVIA_DIR="${NN2FPGA_ROOT_DIR}/deps/SILVIA"
PASS_LIB_PATH="${SILVIA_DIR}/build/SILVIAMuladd/LLVMSILVIAMuladd.so"
VIVADO_PATH="${XILINX_DIR}/Xilinx/Vivado/${XILINX_VERSION}"
VITIS_PATH="${XILINX_DIR}/Xilinx/Vitis/${XILINX_VERSION}"
HLS_PATH="${XILINX_DIR}/Xilinx/Vitis_HLS/${XILINX_VERSION}"

echo "Project root: $NN2FPGA_ROOT_DIR"
echo "Sourcing Xilinx tools from $XILINX_DIR"

if [ -f "$VITIS_PATH/settings64.sh" ]; then
    source "$VITIS_PATH/settings64.sh"
else
    echo "Unable to find Vitis" >&2
    exit 1
fi
if [ -f "$HLS_PATH/settings64.sh" ]; then
    source "$HLS_PATH/settings64.sh"
else
    echo "Unable to find Vitis HLS" >&2
    exit 1
fi
if [ -f "$VIVADO_PATH/settings64.sh" ]; then
    source "$VIVADO_PATH/settings64.sh"
else
    echo "Unable to find Vivado" >&2
    exit 1
fi

echo "Looking for compiled pass at: $PASS_LIB_PATH"

# Compile if pass is missing
if [ ! -f "$PASS_LIB_PATH" ]; then
    echo "LLVM pass not found — compiling..."

    cd "$SILVIA_DIR"
    chmod +x install_llvm.sh build_pass.sh

    if ! ./install_llvm.sh; then
        echo "install_llvm.sh failed." >&2
        exit 1
    fi

    if ! ./build_pass.sh; then
        echo "compile_pass.sh failed." >&2
        exit 1
    fi

    if [ ! -f "$PASS_LIB_PATH" ]; then
        echo "Compilation finished, but LLVM pass not found at $PASS_LIB_PATH" >&2
        exit 1
    fi

    cd $NN2FPGA_ROOT_DIR

    echo "LLVM pass compiled successfully."
else
    echo "LLVM pass already compiled. Skipping rebuild."
fi

# Run user-supplied command
exec "$@"
