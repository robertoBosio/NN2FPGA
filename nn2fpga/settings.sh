
if [ $# -eq 0 ]; then
    echo "Usage: source settings.sh <Version>"
    return
fi

version=$1
echo "Setting up Xilinx tools for version ${version}"
source /tools/xilinx/Vitis/${version}/.settings64-Vitis.sh
source /tools/xilinx/Vitis_HLS/${version}/.settings64-Vitis_HLS.sh
source /tools/xilinx/Vivado/${version}/.settings64-Vivado.sh
source /opt/xilinx/xrt/setup.sh

# Must be run on gandalf
# sudo sshfs -o allow_other roberto@smaug:/home-ssd/datasets/ /tools/datasets/
