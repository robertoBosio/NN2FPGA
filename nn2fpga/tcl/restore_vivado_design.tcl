set NN2FPGA_ROOT [lindex $argv 0]
source "${NN2FPGA_ROOT}/tcl/settings.tcl"

set PRJ_NAME ${BOARD}_${TOP_NAME}_example

open_project ${PRJ_NAME}/${PRJ_NAME}.xpr 