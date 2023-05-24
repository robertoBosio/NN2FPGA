set NN2FPGA_ROOT [lindex $argv 2]
source "${NN2FPGA_ROOT}/tcl/settings.tcl"

set impl_sel "solution_0"

set PRJ_NAME ${TOP_NAME}${BOARD}

open_project ${PRJ_NAME}_ip
open_solution solution_1
