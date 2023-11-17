set NN2FPGA_ROOT [lindex $argv 2]
source "${NN2FPGA_ROOT}/tcl/settings.tcl"

set impl_sel "solution_0"

set PRJ_NAME ${TOP_NAME}_${BOARD}

if {${CSIM} == 1} {
  set PRJ_NAME ${PRJ_NAME}_csim
} elseif {${COSIM} == 1} {
  set PRJ_NAME ${PRJ_NAME}_cosim
} else {
  set PRJ_NAME ${PRJ_NAME}
}

open_project ${PRJ_NAME}_ip
open_solution solution_1
