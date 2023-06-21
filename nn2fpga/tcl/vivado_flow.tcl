set NN2FPGA_ROOT [lindex $argv 0]
source "${NN2FPGA_ROOT}/tcl/settings.tcl"

set PRJ_NAME ${BOARD}_${TOP_NAME}_example

create_project ${PRJ_NAME} ${PRJ_NAME} -force -part ${FPGA_PART}

set_param board.repoPaths [list ${BOARD_PATH}]
set_property ip_repo_paths \
  ${PRJ_ROOT}/${TOP_NAME}_${BOARD}_ip/solution_1/impl/ip [current_project]
update_ip_catalog

source ${NN2FPGA_ROOT}/tcl/bd_design.tcl 

make_wrapper -files \
  [get_files ${PRJ_ROOT}/${PRJ_NAME}/${PRJ_NAME}.srcs/sources_1/bd/design_1/design_1.bd] \
  -top
add_files -norecurse \
  ${PRJ_ROOT}/${PRJ_NAME}/${PRJ_NAME}.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

if {${BOARD} == "ULTRA96v2"} {
  set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
  # set_property strategy Flow_AreaMapLargeShiftRegToBRAM [get_runs synth_1]
}

set_property top design_1_wrapper [current_fileset]; #

launch_runs synth_1 -jobs 10
wait_on_run synth_1
open_run synth_1

launch_runs impl_1 -jobs 11
wait_on_run impl_1
open_run impl_1

write_bitstream -file ${PRJ_ROOT}/${PRJ_NAME}/design_1.bit -force
