set BOARD $::env(BOARD)
set BOARD_PATH $::env(BOARD_PATH)
set PRJ_ROOT "./tmp/"
set PRJ_NAME ${BOARD}_example

if {${BOARD} == "ULTRA96v2"} {
 set FPGA_PART "xczu3eg-sbva484-1-i"
 set BOARD_PART "avnet.com:ultra96v2:part0:1.2"
}

if {${BOARD} == "KRIA"} {
 set FPGA_PART "xck26-sfvc784-2LV-c"
 set BOARD_PART "xilinx.com:kv260_som:part0:1.4"
}

create_project ${PRJ_NAME} ${PRJ_ROOT}/${PRJ_NAME} -force -part ${FPGA_PART}

set_param board.repoPaths [list ${BOARD_PATH}]
set_property ip_repo_paths ./Network${BOARD}_ip/solution_1/impl/ip [current_project]
update_ip_catalog

source tcl/bd_design.tcl

make_wrapper -files [get_files ${PRJ_ROOT}/${PRJ_NAME}/${PRJ_NAME}.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ${PRJ_ROOT}/${PRJ_NAME}/${PRJ_NAME}.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]; #

launch_runs synth_1 -jobs 2
wait_on_run synth_1
open_run synth_1

launch_runs impl_1 -jobs 2
wait_on_run impl_1
open_run impl_1

write_bitstream -file /tmp/${BOARD}_example/design_1.bit
