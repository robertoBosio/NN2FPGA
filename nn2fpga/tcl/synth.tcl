set NN2FPGA_ROOT [lindex $argv 2]
source "${NN2FPGA_ROOT}/tcl/settings.tcl"

set impl_sel "solution_0"
set PRJ_NAME ${TOP_NAME}_${BOARD}

delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${TOP_NAME}

open_solution solution_1
set_part ${FPGA_PART}

add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include" \
  cc/src/memory_management.cc
add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include" \
  cc/src/${TOP_NAME}.cc
add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include" \
  -tb cc/${TOP_NAME}_tb.cc

create_clock -period 5

# config_interface -m_axi_max_widen_bitwidth 0
# config_interface -m_axi_alignment_byte_size 1

config_interface -s_axilite_auto_restart_counter 1
# config_interface -s_axilite_sw_reset
config_interface -m_axi_max_widen_bitwidth 128
config_interface -m_axi_alignment_byte_size 16
config_interface -m_axi_max_read_burst_length 256
config_interface -m_axi_num_read_outstanding 1
config_interface -m_axi_latency 1
config_compile -pipeline_style flp

# Done to avoid forced pipelining of loops that could waste resources
# config_compile -pipeline_loops 200000
# csim_design

csynth_design

export_design

if {${COSIM} == 1} {
  cosim_design
}

exit
