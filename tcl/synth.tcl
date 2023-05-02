set impl_sel "solution_0"

set BOARD $::env(BOARD)
set COSIM $::env(COSIM)
# set board "KRIA"

set TOP_NAME Network
set PRJ_NAME ${TOP_NAME}${BOARD}

delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${TOP_NAME}

open_solution solution_1
if {$BOARD == "PYNQ"} {
	set_part {xc7z020clg400-1}
}

if {$BOARD == "ULTRA96v2"} {
	set_part {xczu3eg-sbva484-1-i}
}

if {$BOARD == "KRIA"} {
	set_part {xczu5eg-sfvc784-1-i}
}

add_files src/MemoryManagement.cpp
add_files src/${TOP_NAME}.cpp
add_files -tb tb/${TOP_NAME}Tb.cpp

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

# Done to avoid forced pipelining of loops that could waste resources
# config_compile -pipeline_loops 200000
# csim_design

csynth_design

export_design

if {${COSIM} == 1} {
  cosim_design
}

exit
