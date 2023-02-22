set impl_sel "solution_0"

set board "ULTRA"

set TOP_NAME Network
if {$board == "PYNQ"} {
	set PRJ_NAME ${TOP_NAME}PYNQ2
} else {
	set PRJ_NAME ${TOP_NAME}ULTRA
}

delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${TOP_NAME}

open_solution solution_1
if {$board == "PYNQ"} {
	set_part {xc7z020clg400-1}
} else {
	set_part {xczu3eg-sbva484-1-i}
}

add_files src/MemoryManagement.cpp
add_files src/${TOP_NAME}.cpp
add_files -tb tb/${TOP_NAME}Tb.cpp

create_clock -period 5

# config_interface -m_axi_max_widen_bitwidth 0
# config_interface -m_axi_alignment_byte_size 1

config_interface -m_axi_max_widen_bitwidth 128
config_interface -m_axi_alignment_byte_size 16
config_interface -m_axi_latency 1

# Done to avoid forced pipelining of loops that could waste resources
# config_compile -pipeline_loops 200000
# csim_design

csynth_design

export_design
