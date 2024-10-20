
# Grep firt element of args list
set NN2FPGA_ROOT [lindex [lindex $argv 2] 0]

# Take the rest of the args list
set CMD_ARGS [lreplace [lindex $argv 2] 0 0]
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

#delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${TOP_NAME}

open_solution solution_nopack
set_part ${FPGA_PART}

if {${SIMD_DSP} == 1} {
  set simd_flag "-DSIMD_DSP"
} else {
  set simd_flag ""
}

if {${CSIM} == 1} {
  # add_files -cflags "  -g -D_GLIBCXX_DEBUG -Wall -Wextra -O2 -Icc/include -I${NN2FPGA_ROOT}/cc/include${simd_flag}" \
  #   cc/src/${TOP_NAME}.cc
  add_files -cflags "-O3 -Icc/include -I${NN2FPGA_ROOT}/cc/include${simd_flag}" \
    cc/src/${TOP_NAME}.cc
} else {
  add_files -cflags " -Icc/include -I${NN2FPGA_ROOT}/cc/include${simd_flag}" \
    cc/src/${TOP_NAME}.cc
}
add_files -cflags " -I${TB_ROOT}/common/logger/ " ${TB_ROOT}/common/cmdparser/cmdlineparser.cpp
add_files ${TB_ROOT}/common/logger/logger.cpp

if {${SIMD_DSP} == 1} {
  add_files -blackbox \
    ${NN2FPGA_ROOT}/cc/include/nn2fpga/black_box/mac/mac_simd.json
}

if {${DATASET} != "cifar10"} {
  puts ${PRJ_FULL_ROOT}/Vitis_Libraries/vision/L1/include/common/

  add_files -cflags " -DCSIM -Icc/include -I${NN2FPGA_ROOT}/cc/include -I${TB_ROOT}/common/cmdparser -I${TB_ROOT}/common/logger -I${PRJ_FULL_ROOT}/Vitis_Libraries/vision/L1/include/" \
    -tb ${TB_ROOT}/${DATASET}/network_tb.cc
} else {
  add_files -cflags " -DCSIM -Icc/include -I${NN2FPGA_ROOT}/cc/include -I${TB_ROOT}/common/cmdparser -I${TB_ROOT}/common/logger -I${TB_ROOT}/${DATASET}/include -I${PRJ_FULL_ROOT}/cc/include" \
    -tb ${TB_ROOT}/${DATASET}/network_tb.cc
}
add_files -tb ${TB_ROOT}/${DATASET}/preprocess.py
add_files -tb ${TB_ROOT}/../py/utils
add_files -tb ${PRJ_FULL_ROOT}/npy

if {${CSIM} == 1} {
  puts "CSIM: selected dataset is ${DATASET}"
  csim_design -argv ${CMD_ARGS}
  exit
}

if {${BOARD} == "PYNQ" || ${BOARD} == "ZC706"} { 
  create_clock -period 7
} else {
  create_clock -period 5
}

# config_interface -m_axi_max_widen_bitwidth 0
# config_interface -m_axi_alignment_byte_size 1

config_interface -s_axilite_auto_restart_counter 1
# config_interface -s_axilite_sw_reset
config_interface -m_axi_max_widen_bitwidth 128
config_interface -m_axi_alignment_byte_size 16
config_interface -m_axi_max_read_burst_length 256
config_interface -m_axi_num_read_outstanding 1
config_interface -m_axi_latency 1
# config_array_partition -throughput_driven auto -complete_threshold 32
# config_compile -pipeline_style flp
# MOD done to reduce LUT usage with a small performance degradation
config_compile -pipeline_style stp -enable_auto_rewind=false

#csynth_design
SILVIA::csynth_design

export_design -flow syn
#export_design

if {${COSIM} == 1} {
  cosim_design -trace_level all
}

exit
