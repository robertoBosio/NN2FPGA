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

delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${TOP_NAME}

open_solution solution_1
set_part ${FPGA_PART}

add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include" \
  cc/src/memory_management_${TOP_NAME}.cc

if {${SIMD_DSP} == 1} {
  set simd_flag "-DSIMD_DSP"
} else {
  set simd_flag ""
}

add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include ${simd_flag}" \
  cc/src/${TOP_NAME}.cc

if {${SIMD_DSP} == 1} {
  add_files -blackbox \
    ${NN2FPGA_ROOT}/cc/include/nn2fpga/black_box/mac/mac_simd.json
}

if {${DATASET} == "dac2023"} {
  puts ${PRJ_ROOT}/Vitis_Libraries/vision/L1/include/common/

  add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include -I/usr/local/include/opencv4/ -I${PRJ_ROOT}/Vitis_Libraries/vision/L1/include/ -lopencv_imgproc -lopencv_core -lopencv_imgcodecs" \
    -tb ${TB_ROOT}/${DATASET}/network_tb.cc
} else {
  add_files -cflags "-Icc/include -I${NN2FPGA_ROOT}/cc/include -I${TB_ROOT}/${DATASET}/include -I${PRJ_ROOT}/cc/include" \
    -tb ${TB_ROOT}/${DATASET}/network_tb.cc
}

if {${CSIM} == 1} {
  csim_design
  exit
}

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
# config_compile -pipeline_style flp
# MOD done to reduce LUT usage with a small performance degradation
config_compile -pipeline_style stp -enable_auto_rewind=false

csynth_design

export_design

if {${COSIM} == 1} {
  cosim_design
}

exit
