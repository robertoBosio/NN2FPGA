set NN2FPGA_CC_ROOT $::env(NN2FPGA_CC_ROOT)

set BOARD $::env(BOARD)
set COSIM $::env(COSIM)
set CSIM $::env(CSIM)
set NN2FPGA_CC_ROOT $::env(NN2FPGA_CC_ROOT)
set NN2FPGA_ROOT $::env(NN2FPGA_ROOT)
set BOARD_PATH $::env(BOARD_PATH)
set TB_ROOT $::env(TB_ROOT)
set DATASET $::env(DATASET)
set PRJ_FULL_ROOT $::env(PRJ_FULL_ROOT)

set TOP_NAME $::env(TOP_NAME)
set URAM_STORAGE $::env(URAM_STORAGE)
set DYNAMIC_INIT $::env(DYNAMIC_INIT)
set OBJECT_DETECTION $::env(OBJECT_DETECTION)
set SIMD_DSP $::env(SIMD_DSP)
set OPENCV $::env(OPENCV)

set VIVADO_VERSION $::env(VIVADO_VERSION)

if {$BOARD == "PYNQ"} {
 set FPGA_PART "xc7z020clg400-1"
 set BOARD_PART "tul.com.tw:pynq-z2:part0:1.0"
}

if {$BOARD == "ZC706"} {
 set FPGA_PART "xc7z045ffg900-2"
 set BOARD_PART "xilinx.com:zc706:part0:1.4"
}

if {${BOARD} == "ULTRA96v2"} {
 set FPGA_PART "xczu3eg-sbva484-1-i"
 set BOARD_PART "avnet.com:ultra96v2:part0:1.2"
}

if {${BOARD} == "ZCU102"} {
 set FPGA_PART "xczu9eg-ffvb1156-2-e"
 set BOARD_PART " xilinx.com:zcu102:part0:3.4"
}

if {${BOARD} == "KRIA"} {
 set FPGA_PART "xck26-sfvc784-2LV-c"
 set BOARD_PART "xilinx.com:kv260_som:part0:1.4"
}

if {${BOARD} == "U280"} {
 set FPGA_PART "xcu280-fsvh2892-2L-e"
}

if {${BOARD} == "U250"} {
 set FPGA_PART "xcu280-fsvh2892-2L-e"
}

if {${BOARD} == "U55C"} {
 set FPGA_PART "xcu280-fsvh2892-2L-e"
}
