set NN2FPGA_CC_ROOT $::env(NN2FPGA_CC_ROOT)

set BOARD $::env(BOARD)
set COSIM $::env(COSIM)
set CSIM $::env(CSIM)
set NN2FPGA_CC_ROOT $::env(NN2FPGA_CC_ROOT)
set NN2FPGA_ROOT $::env(NN2FPGA_ROOT)
set BOARD_PATH $::env(BOARD_PATH)
set TB_ROOT $::env(TB_ROOT)
# set board "KRIA"
set DATASET $::env(DATASET)

set PRJ_ROOT $::env(PRJ_ROOT)
set TOP_NAME $::env(TOP_NAME)
set URAM_STORAGE $::env(URAM_STORAGE)
set OBJECT_DETECTION $::env(OBJECT_DETECTION)
set SIMD_DSP $::env(SIMD_DSP)

if {$BOARD == "PYNQ"} {
	set FPGA_PART "xc7z020clg400-1"
}

if {${BOARD} == "ULTRA96v2"} {
 set FPGA_PART "xczu3eg-sbva484-1-i"
 set BOARD_PART "avnet.com:ultra96v2:part0:1.2"
}

if {${BOARD} == "KRIA"} {
 set FPGA_PART "xck26-sfvc784-2LV-c"
 set BOARD_PART "xilinx.com:kv260_som:part0:1.4"
}
