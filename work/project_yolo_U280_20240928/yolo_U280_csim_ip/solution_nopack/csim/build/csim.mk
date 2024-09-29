# ==============================================================
# Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
# Tool Version Limit: 2023.10
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
# 
# ==============================================================
CSIM_DESIGN = 1

__SIM_FPO__ = 1

__SIM_MATHHLS__ = 1

__SIM_FFT__ = 1

__SIM_FIR__ = 1

__SIM_DDS__ = 1

ObjDir = obj

HLS_SOURCES = ../../../../../../models/tb/imagenet/network_tb.cc ../../../../../../models/tb/coco/network_tb.cc ../../../../../../models/tb/common/logger/logger.cpp ../../../../../../models/tb/common/cmdparser/cmdlineparser.cpp ../../../../cc/src/yolo.cc

override TARGET := csim.exe

AUTOPILOT_ROOT := /tools/Xilinx/Vitis_HLS/2023.2
AUTOPILOT_MACH := lnx64
ifdef AP_GCC_M32
  AUTOPILOT_MACH := Linux_x86
  IFLAG += -m32
endif
IFLAG += -fPIC
ifndef AP_GCC_PATH
  AP_GCC_PATH := /tools/Xilinx/Vitis_HLS/2023.2/tps/lnx64/gcc-8.3.0/bin
endif
AUTOPILOT_TOOL := ${AUTOPILOT_ROOT}/${AUTOPILOT_MACH}/tools
AP_CLANG_PATH := ${AUTOPILOT_TOOL}/clang-3.9/bin
AUTOPILOT_TECH := ${AUTOPILOT_ROOT}/common/technology


IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${AUTOPILOT_ROOT}/include/ap_sysc"
IFLAG += -I "${AUTOPILOT_TECH}/generic/SystemC"
IFLAG += -I "${AUTOPILOT_TECH}/generic/SystemC/AESL_FP_comp"
IFLAG += -I "${AUTOPILOT_TECH}/generic/SystemC/AESL_comp"
IFLAG += -I "${AUTOPILOT_TOOL}/auto_cc/include"
IFLAG += -I "/usr/include/x86_64-linux-gnu"
IFLAG += -D__HLS_COSIM__

IFLAG += -D__HLS_CSIM__

IFLAG += -D__VITIS_HLS__

IFLAG += -D__SIM_FPO__

IFLAG += -D__SIM_FFT__

IFLAG += -D__SIM_FIR__

IFLAG += -D__SIM_DDS__

IFLAG += -D__DSP48E2__
IFLAG += -g
DFLAG += -D__xilinx_ip_top= -DAESL_TB
CCFLAG += -Werror=return-type
CCFLAG += -Wno-abi
TOOLCHAIN += 



include ./Makefile.rules

all: $(TARGET)



$(ObjDir)/network_tb.o: ../../../../../../models/tb/imagenet/network_tb.cc $(ObjDir)/.dir
	$(Echo) "   Compiling ../../../../../../models/tb/imagenet/network_tb.cc in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(CC) ${CCFLAG} -c -MMD -I../../../../cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/nn2fpga/cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/cmdparser -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/logger -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/work/project_yolo_U280_20240928/Vitis_Libraries/vision/L1/include/. -DCSIM -Wno-unknown-pragmas -Wno-unknown-pragmas  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/network_tb.d

$(ObjDir)/network_tb.o: ../../../../../../models/tb/coco/network_tb.cc $(ObjDir)/.dir
	$(Echo) "   Compiling ../../../../../../models/tb/coco/network_tb.cc in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(CC) ${CCFLAG} -c -MMD -I../../../../cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/nn2fpga/cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/cmdparser -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/logger -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/work/project_yolo_U280_20240928/Vitis_Libraries/vision/L1/include/. -DCSIM -Wno-unknown-pragmas -Wno-unknown-pragmas  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/network_tb.d

$(ObjDir)/logger.o: ../../../../../../models/tb/common/logger/logger.cpp $(ObjDir)/.dir
	$(Echo) "   Compiling ../../../../../../models/tb/common/logger/logger.cpp in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(CC) ${CCFLAG} -c -MMD  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/logger.d

$(ObjDir)/cmdlineparser.o: ../../../../../../models/tb/common/cmdparser/cmdlineparser.cpp $(ObjDir)/.dir
	$(Echo) "   Compiling ../../../../../../models/tb/common/cmdparser/cmdlineparser.cpp in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(CC) ${CCFLAG} -c -MMD -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/logger/.  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/cmdlineparser.d

$(ObjDir)/yolo.o: ../../../../cc/src/yolo.cc $(ObjDir)/.dir
	$(Echo) "   Compiling ../../../../cc/src/yolo.cc in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(CC) ${CCFLAG} -c -MMD -I../../../.././cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/nn2fpga/cc/include -O3  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/yolo.d
