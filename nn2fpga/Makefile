# Including the settings.
include settings.mk

check-board:
ifndef BOARD
	$(error ERROR! Please, provide the target board via BOARD)
endif

ALVEO_BOARDS:= U280 U250 U55C
ifeq ($(filter $(BOARD),$(ALVEO_BOARDS)), $(BOARD))
	export VITIS_FLOW := 1
	ifndef PLATFORM
	$(error ERROR! Please, provide the .xpfm file path via PLATFORM)
	endif
else
	export VITIS_FLOW := 0
endif

check-onnx:
ifndef ONNX_PATH
	$(error ERROR! Please, provide the ONNX file path via ONNX_PATH)
endif

check-top:
ifndef TOP_NAME
	$(error ERROR! Please, provide the design top name via TOP_NAME)
endif

check-vivado:
ifndef XILINX_VIVADO
	$(error ERROR! XILINX_VIVADO variable is not set, please set correctly using "source <Vivado_install_path>/Vivado/<Version>/.settings64.sh" and rerun)
endif

check-vitis:
ifndef XILINX_VITIS
	$(error ERROR! XILINX_VITIS variable is not set, please set correctly using "source <Vitis_install_path>/Vitis/<Version>/.settings64.sh" and rerun)
endif

check-hls:
ifndef XILINX_HLS
	$(error ERROR! XILINX_HLS variable is not set, please set correctly using "source <Vitis_HLS_install_path>/Vitis_HLS/<Version>/.settings64.sh" and rerun)
endif

check-xrt:
ifndef XILINX_XRT
	$(error ERROR! XILINX_XRT variable is not set, please set correctly using "source /opt/xilinx/xrt/setup.sh" and rerun)
endif

check-target:
ifndef TARGET
	$(error ERROR! TARGET variable is not set, please set it to sw_emu, hw_emu or hw when using ALVEO boards)
endif


device2xsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))

XSA := 
ifneq ($(PLATFORM), )
XSA := $(call device2xsa, $(PLATFORM))
endif

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/*}')
PWD = $(shell readlink -f .)

TEMP_DIR := $(PRJ_ROOT)/_x.$(TARGET).$(XSA)
BUILD_DIR := $(PRJ_ROOT)/build_dir.$(TARGET).$(XSA)
LINK_OUTPUT := $(BUILD_DIR)/$(TOP_NAME).link.xclbin
PACKAGE_OUT = ./package.$(TARGET)

VPP_PFLAGS := 
VPP_LDFLAGS :=

############################## Setting up Host Variables ##############################
# Execution arguments
ifndef IMAGES
IMAGES := 1
endif

ifndef UPLOAD_WEIGHTS
UPLOAD_WEIGHTS := 1
endif
CMD_ARGS = -x $(BUILD_DIR)/$(TOP_NAME).xclbin
CMD_ARGS += -n $(IMAGES)
CMD_ARGS += -w $(UPLOAD_WEIGHTS)

#Include Required Host Source Files
CXXFLAGS := -I$(TB_ROOT)/common/cmdparser
CXXFLAGS += -I$(TB_ROOT)/common/logger
CXXFLAGS += -I$(NN2FPGA_ROOT)/cc/include 
CXXFLAGS += -I$(TB_ROOT)/$(DATASET)/include 
CXXFLAGS += -I$(PRJ_ROOT)/cc/include
CXXFLAGS += -I$(PRJ_ROOT)/cc/src
CXXFLAGS += -I$(XILINX_XRT)/include
CXXFLAGS += -I$(XILINX_VIVADO)/include
CXXFLAGS += -I$(XILINX_HLS)/include
CXXFLAGS += -Wno-attributes
CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -O0
CXXFLAGS += -g
CXXFLAGS += -std=c++17
CXXFLAGS += -fmessage-length=0

# Adding CSIM symbol to use exploit csim of vitis_hls even for vitis flow
ifeq ($(VITIS_FLOW), 0)
CXXFLAGS += -DCSIM
endif

HOST_SRCS := $(TB_ROOT)/common/cmdparser/cmdlineparser.cpp 
HOST_SRCS += $(TB_ROOT)/common/logger/logger.cpp 
HOST_SRCS += $(TB_ROOT)/$(DATASET)/network_tb.cc

# Host compiler global settings
LDFLAGS += -lrt -lstdc++ 
LDFLAGS += -luuid -lxrt_coreutil -pthread
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lOpenCL

EXECUTABLE = host.o
EMCONFIG_DIR = $(TEMP_DIR)

############################## Kernel Source Files Repository##########################
SRC_REPO := $(PRJ_ROOT)/cc/src
KERNEL_SRC_CPP := $(SRC_REPO)/$(TOP_NAME).cc
KERNEL_INCLUDES := -I$(PRJ_ROOT)/cc/include

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

.PHONY: test
test: $(EXECUTABLE) emconfig check-target check-top
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(CMD_ARGS)
else
	./$(EXECUTABLE) $(CMD_ARGS)
endif
############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += --save-temps
VPP_FLAGS += --config NN2FPGA_hls.cfg
# VPP_FLAGS += --profile_kernel data:all:all:all

VPP_LDFLAGS += --vivado.synth.jobs 4
VPP_LDFLAGS += --vivado.impl.jobs 4
VPP_LDFLAGS += --config $(PRJ_ROOT)/$(TOP_NAME)_link.cfg
# VPP_LDFLAGS += --profile_kernel data:all:all:all

$(TEMP_DIR)/$(TOP_NAME).xo: $(PRJ_ROOT)/cc/src/$(TOP_NAME).cc
	mkdir -p $(TEMP_DIR)
	v++ -g -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k $(TOP_NAME) --temp_dir $(TEMP_DIR) -I$(PRJ_ROOT)/cc/include -I$(PRJ_ROOT)/cc/src -I$(PRJ_ROOT)/../nn2fpga/cc/include/ -o'$@' '$<'

$(TEMP_DIR)/mm2s_a.xo: $(PRJ_ROOT)/cc/src/mm2s_activations.cc
	mkdir -p $(TEMP_DIR)
	v++ -g -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k mm2s_activations --temp_dir $(TEMP_DIR) -I$(PRJ_ROOT)/cc/include -I$(PRJ_ROOT)/../nn2fpga/cc/include/ -o'$@' '$<'

$(TEMP_DIR)/mm2s_w.xo: $(PRJ_ROOT)/cc/src/mm2s_weights.cc
	mkdir -p $(TEMP_DIR)
	v++ -g -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k mm2s_weights --temp_dir $(TEMP_DIR) -I$(PRJ_ROOT)/cc/include -I$(PRJ_ROOT)/../nn2fpga/cc/include/ -o'$@' '$<'

$(TEMP_DIR)/s2mm.xo: $(PRJ_ROOT)/cc/src/s2mm_outputs.cc
	mkdir -p $(TEMP_DIR)
	v++ -g -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k s2mm_outputs --temp_dir $(TEMP_DIR) -I$(PRJ_ROOT)/cc/include -I$(PRJ_ROOT)/../nn2fpga/cc/include/ -o'$@' '$<'

$(BUILD_DIR)/$(TOP_NAME).xclbin: $(TEMP_DIR)/$(TOP_NAME).xo $(TEMP_DIR)/mm2s_a.xo $(TEMP_DIR)/mm2s_w.xo $(TEMP_DIR)/s2mm.xo
	mkdir -p $(BUILD_DIR)
	v++ -g -l  $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR) -o $(LINK_OUTPUT) $(+)
	v++ -p $(LINK_OUTPUT) $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/$(TOP_NAME).xclbin

.PHONY: all
all: check_prj_root generate backend

check_prj_root:
	$(NN2FPGA_ROOT)/scripts/check_settings.sh

restore_design:
	cd $(PRJ_ROOT) && vitis_hls -f $(NN2FPGA_ROOT)/tcl/restore_design.tcl \
		$(NN2FPGA_ROOT)

restore_vivado_design:
	cd $(PRJ_ROOT) && vivado -mode tcl -source $(NN2FPGA_ROOT)/tcl/restore_vivado_design.tcl \
		-tclargs $(NN2FPGA_ROOT)

syn:
	mkdir -p $(PRJ_ROOT)/cc/src/ $(PRJ_ROOT)/cc/include/ && \
	cd $(PRJ_ROOT) && vitis_hls -f $(NN2FPGA_ROOT)/tcl/synth.tcl \
		"$(NN2FPGA_ROOT) $(CMD_ARGS)"

syn-vitis: check-vitis check-hls check-vivado check-top check-board check-target $(BUILD_DIR)/$(TOP_NAME).xclbin

vivado_flow:
	cd $(PRJ_ROOT) && vivado -mode tcl -source \
		$(NN2FPGA_ROOT)/tcl/vivado_flow.tcl -tclargs $(NN2FPGA_ROOT)

.PHONY: generate
generate: clean-hlscode check_prj_root check-top check-board check-onnx
	cd $(PRJ_ROOT) && python $(NN2FPGA_ROOT)/py/code_gen_qonnx.py

host: check-xrt check-top $(HOST_SRCS) $(PRJ_ROOT)/cc/include/$(TOP_NAME)_sim.h
	cp $(PRJ_ROOT)/cc/include/$(TOP_NAME)_sim.h $(PRJ_ROOT)/cc/include/network_sim.h
	g++ -o host.o $(HOST_SRCS) $(CXXFLAGS) $(LDFLAGS)

run_model:
	cd $(PRJ_ROOT) && python $(NN2FPGA_ROOT)/py/utils/test_model.py

download_cifar10:
	$(TB_ROOT)/cifar10/download_cifar10.sh

cosim:
	cp $(PRJ_ROOT)/cc/include/$(TOP_NAME)_sim.h $(PRJ_ROOT)/cc/include/network_sim.h
ifeq ($(VITIS_FLOW), 1)
	make build TARGET=hw_emu
else
	make syn COSIM=1
endif

csim: check-top check-hls check-board
	cp $(PRJ_ROOT)/cc/include/$(TOP_NAME)_sim.h $(PRJ_ROOT)/cc/include/network_sim.h
	make syn CSIM=1

backend: syn vivado_flow

deploy:
	./scripts/inference.sh $(BOARD)

build: $(BUILD_DIR)/$(TOP_NAME).xclbin

############################## Cleaning Rules ##############################
# Cleaning stuff
.PHONY: clean cleanall clean-hlscode

clean:
	rm -rf $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	rm -rf profile_* TempConfig system_estimate.xtxt *.rpt *.csv *.run_summary
	rm -rf src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	
clean-hlscode:
	rm -rf $(PRJ_ROOT)/cc/*

cleanall: clean clean-hlscode
	rm -f $(PRJ_ROOT)/parallel_ops.lp
	rm -f $(PRJ_ROOT)/*_cfg
	rm -f $(PRJ_ROOT)/*.log
	rm -rf $(PRJ_ROOT)/*_ip
	rm -rf $(PRJ_ROOT)/npy
	rm -rf $(PRJ_ROOT)/build_dir*
	rm -rf $(PRJ_ROOT)/package.*
	rm -rf $(PRJ_ROOT)/_x* 
	rm -f $(PRJ_ROOT)/*xclbin.run_summary 
	rm -f $(PRJ_ROOT)/qemu-memory-_* 
	rm -f $(PRJ_ROOT)/emulation 
	rm -f $(PRJ_ROOT)/_vimage 
	rm -f $(PRJ_ROOT)/pl* 
	rm -f $(PRJ_ROOT)/start_simulation.sh 
	rm -f $(PRJ_ROOT)/*.xclbin
	rm -rf _x
	rm -rf .ipcache
	rm -rf .Xil
	rm -rf package.*

