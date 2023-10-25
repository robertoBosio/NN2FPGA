export MEM := memory_management
export NAME := network
export CC := gcc
export CXX := g++
export VITIS_HLS_ROOT := /eda/xilinx/Vitis_HLS/2022.2
export BOARD_PATH := \
	/home/filippo/.Xilinx/Vivado/2022.2/xhub/board_store/xilinx_board_store
export DEPS := $(NAME).h
export COSIM := 0
export CSIM := 0
export OFF_CHIP_STORAGE := 0
export URAM_STORAGE := 0
export DATASET := cifar10
export OBJECT_DETECTION := 0
export PACKING := 0
export SIMD_DSP := 0
export NN2FPGA_CC_ROOT := $(shell pwd)/cc
export NN2FPGA_ROOT := $(shell pwd)
export CXXFLAGS := -I$(VITIS_HLS_ROOT)/include -I$(NN2FPGA_CC_ROOT)/include
export PRJ_ROOT := $(shell cd .. && pwd)/work
# export TOP_NAME := network
export TEST_ROOT := $(shell cd .. && pwd)/test
export TB_ROOT := $(TEST_ROOT)/tb
export PLATFORM := xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm

