export MEM := MemoryManagement
export NAME := Network
export CC := gcc
export CXX := g++
export VITIS_HLS_ROOT := /eda/xilinx/Vitis_HLS/2022.2/
export BOARD_PATH := \
	/home/filippo/.Xilinx/Vivado/2022.2/xhub/board_store/xilinx_board_store
export DEPS := $(NAME).hpp
export COSIM := 0
export NN2FPGA_CC_ROOT := $(shell pwd)/cc
export NN2FPGA_ROOT := $(shell pwd)

export CXXFLAGS := -I$(VITIS_HLS_ROOT)/include -I$(NN2FPGA_CC_ROOT)/include
export PRJ_ROOT := $(shell cd .. && pwd)/work
export TOP_NAME := Network
# export BOARD=ULTRA96v2
# export ONNX_PATH=./onnx/Brevonnx_resnet_final_fx.onnx

