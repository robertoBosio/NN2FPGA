export MEM=MemoryManagement
export NAME=Network
export CC=gcc
export CXX := g++
export VITIS_HLS_ROOT := /tools/xilinx/Vitis_HLS/2022.2/include/
export BOARD_PATH=/home/filippo/.Xilinx/Vivado/2022.2/xhub/board_store/xilinx_board_store
export CFLAGS=-I$(VITIS_HLS_ROOT)
export DEPS=$(NAME).hpp
export COSIM=0
# export BOARD=ULTRA96v2
# export ONNX_PATH=./onnx/Brevonnx_resnet_final_fx.onnx

restore_design:
	vitis_hls -f tcl/restore_design.tcl

syn:
	vitis_hls -f tcl/synth.tcl

vivado_flow:
	rm -rf tmp/$(BOARD)_example
	vivado -mode tcl -source tcl/vivado_flow.tcl -tclargs

generate:
	python py/code_gen_qonnx.py

compile:
	$(CXX) -c $(CFLAGS) -Isrc/ src/$(NAME).cpp -o $(NAME).o
	$(CXX) -c $(CFLAGS) -Isrc/ src/$(MEM).cpp -o $(MEM).o

compile_tb:
	$(CXX) -c $(CFLAGS) -Itb/ tb/$(NAME)Tb.cpp -o $(NAME)Tb.o
	$(CXX) $(NAME)Tb.o $(NAME).o $(MEM).o -o $(NAME)Tb

sim:
	chmod u+x $(NAME)Tb
	./$(NAME)Tb

compile_sim: compile compile_tb sim

all_sim: generate compile_sim

run_model:
	python py/utils/test_model.py

cosim: generate syn

backend: syn vivado_flow

all: generate backend
