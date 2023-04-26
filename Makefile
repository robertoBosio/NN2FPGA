export MEM=MemoryManagement
export NAME=Network
export CC=gcc
export XILINC=/tools/xilinx/Vitis_HLS/2022.2/include
export BOARD_PATH=/home/filippo/.Xilinx/Vivado/2022.2/xhub/board_store/xilinx_board_store
export CFLAGS=-I${XILINC}
export DEPS=${NAME}.hpp
export BOARD=ULTRA96v2

syn:
	vitis_hls -f tcl/synth.tcl

vivado_flow:
	rm -rf tmp/${BOARD}_example
	vivado -mode tcl -source tcl/vivado_flow.tcl -tclargs

generate:
	python py/code_gen_qonnx.py

compile:
	g++ -c ${CFLAGS} -Isrc/ src/${NAME}.cpp -o ${NAME}.o
	g++ -c ${CFLAGS} -Isrc/ src/${MEM}.cpp -o ${MEM}.o

compile_tb:
	g++ -c ${CFLAGS} -Itb/ tb/${NAME}Tb.cpp -o ${NAME}Tb.o
	g++ ${NAME}Tb.o ${NAME}.o ${MEM}.o -o ${NAME}Tb

sim:
	chmod +x ${NAME}Tb
	./${NAME}Tb

compile_sim: compile compile_tb sim

all_sim: generate compile_sim

run_model:
	python py/utils/test_model.py

all: generate syn vivado_flow
