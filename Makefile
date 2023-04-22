MEM=MemoryManagement
NAME=Network
CC=gcc
XILINC=/home/filippo/xilinx/Vitis_HLS/2022.2/include
CFLAGS=-I${XILINC}
DEPS=${NAME}.hpp
BOARD=ULTRA96v2

syn:
	vitis_hls -f tcl/synth.tcl

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

all: generate compile_sim

run_model:
	python py/utils/test_model.py
