MEM=MemoryManagement
NAME=Network
CC=gcc
XILINC=/home/filippo/xilinx/Vitis_HLS/2022.2/include
CFLAGS=-I${XILINC}
DEPS=${NAME}.hpp

generate:
	python py/code_gen_qonnx.py

compile:
	gcc -c ${CFLAGS} -Isrc/ src/${NAME}.cpp -o ${NAME}.o

