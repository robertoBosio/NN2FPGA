# Including the settings.
include settings.mk

ifndef BOARD
$(error ERROR! Please, provide the target board via BOARD)
endif

ifndef ONNX_PATH
$(error ERROR! Please, provide the ONNX file path via ONNX_PATH)
endif

.PHONY: all
all: generate backend

restore_design:
	cd $(PRJ_ROOT) && vitis_hls -f $(NN2FPGA_ROOT)/tcl/restore_design.tcl \
		$(NN2FPGA_ROOT)

syn:
	cd $(PRJ_ROOT) && vitis_hls -f $(NN2FPGA_ROOT)/tcl/synth.tcl \
		$(NN2FPGA_ROOT)

vivado_flow:
	cd $(PRJ_ROOT) && vivado -mode tcl -source \
		$(NN2FPGA_ROOT)/tcl/vivado_flow.tcl -tclargs $(NN2FPGA_ROOT)

generate:
	cd $(PRJ_ROOT) && python $(NN2FPGA_ROOT)/py/code_gen_qonnx.py

compile:
	$(CXX) -c $(CXXFLAGS) -I$(NN2FPGA_CC_ROOT)/include \
		$(NN2FPGA_CC_ROOT)/src/$(NAME).cpp -o $(NAME).o
	$(CXX) -c $(CXXFLAGS) -I$(NN2FPGA_CC_ROOT)/include \
		$(NN2FPGA_CC_ROOT)/src/$(MEM).cpp -o $(MEM).o

compile_tb:
	$(CXX) -c $(CXXFLAGS) -I$(NN2FPGA_CC_ROOT)/tb/ tb/$(NAME)Tb.cpp \
		-o $(NAME)Tb.o
	$(CXX) $(NAME)Tb.o $(NAME).o $(MEM).o -o $(NAME)Tb

sim:
	cd $(PRJ_ROOT) && chmod u+x $(NAME)Tb && ./$(NAME)Tb

compile_sim: compile compile_tb sim

all_sim: generate compile_sim

run_model:
	cd $(PRJ_ROOT) && python $(NN2FPGA_ROOT)/py/utils/test_model.py

cosim: generate
	make syn COSIM=1

backend: syn vivado_flow

.PHONY: clean
clean:
	rm -rf $(PRJ_ROOT)/$(BOARD)_example
