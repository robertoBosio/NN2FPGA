# Including the settings.
include settings.mk

ifndef BOARD
$(error ERROR! Please, provide the target board via BOARD)
endif

ifndef ONNX_PATH
$(error ERROR! Please, provide the ONNX file path via ONNX_PATH)
endif

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
		$(NN2FPGA_ROOT)

vivado_flow:
	cd $(PRJ_ROOT) && vivado -mode tcl -source \
		$(NN2FPGA_ROOT)/tcl/vivado_flow.tcl -tclargs $(NN2FPGA_ROOT)

generate:
	cd $(PRJ_ROOT) && python $(NN2FPGA_ROOT)/py/code_gen_qonnx.py

compile:
	$(CXX) -c $(CXXFLAGS) -I$(PRJ_ROOT)/include \
		$(PRJ_ROOT)/src/$(NAME).cc -o $(PRJ_ROOT)/$(NAME).o
	$(CXX) -c $(CXXFLAGS) -I$(PRJ_ROOT)/include \
		$(PRJ_ROOT)/src/$(MEM).cc -o $(PRJ_ROOT)/$(MEM).o

compile_tb:
	$(CXX) -c $(CXXFLAGS) -I$(PRJ_ROOT)/include $(PRJ_ROOT)/src/$(NAME)Tb.cpp \
		-o $(PRJ_ROOT)/$(NAME)_tb.o
	$(CXX) $(PRJ_ROOT)/$(NAME)_tb.o $(PRJ_ROOT)/$(NAME).o $(PRJ_ROOT)/$(MEM).o \
		-o $(PRJ_ROOT)/$(NAME)_tb

sim:
	cd $(PRJ_ROOT) && chmod u+x $(NAME)_tb && ./$(NAME)_tb

compile_sim: compile compile_tb sim

all_sim: generate compile_sim

run_model:
	cd $(PRJ_ROOT) && python $(NN2FPGA_ROOT)/py/utils/test_model.py

download_cifar10:
	$(TB_ROOT)/cifar10/download_cifar10.sh

cosim: generate
	cp $(PRJ_ROOT)/cc/include/$(TOP_NAME)_sim.h $(PRJ_ROOT)/cc/include/network_sim.h
	make syn COSIM=1

csim: generate
	cp $(PRJ_ROOT)/cc/include/$(TOP_NAME)_sim.h $(PRJ_ROOT)/cc/include/network_sim.h
	make syn CSIM=1

backend: syn vivado_flow

deploy:
	./scripts/inference.sh ${BOARD}

.PHONY: clean
clean:
	rm -rf $(PRJ_ROOT)/$(BOARD)_example
