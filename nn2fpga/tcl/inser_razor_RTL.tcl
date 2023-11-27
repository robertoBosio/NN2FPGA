#############################################################
# Insert Razor circuit
#############################################################
# Disconnect BRAM write port
# Connect write port to clk of clk_wiz_0
# Save pins with slack < perc
# create multiplexer
# For each endpoint create a shadow flip flop
# connect clk1 of shadow ff to clk1 of clk_wiz that is delayed of T/2
# create xor2 -> even/odd algo
# create nor 


# connect to clk1 
#produce_stream_ap_fixed_vector_ap_fixed_1ul_1_16_32_32_1_1_1_1_U0
#produce_stream_ap_fixed_vector_ap_fixed_1ul_3_16_32_32_3_3_1_1_U0
#shift_op_U0

# Constraints 
set_property CLOCK_DOMAINS INDEPENDENT [get_cells s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/mem_reg_bram_0]
set_property CLOCK_DOMAINS INDEPENDENT [get_cells s_net_produce_043_U/U_two_layers_fifo_w9_d3_B_ram/mem_reg_bram_0]

# Create the two clocks 
create_clock -period 5.00 -name ap_clk -waveform {0.000 2.500} [get_ports ap_clk]
create_port -direction IN clk1
create_port -direction IN clk2
create_clock -period 5.000 -name clk1 -waveform {0.000 2.500} [get_ports clk1]
create_cell -reference IBUF clk1_IBUF_inst
create_net clk1
connect_net -net clk1 -objects [get_ports clk1]
connect_net -net clk1 -objects [get_pins clk1_IBUF_inst/I]
create_net clk1_IBUF
connect_net -net clk1_IBUF -objects [get_pins clk1_IBUF_inst/O]
create_cell -reference BUFGCE clk1_IBUF_BUFG_inst
connect_net -net clk1_IBUF -objects [get_pins clk1_IBUF_BUFG_inst/I]
create_net clk1_IBUF_BUFG
connect_net -net VCC_2 -objects [get_pins clk1_IBUF_BUFG_inst/CE]
connect_net -net clk1_IBUF_BUFG -objects [get_pins clk1_IBUF_BUFG_inst/O]

create_clock -period 5.000 -name clk2 -waveform {2.500 5.000} [get_ports clk2]
create_cell -reference IBUF clk2_IBUF_inst
create_net clk2
connect_net -net clk2 -objects [get_ports clk2]
connect_net -net clk2 -objects [get_pins clk2_IBUF_inst/I]
create_net clk2_IBUF
connect_net -net clk2_IBUF -objects [get_pins clk2_IBUF_inst/O]
create_cell -reference BUFGCE clk2_IBUF_BUFG_inst
connect_net -net clk2_IBUF -objects [get_pins clk2_IBUF_BUFG_inst/I]
create_net clk2_IBUF_BUFG
connect_net -net VCC_2 -objects [get_pins clk2_IBUF_BUFG_inst/CE]
connect_net -net clk2_IBUF_BUFG -objects [get_pins clk2_IBUF_BUFG_inst/O]

# Resets 
create_port -direction IN clk1_rst_n

create_net clk1_rst_n 
connect_net -net clk1_rst_n -objects [get_ports clk1_rst_n]
create_cell -reference IBUF clk1_rst_n_IBUF_inst
connect_net -net clk1_rst_n -objects [get_pins clk1_rst_n_IBUF_inst/I]
create_net clk1_rst_n_IBUF
connect_net -net clk1_rst_n_IBUF -objects [get_pins clk1_rst_n_IBUF_inst/O]

create_port -direction IN clk2_rst_n

create_net clk2_rst_n 
connect_net -net clk2_rst_n -objects [get_ports clk2_rst_n]
create_cell -reference IBUF clk2_rst_n_IBUF_inst
connect_net -net clk2_rst_n -objects [get_pins clk2_rst_n_IBUF_inst/I]
create_net clk2_rst_n_IBUF
connect_net -net clk2_rst_n_IBUF -objects [get_pins clk2_rst_n_IBUF_inst/O]

# Time Constraints
set_multicycle_path -setup -from [get_clocks ap_clk] -to [get_clocks clk1] 2
set_multicycle_path -hold -from [get_clocks ap_clk] -to [get_clocks clk1] 1
set_multicycle_path -setup -from [get_clocks clk1] -to [get_clocks ap_clk] 2
set_multicycle_path -hold -from [get_clocks clk1] -to [get_clocks ap_clk] 1
set_multicycle_path 2 -setup -from [ get_clocks clk1 ] -to [ get_clocks clk2]
set_multicycle_path 1 -hold -from [ get_clocks clk1 ] -to [ get_clocks clk2]
set_multicycle_path -setup -from [get_clocks ap_clk] -to [get_clocks clk2] 2
set_multicycle_path -hold -from [get_clocks ap_clk] -to [get_clocks clk2] 1
#set_false_path -from [get_clocks clk1] -to [get_clocks ap_clk]
#set_false_path -from [get_clocks ap_clk] -to [get_clocks clk1]
#set_false_path -from [get_clocks ap_clk] -to [get_clocks clk2]

# RAM clocks
set pin1 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "s_net_conv_145_U*"}] -filter {REF_PIN_NAME == CLKBWRCLK}]
set pin2 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "s_net_produce_043_U*"}] -filter {REF_PIN_NAME == CLKARDCLK}]
set net1 [get_nets -of_objects ${pin1}]
set net2 [get_nets -of_objects ${pin2}]
disconnect_net -net ${net1} -objects ${pin1}
disconnect_net -net ${net2} -objects ${pin2}

# Connect to FIFOs clkou1 and clk_wiz_0/s_axi_aresetn
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin1}
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin2}

# Synchronize reset of block s_net_produce_0_compute44
create_cell -reference FDRE syn_FF1
create_cell -reference FDRE syn_FF2
connect_net -hier -net ap_rst_n_inv -objects [get_pins syn_FF1/D]
connect_net -hier -net clk1_IBUF_BUFG -objects [get_pins syn_FF1/C]
connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins syn_FF1/R]
connect_net -hier -net s_net_produce_0_compute44_U/<const1> -objects [get_pins syn_FF1/CE]
create_net syn_FF1_out
connect_net -hier -net syn_FF1_out -objects [get_pins syn_FF1/Q]
connect_net -hier -net syn_FF1_out -objects [get_pins syn_FF2/D]
connect_net -hier -net clk1_IBUF_BUFG -objects [get_pins syn_FF2/C]
connect_net -hier -net s_net_produce_0_compute44_U/<const1> -objects [get_pins syn_FF2/CE]
connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins syn_FF2/R]
create_net ap_rst_n_inv_clk1
connect_net -hier -net ap_rst_n_inv_clk1 -objects [get_pins syn_FF2/Q]

set pin_stream1 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "produce_stream_ap_fixed_vector_ap_fixed_1ul_1_16_32_32_1_1_1_1_U0"}] -filter {REF_PIN_NAME =~ "ap_clk_IBUF_BUFG"}] 
set pin_stream1_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "produce_stream_ap_fixed_vector_ap_fixed_1ul_1_16_32_32_1_1_1_1_U0"}] -filter {REF_PIN_NAME =~ "ap_rst_n_IBUF"}]
set pin_stream2 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "produce_stream_ap_fixed_vector_ap_fixed_1ul_3_16_32_32_3_3_1_1_U0"}] -filter {REF_PIN_NAME =~ "ap_clk_IBUF_BUFG"}] 
set pin_stream2_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "produce_stream_ap_fixed_vector_ap_fixed_1ul_3_16_32_32_3_3_1_1_U0"}] -filter {REF_PIN_NAME =~ "ap_rst_n_IBUF"}]
set pin_stream1_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "produce_stream_ap_fixed_vector_ap_fixed_1ul_1_16_32_32_1_1_1_1_U0"}] -filter {REF_PIN_NAME =~ "ap_rst_n_inv"}]
set pin_stream2_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "produce_stream_ap_fixed_vector_ap_fixed_1ul_3_16_32_32_3_3_1_1_U0"}] -filter {REF_PIN_NAME =~ "ap_rst_n_inv"}]
disconnect_net -net [get_nets -of_objects $pin_stream1] -objects ${pin_stream1}
disconnect_net -net [get_nets -of_objects $pin_stream1_rst] -objects ${pin_stream1_rst}
disconnect_net -net [get_nets -of_objects $pin_stream2] -objects ${pin_stream2}
disconnect_net -net [get_nets -of_objects $pin_stream2_rst] -objects ${pin_stream2_rst}
disconnect_net -net [get_nets -of_objects $pin_stream1_invrst] -objects ${pin_stream1_invrst}
disconnect_net -net [get_nets -of_objects $pin_stream2_invrst] -objects ${pin_stream2_invrst}
# Connect to streams clkou1 and clk_wiz_0/s_axi_aresetn
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin_stream1}
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin_stream2}
connect_net -hier -net clk1_rst_n_IBUF -objects ${pin_stream1_rst}
connect_net -hier -net clk1_rst_n_IBUF -objects ${pin_stream2_rst}
connect_net -hier -net ap_rst_n_inv_clk1 -objects ${pin_stream1_invrst}
connect_net -hier -net ap_rst_n_inv_clk1 -objects ${pin_stream2_invrst}

# Produce compute 44
set pin_produce_44 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_compute44_U"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}] 
set pin_produce_44_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_compute44_U"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}] 
disconnect_net -net [get_nets -of_objects $pin_produce_44] -objects ${pin_produce_44}
disconnect_net -net [get_nets -of_objects $pin_produce_44_rst] -objects ${pin_produce_44_rst}
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin_produce_44}
connect_net -hier -net clk1_rst_n_IBUF -objects ${pin_produce_44_rst}

disconnect_net -net [get_nets -of_objects [get_pins s_net_produce_0_compute44_U/U_two_layers_fifo_w73_d2_B_ram/ap_CS_fsm[2]_i_1__17/I0]] -objects [get_pins s_net_produce_0_compute44_U/U_two_layers_fifo_w73_d2_B_ram/ap_CS_fsm[2]_i_1__17/I0]
connect_net -hier -net ap_rst_n_IBUF -objects [get_pins s_net_produce_0_compute44_U/U_two_layers_fifo_w73_d2_B_ram/ap_CS_fsm[2]_i_1__17/I0]

set pins_shift [get_pins -of_objects [get_cells -hier -filter {NAME =~ "shift_op_*"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}]
foreach {pin} $pins_shift {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net clk1_IBUF_BUFG -objects $pin
}

set pins_shift_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "shift_op_*"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}]
foreach {pin} $pins_shift_rst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net clk1_rst_n_IBUF -objects $pin
}

set pins_shift_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "shift_op_*"}] -filter {REF_PIN_NAME == ap_rst_n_inv}]
foreach {pin} $pins_shift_invrst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_inv_clk1 -objects $pin
}

# RAM in output
# WR - clk1 (wizard)
# R  - pl_clk
set pin_conv [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_conv_145_U"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}] 
set pin_conv_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_conv_145_U"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}] 
set pin_conv_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_conv_145_U"}] -filter {REF_PIN_NAME == ap_rst_n_inv}] 
disconnect_net -net [get_nets -of_objects $pin_conv] -objects ${pin_conv}
disconnect_net -net [get_nets -of_objects $pin_conv_rst] -objects ${pin_conv_rst}
disconnect_net -net [get_nets -of_objects $pin_conv_invrst] -objects ${pin_conv_invrst}
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin_conv}
connect_net -hier -net clk1_rst_n_IBUF -objects ${pin_conv_rst}
connect_net -hier -net ap_rst_n_inv_clk1 -objects ${pin_conv_invrst}
set RAM_rst_1 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "s_net_conv_145_U*"}] -filter {REF_PIN_NAME == RSTRAMARSTRAM}]
disconnect_net -net [get_nets -of_objects ${RAM_rst_1}] -objects ${RAM_rst_1}
connect_net -hier -net clk1_rst_n_IBUF -objects ${RAM_rst_1}

# RAM in input
# WR - pl_clk
# R  - clk1 (wizard)
set pin_produce_043 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_043_U"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}] 
set pin_produce_043_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_043_U"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}] 
set pin_produce_043_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_043_U"}] -filter {REF_PIN_NAME == ap_rst_n_inv}] 
disconnect_net -net [get_nets -of_objects $pin_produce_043] -objects ${pin_produce_043}
disconnect_net -net [get_nets -of_objects $pin_produce_043_rst] -objects ${pin_produce_043_rst}
disconnect_net -net [get_nets -of_objects $pin_produce_043_invrst] -objects ${pin_produce_043_invrst}
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin_produce_043}
connect_net -hier -net clk1_rst_n_IBUF -objects ${pin_produce_043_rst}
connect_net -hier -net ap_rst_n_inv_clk1 -objects ${pin_produce_043_invrst}
set RAM_rst_2 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "s_net_produce_043_U*"}] -filter {REF_PIN_NAME == RSTRAMARSTRAM}]
disconnect_net -net [get_nets -of_objects ${RAM_rst_2}] -objects ${RAM_rst_2}
connect_net -hier -net clk1_rst_n_IBUF -objects ${RAM_rst_2}

# output RAM clock 
set pin3 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "s_net_conv_145_U*"}] -filter {REF_PIN_NAME == CLKARDCLK}]
set net3 [get_nets -of_objects ${pin3}]
disconnect_net -net ap_clk_IBUF_BUFG -objects ${pin3}
#connect_net -hier -net ap_clk_IBUF_BUFG -objects ${pin3}
#create_cell -reference FDRE s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF
#connect_net -hier -net ${net3} -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF/D]
#connect_net -hier -net ap_clk_IBUF_BUFG -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF/C]
#connect_net -hier -net s_net_conv_145_U/<const1> -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF/CE]
#connect_net -hier -net s_net_conv_145_U/<const0> -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF/R]
#create_net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk
#connect_net -hier -net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF/Q]
#connect_net -hier -net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk -objects ${pin3}
#create_cell -reference FDRE s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF2
#connect_net -hier -net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF2/D]
#connect_net -hier -net ap_clk_IBUF_BUFG -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF2/C]
#connect_net -hier -net s_net_conv_145_U/<const1> -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF2/CE]
#connect_net -hier -net s_net_conv_145_U/<const0> -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF2/R]
#create_net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk2
#connect_net -hier -net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk2 -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk_FF2/Q]
#connect_net -hier -net s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/synch_clk2 -objects ${pin3}

set pin4 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "s_net_produce_043_U*"}] -filter {REF_PIN_NAME == CLKBWRCLK}]
set net4 [get_nets -of_objects ${pin4}]
disconnect_net -net ${net4} -objects ${pin4}
connect_net -hier -net ap_clk_IBUF_BUFG -objects ${pin4}

set pin_pad [get_pins -of_objects [get_cells -hier -filter {NAME =~ "pad_input_U0"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}] 
set pin_pad_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "pad_input_U0"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}] 
set pin_pad_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "pad_input_U0"}] -filter {REF_PIN_NAME == ap_rst_n_inv}] 
disconnect_net -net [get_nets -of_objects $pin_pad] -objects ${pin_pad}
disconnect_net -net [get_nets -of_objects $pin_pad_rst] -objects ${pin_pad_rst}
disconnect_net -net [get_nets -of_objects $pin_pad_invrst] -objects ${pin_pad_invrst}
connect_net -hier -net clk1_IBUF_BUFG -objects ${pin_pad}
connect_net -hier -net clk1_rst_n_IBUF -objects ${pin_pad_rst}
connect_net -hier -net ap_rst_n_inv_clk1 -objects ${pin_pad_invrst}

set pins_prepad [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_pre_pad_*"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}]
foreach {pin} $pins_prepad {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net clk1_IBUF_BUFG -objects $pin
}

set pins_prepad_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_pre_pad_*"}] -filter {REF_PIN_NAME == ap_rst_n_inv}]
foreach {pin} $pins_prepad_invrst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_inv_clk1 -objects $pin
}

set pins_data [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_data_*"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}]
foreach {pin} $pins_data {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net clk1_IBUF_BUFG -objects $pin
}

set pins_data_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_data_*"}] -filter {REF_PIN_NAME == ap_rst_n_inv}]
foreach {pin} $pins_data_invrst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_inv_clk1 -objects $pin
}

set pins_const [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_const*"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}]
foreach {pin} $pins_const {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net clk1_IBUF_BUFG -objects $pin
}
set pins_const_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_const*"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}]
foreach {pin} $pins_const_rst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net clk1_rst_n_IBUF -objects $pin
}

set pins_const_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_const*"}] -filter {REF_PIN_NAME == ap_rst_n_inv}]
foreach {pin} $pins_const_invrst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_inv_clk1 -objects $pin
}

# Connected to conv1

set pins_const_642 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_const_642_U"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}]
foreach {pin} $pins_const_642 {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_clk_IBUF_BUFG -objects $pin
}
set pins_const_642_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_const_642_U"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}]
foreach {pin} $pins_const_642_rst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_IBUF -objects $pin
}

set pins_const_642_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_const_642_U"}] -filter {REF_PIN_NAME == ap_rst_n_inv}]
foreach {pin} $pins_const_642_invrst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_inv -objects $pin
}

set pins_shift_9 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "shift_op_9_U0"}] -filter {REF_PIN_NAME == ap_clk_IBUF_BUFG}]
foreach {pin} $pins_shift_9 {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_clk_IBUF_BUFG -objects $pin
}

set pins_shift_9_rst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "shift_op_9_U0"}] -filter {REF_PIN_NAME == ap_rst_n_IBUF}]
foreach {pin} $pins_shift_9_rst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_IBUF -objects $pin
}

set pins_shift_9_invrst [get_pins -of_objects [get_cells -hier -filter {NAME =~ "shift_op_9_U0"}] -filter {REF_PIN_NAME == ap_rst_n_inv}]
foreach {pin} $pins_shift_9_invrst {
    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
    connect_net -hier -net ap_rst_n_inv -objects $pin
}

##################################################################################################################################################
#set pins_C_44 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_compute44_U*" && REF_NAME == FDRE}] -filter {REF_PIN_NAME == C}]
#foreach {pin} $pins_C_44 {
#    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
#    connect_net -hier -net clk1_IBUF_BUFG -objects $pin
#}
#
#set pins_R_44 [get_pins -of_objects [get_cells -hier -filter {NAME =~ "s_net_produce_0_compute44_U*" && REF_NAME == FDRE }] -filter {REF_PIN_NAME == R}]
#foreach {pin} $pins_R_44 {
#    disconnect_net -net [get_nets -of_objects $pin] -objects $pin
#    connect_net -hier -net clk1_rst_n_IBUF -objects $pin
#}
#
##################################################################################################################################################
# Enable read with right reset 
disconnect_net -net [get_nets -of_objects [get_pins s_net_produce_043_U/U_two_layers_fifo_w9_d3_B_ram/mem_reg_bram_0_i_1/I1]] -objects [get_pins s_net_produce_043_U/U_two_layers_fifo_w9_d3_B_ram/mem_reg_bram_0_i_1/I1]
connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins s_net_produce_043_U/U_two_layers_fifo_w9_d3_B_ram/mem_reg_bram_0_i_1/I1]

disconnect_net -net [get_nets -of_objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/mem_reg_bram_0/ENBWREN]] -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/mem_reg_bram_0/ENBWREN]
connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/mem_reg_bram_0/ENBWREN]

##################################################################################################################################################

# Change clk of the conv0 block
disconnect_net -net ap_clk_IBUF_BUFG -objects conv_comp_U0/ap_clk_IBUF_BUFG
connect_net -hier -net clk1_IBUF_BUFG -objects conv_comp_U0/ap_clk_IBUF_BUFG
disconnect_net -net ap_rst_n_IBUF -objects conv_comp_U0/ap_rst_n_IBUF
connect_net -hier -net clk1_rst_n_IBUF -objects conv_comp_U0/ap_rst_n_IBUF
disconnect_net -net ap_rst_n_inv -objects conv_comp_U0/ap_rst_n_inv
connect_net -hier -net ap_rst_n_inv_clk1 -objects conv_comp_U0/ap_rst_n_inv

# Searching for endpoints with a lesser slack than a percentage
# Intra-clock on conv0
set period [get_property PERIOD [get_clocks clk1]]
#set period [get_property PERIOD [get_clocks ap_clk_IBUF_BUFG]]
set perc 0.7
set perc_period [expr ${period}*${perc}]

set endpoints [get_pins -of_objects [get_cells -hier -filter {NAME =~ "conv_comp_U0*" && NAME !~ "*DSP48*"  && NAME !~ "*mul_8s_8s_16_1_1_U91" && NAME !~ "*s_acc_buff_V_0_U*" && IS_SEQUENTIAL }] -filter "DIRECTION == IN && SETUP_SLACK < ${perc_period} && REF_PIN_NAME == D"]

set i $0

foreach endpoint ${endpoints} {
    
    #set cell_${i} [get_cells -of_objects $endpoints]
    set cell_${i} [get_cells -of_objects $endpoint]
    set cell_tmp [set cell_${i}]
    set CE_net [get_nets -of_objects [get_pins -of_objects ${cell_tmp} -filter {REF_PIN_NAME == CE}]]
    set R_net [get_nets -of_objects [get_pins -of_objects ${cell_tmp} -filter {REF_PIN_NAME == R}]]
    #set input_net_${i} [get_nets -of $endpoints]
    set input_net_${i} [get_nets -of $endpoint]
    set input_net_tmp [set input_net_${i}]
    create_cell -reference MUXF7 conv_comp_U0/mux_${i}
    set mux_tmp [get_cells -quiet conv_comp_U0/mux_${i}]
    disconnect_net -net ${input_net_tmp} -objects [get_pins -of  ${cell_tmp} -filter {REF_PIN_NAME==D}]
    connect_net -net ${input_net_tmp} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==I0}]
    create_net conv_comp_U0/mux_out_${i}
    # MUX/O -> MAINFF/D

    connect_net -net conv_comp_U0/mux_out_${i} -objects [get_pins -of  ${mux_tmp} -filter {REF_PIN_NAME==O}]
    connect_net -net conv_comp_U0/mux_out_${i} -objects [get_pins -of  ${cell_tmp} -filter {REF_PIN_NAME==D}]
    
    # Create SHADOW FF
    create_cell -reference FDRE conv_comp_U0/shadowFF_${i}
    set shadow_tmp [get_cells -quiet conv_comp_U0/shadowFF_${i}]
    # Connect nets shadow FF
    # MUX/O -> SHADOWFF/D
    connect_net -hier -net conv_comp_U0/mux_out_${i}  -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==D}]
    connect_net -hier -net clk2_IBUF_BUFG -objects [get_pins -of  ${shadow_tmp} -filter {REF_PIN_NAME==C}]
    connect_net -hier -net ${CE_net} -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==CE}]
    connect_net -hier -net clk2_rst_n_IBUF -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==R}]
    create_net conv_comp_U0/shadow_value_${i}
    connect_net -hier -net conv_comp_U0/shadow_value_${i} -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==Q}]
    # SHADOWFF/O -> MUX/I1
    connect_net -hier -net conv_comp_U0/shadow_value_${i} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==I1}]
    
    # Output MAINFF
    set output_net_${i} [get_nets -of [get_pins -of  ${cell_tmp} -filter {REF_PIN_NAME==Q}]]
    set output_net_tmp [set output_net_${i}]

    # Creating the xor gate
    create_cell -reference xor2 conv_comp_U0/xor_${i}
    set xor_err_tmp [get_cells -quiet conv_comp_U0/xor_${i}]
    connect_net -hier -net ${output_net_tmp} -objects [get_pins -of ${xor_err_tmp} -filter {REF_PIN_NAME==I0}]
    connect_net -hier -net conv_comp_U0/shadow_value_${i} -objects [get_pins -of ${xor_err_tmp} -filter {REF_PIN_NAME==I1}]
    # XOR OUT
    create_net conv_comp_U0/razor_error_${i}
    connect_net -hier -net conv_comp_U0/razor_error_${i} -objects [get_pins -of  ${xor_err_tmp} -filter {REF_PIN_NAME==O}]
    # Connect XOR out to select signal of MUX
    connect_net -hier -net conv_comp_U0/razor_error_${i} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==S}]

    incr i
}

set i [expr {$i-1}]

# ShadowFF are more than one
if "$i > 1" {

        # if the XORs are 4 or less -> create one between or4, or3, or2
        # Output of the single OR -> error net
        if "$i < 5" {
            create_cell -reference or${i} conv_comp_U0/or
            set or_err [get_cells -quiet conv_comp_U0/or]

            while "$i > 0" {
                set pin [expr {$i-1}]
                connect_net -hier -net conv_comp_U0/razor_error_${i} -objects [get_pins conv_comp_U0/or/I${pin}]
                set i [expr {$i-1}]

                if {$i == 0} break
            }

            create_net conv_comp_U0/error 
            connect_net -hier -net conv_comp_U0/error -objects [get_pins conv_comp_U0/or/O]
            create_cell -reference FDRE conv_comp_U0/sample_error
            connect_net -hier -net conv_comp_U0/error -objects [get_pins conv_comp_U0/sample_error/D]
            connect_net -hier -net clk1_IBUF_BUFG -objects [get_pins conv_comp_U0/sample_error/C]
            connect_net -hier -net conv_comp_U0/<const1> -objects [get_pins conv_comp_U0/sample_error/CE]
            connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins conv_comp_U0/sample_error/R]
            create_net conv_comp_U0/error_s
            connect_net -hier -net conv_comp_U0/error_s -objects [get_pins conv_comp_U0/sample_error/Q]

            set flag 1
            set error 1
        # if XORs are more than 4 -> create or2s
        } else {
            set error 1
            set flag 0
            # i = 9
            # create or2 with xor_9 and xor_8
            # i-2 = 7
            # create or2 with xor_7 and xor_6
            # i-2 = 5
            # create or2 with xor_5 and xor_4
            # i-2 = 3
            # create or2 with xor_3 and xor_2
            # i-2 = 1
            # 
            set n $0
            set i_int $i
            # if odd = 1 -> n OR2 + 1 XOR2
            set odd [expr $i%2] 

            while "$i_int > 0" {
                
                if {$i_int == 0} break

                if {$i_int == 1} break

                create_cell -reference or2 conv_comp_U0/or$n
                
                set pin2 [expr {$i_int -1}]
                connect_net -hier -net conv_comp_U0/razor_error_${i_int} -objects [get_pins conv_comp_U0/or$n/I0]
                connect_net -hier -net conv_comp_U0/razor_error_${pin2} -objects [get_pins conv_comp_U0/or$n/I1]
                create_net conv_comp_U0/or_net${n}
                connect_net -hier -net conv_comp_U0/or_net${n} -objects [get_pins conv_comp_U0/or$n/O]
                incr n 
                incr i_int -2
            }
        }

        if "$flag == 0" {
            incr n -1

            create_cell -reference or${n} conv_comp_U0/or_l2
            set or_err_l2 [get_cells -quiet conv_comp_U0/or_l2]

            while "$n > 0" {
                set pin [expr {$n-1}]
                connect_net -hier -net conv_comp_U0/or_net${n} -objects [get_pins conv_comp_U0/or_l2/I${pin}]
                set n [expr {$n-1}]

                if {$n == 0} break
            }

            create_net conv_comp_U0/error_l2 
            connect_net -hier -net conv_comp_U0/error_l2 -objects [get_pins conv_comp_U0/or_l2/O]
            
            if "$odd == 0" {
                create_cell -reference FDRE conv_comp_U0/sample_error
                connect_net -hier -net conv_comp_U0/error_l2 -objects [get_pins conv_comp_U0/sample_error/D]
                connect_net -hier -net clk1_IBUF_BUFG -objects [get_pins conv_comp_U0/sample_error/C]
                connect_net -hier -net conv_comp_U0/<const1> -objects [get_pins conv_comp_U0/sample_error/CE]
                connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins conv_comp_U0/sample_error/R]
                create_net conv_comp_U0/error_s
                connect_net -hier -net conv_comp_U0/error_s -objects [get_pins conv_comp_U0/sample_error/Q]
            } else {
                create_cell -reference or2 conv_comp_U0/or_final 
                connect_net -hier -net conv_comp_U0/error_l2 -objects [get_pins conv_comp_U0/or_final/I0]
                connect_net -hier -net conv_comp_U0/razor_error_1 -objects [get_pins conv_comp_U0/or_final/I1]
                create_net conv_comp_U0/error
                connect_net -hier -net conv_comp_U0/error -objects [get_pins conv_comp_U0/or_final/O]
                create_cell -reference FDRE conv_comp_U0/sample_error
                connect_net -hier -net conv_comp_U0/error -objects [get_pins conv_comp_U0/sample_error/D]
                connect_net -hier -net clk1_IBUF_BUFG -objects [get_pins conv_comp_U0/sample_error/C]
                connect_net -hier -net conv_comp_U0/<const1> -objects [get_pins conv_comp_U0/sample_error/CE]
                connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins conv_comp_U0/sample_error/R]
                create_net conv_comp_U0/error_s
                connect_net -hier -net conv_comp_U0/error_s -objects [get_pins conv_comp_U0/sample_error/Q]
            }
        }

} else {
    # Only one ShadowFF
    # conv_comp_U0/razor_error_${i} output of the xor
    create_cell -reference FDRE conv_comp_U0/sample_error
    connect_net -hier -net conv_comp_U0/razor_error_1 -objects [get_pins conv_comp_U0/sample_error/D]
    connect_net -hier -net clk1_IBUF_BUFG -objects [get_pins conv_comp_U0/sample_error/C]
    connect_net -hier -net conv_comp_U0/<const1> -objects [get_pins conv_comp_U0/sample_error/CE]
    connect_net -hier -net clk1_rst_n_IBUF -objects [get_pins conv_comp_U0/sample_error/R]
    create_net conv_comp_U0/error_s
    connect_net -hier -net conv_comp_U0/error_s -objects [get_pins conv_comp_U0/sample_error/Q]
} 

#set empty_sig [get_pins -of_objects [get_cells -hier -filter {NAME =~ "*" && NAME =~ "*empty_n_reg"}] -filter { REF_PIN_NAME == Q}]
# Connect error to an empty_n_reg/Q 
#disconnect_net -net [get_nets -of_objects [get_pins conv_comp_U0/dout_vld_i_1/I1]] -objects conv_comp_U0/dout_vld_i_1/I1
#disconnect_net -net [get_nets -of_objects [get_pins conv_comp_U0/mOutPtr[1]_i_3__0/I4]] -objects conv_comp_U0/mOutPtr[1]_i_3__0/I4

disconnect_net -net [get_nets -of_objects [get_pins conv_comp_U0/dout_vld_i_1__5/I3]] -objects conv_comp_U0/dout_vld_i_1__5/I3
disconnect_net -net [get_nets -of_objects [get_pins conv_comp_U0/empty_n_i_2__22/I3]] -objects conv_comp_U0/empty_n_i_2__22/I3
#disconnect_net -net [get_nets -of_objects [get_pins empty_n]] -objects empty_n
create_cell -reference and2 conv_comp_U0/and_empty_error
create_net conv_comp_U0/empty_n_out
#connect_net -hier -net conv_comp_U0/empty_n -objects [get_pins conv_comp_U0/and_empty_error/I0]
connect_net -hier -net conv_comp_U0/empty_n_0 -objects [get_pins conv_comp_U0/and_empty_error/I0]

create_cell -reference xor2 conv_comp_U0/not_empty_error
connect_net -hier -net conv_comp_U0/error_s -objects [get_pins conv_comp_U0/not_empty_error/I0]
connect_net -hier -net conv_comp_U0/<const1> -objects [get_pins conv_comp_U0/not_empty_error/I1]
create_net conv_comp_U0/error_s_n
connect_net -hier -net conv_comp_U0/error_s_n -objects [get_pins conv_comp_U0/not_empty_error/O]
connect_net -hier -net conv_comp_U0/error_s_n -objects [get_pins conv_comp_U0/and_empty_error/I1]

connect_net -hier -net conv_comp_U0/empty_n_out -objects [get_pins conv_comp_U0/and_empty_error/O]
#connect_net -hier -net conv_comp_U0/empty_n_out -objects [get_pins conv_comp_U0/dout_vld_i_1/I1]
#connect_net -hier -net conv_comp_U0/empty_n_out -objects [get_pins conv_comp_U0/mOutPtr[1]_i_3__0/I4]
connect_net -hier -net conv_comp_U0/empty_n_out -objects [get_pins conv_comp_U0/dout_vld_i_1__5/I3]
connect_net -hier -net conv_comp_U0/empty_n_out -objects [get_pins conv_comp_U0/empty_n_i_2__22/I3]

# create error port to simulate
create_port -direction OUT error_sim
connect_net -hier -net conv_comp_U0/error_s -objects [get_ports error_sim]

############################################################