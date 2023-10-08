# Disconnect BRAM write port
# Connect write port to clk of clk_wiz_0
# Save pins with slack < perc
# create multiplexer
# For each endpoint create a shadow flip flop
# connect clk1 of shadow ff to clk1 of clk_wiz that is delayed of T/2
# create xor2 -> even/odd algo
# create or 

set pin1 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "design_1_i/two_layers_0/inst/s_net_conv_145_U*"}] -filter {REF_PIN_NAME == CLKBWRCLK}]
set pin2 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "design_1_i/two_layers_0/inst/s_net_conv_348_U*"}] -filter {REF_PIN_NAME == CLKBWRCLK}]
set net1 [get_nets -of_objects ${pin1}]
set net2 [get_nets -of_objects ${pin2}]

disconnect_net -net ${net1} -objects ${pin1}
disconnect_net -net ${net2} -objects ${pin2}
connect_net -hier -net design_1_i/clk_wiz_0/clk_out1 -objects ${pin1}
connect_net -hier -net design_1_i/clk_wiz_0/clk_out1 -objects ${pin2}

set period [get_property PERIOD [get_clocks clk_pl_0]]
set perc 0.7
set perc_period [expr ${period}*${perc}]

set endpoints [get_pins -of_objects [get_cells -hier -filter {NAME =~ "design_1_i/two_layers_0/inst/conv_comp_U0*" && NAME !~ "*DSP48*"  && NAME !~ "*mul_8s_8s_16_1_1_U91" && NAME !~ "*s_acc_buff_V_0_U*" && IS_SEQUENTIAL }] -filter "DIRECTION == IN && SETUP_SLACK <= ${perc_period} && REF_PIN_NAME == D"]

set i $0

foreach endpoint ${endpoints} {
    
    #set cell_${i} [get_cells -of_objects $endpoints]
    set cell_${i} [get_cells -of_objects $endpoint]
    set cell_tmp [set cell_${i}]
    #set input_net_${i} [get_nets -of $endpoints]
    set input_net_${i} [get_nets -of $endpoint]
    set input_net_tmp [set input_net_${i}]
    create_cell -reference MUXF7 design_1_i/two_layers_0/inst/conv_comp_U0/mux_${i}
    set mux_tmp [get_cells -quiet design_1_i/two_layers_0/inst/conv_comp_U0/mux_${i}]
    disconnect_net -net ${input_net_tmp} -objects [get_pins -of  ${cell_tmp} -filter {REF_PIN_NAME==D}]
    connect_net -net ${input_net_tmp} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==I0}]
    create_net design_1_i/two_layers_0/inst/conv_comp_U0/mux_out_${i}
    # MUX/O -> MAINFF/D
    connect_net -net design_1_i/two_layers_0/inst/conv_comp_U0/mux_out_${i} -objects [get_pins -of  ${mux_tmp} -filter {REF_PIN_NAME==O}]
    connect_net -net design_1_i/two_layers_0/inst/conv_comp_U0/mux_out_${i} -objects [get_pins -of  ${cell_tmp} -filter {REF_PIN_NAME==D}]
    
    # Create SHADOW FF
    create_cell -reference FDRE design_1_i/two_layers_0/inst/conv_comp_U0/shadowFF_${i}
    set shadow_tmp [get_cells -quiet design_1_i/two_layers_0/inst/conv_comp_U0/shadowFF_${i}]
    # Connect nets shadow FF
    # MUX/O -> SHADOWFF/D
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/mux_out_${i}  -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==D}]
    connect_net -hier -net design_1_i/clk_wiz_0/clk_out2 -objects [get_pins -of  ${shadow_tmp} -filter {REF_PIN_NAME==C}]
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/<const1> -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==CE}]
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/<const0> -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==R}]
    create_net design_1_i/two_layers_0/inst/conv_comp_U0/shadow_value_${i}
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/shadow_value_${i} -objects [get_pins -of ${shadow_tmp} -filter {REF_PIN_NAME==Q}]
    # SHADOWFF/O -> MUX/I1
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/shadow_value_${i} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==I1}]
    
    # Output MAINFF
    set output_net_${i} [get_nets -of [get_pins -of  ${cell_tmp} -filter {REF_PIN_NAME==Q}]]
    set output_net_tmp [set output_net_${i}]

    # Creating the xor gate
    create_cell -reference xor2 design_1_i/two_layers_0/inst/conv_comp_U0/xor_${i}
    set xor_err_tmp [get_cells -quiet design_1_i/two_layers_0/inst/conv_comp_U0/xor_${i}]
    connect_net -hier -net ${output_net_tmp} -objects [get_pins -of ${xor_err_tmp} -filter {REF_PIN_NAME==I0}]
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/shadow_value_${i} -objects [get_pins -of ${xor_err_tmp} -filter {REF_PIN_NAME==I1}]
    # XOR OUT
    create_net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i}
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i} -objects [get_pins -of  ${xor_err_tmp} -filter {REF_PIN_NAME==O}]
    # Connect XOR out to select signal of MUX
    connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==S}]

    incr i
}

set i [expr {$i-1}]
connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i} -objects [get_pins -of ${mux_tmp} -filter {REF_PIN_NAME==S}]



