# Disconnect BRAM write port
# Connect write port to clk of clk_wiz_0
# Save pins with slack < perc
# create multiplexer
# For each endpoint create a shadow flip flop
# connect clk1 of shadow ff to clk1 of clk_wiz that is delayed of T/2
# create xor2 -> even/odd algo
# create or 

set_property CLOCK_DOMAINS INDEPENDENT [get_cells design_1_i/two_layers_0/inst/s_net_conv_145_U/U_two_layers_fifo_w9_d17_B_ram/mem_reg_bram_0]
set_property CLOCK_DOMAINS INDEPENDENT [get_cells design_1_i/two_layers_0/inst/s_net_conv_348_U/U_two_layers_fifo_w9_d11_B_ram/mem_reg_bram_0]
set pin1 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "design_1_i/two_layers_0/inst/s_net_conv_145_U*"}] -filter {REF_PIN_NAME == CLKBWRCLK}]
set pin2 [get_pins -of_objects [get_cells -hier -filter {REF_NAME == RAMB18E2 && NAME =~ "design_1_i/two_layers_0/inst/s_net_conv_348_U*"}] -filter {REF_PIN_NAME == CLKBWRCLK}]
set net1 [get_nets -of_objects ${pin1}]
set net2 [get_nets -of_objects ${pin2}]

disconnect_net -net ${net1} -objects ${pin1}
disconnect_net -net ${net2} -objects ${pin2}
disconnect_net -net clk_out1_0 -objects [get_ports clk_out1_0]
disconnect_net -net design_1_i/clk_out1_0 -objects [get_pins design_1_i/clk_out1_0]

connect_net -hier -net design_1_i/clk_out1_0 -objects ${pin1}
connect_net -hier -net design_1_i/clk_out1_0 -objects ${pin2}

remove_cell clk_out1_0_OBUF_inst
remove_net clk_out1_0_OBUF
remove_port clk_out1_0

disconnect_net -net clk_out2_0 -objects [get_ports clk_out2_0]
disconnect_net -net design_1_i/clk_out2_0 -objects [get_pins design_1_i/clk_out2_0]
# connect_net -hier -net design_1_i/clk_out2_0 -objects PIN

remove_cell clk_out2_0_OBUF_inst
remove_net clk_out2_0_OBUF
remove_port clk_out2_0


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
    connect_net -hier -net design_1_i/clk_out2_0 -objects [get_pins -of  ${shadow_tmp} -filter {REF_PIN_NAME==C}]
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

# ShadowFF are more than one
if "$i > 1" {

        # if the XORs are 4 or less -> create one between or4, or3, or2
        # Output of the single OR -> error net
        if "$i < 5" {
            create_cell -reference or${i} design_1_i/two_layers_0/inst/conv_comp_U0/or
            set or_err [get_cells -quiet design_1_i/two_layers_0/inst/conv_comp_U0/or]

            while "$i > 0" {
                set pin [expr {$i-1}]
                connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i} -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or/I${pin}]
                set i [expr {$i-1}]

                if {$i == 0} break
            }

            create_net design_1_i/two_layers_0/inst/conv_comp_U0/error 
            connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/error -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or/O]

        # if XORs are more than 4 -> create or2s
        } else {
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

                create_cell -reference or2 design_1_i/two_layers_0/inst/conv_comp_U0/or$n
                
                set pin2 [expr {$i_int -1}]
                connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i_int} -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or$n/I0]
                connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${pin2} -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or$n/I1]
                create_net design_1_i/two_layers_0/inst/conv_comp_U0/or_net${n}
                connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/or_net${n} -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or$n/O]
                incr n 
                incr i_int -2
            }
        }

        incr n -1

        create_cell -reference or${n} design_1_i/two_layers_0/inst/conv_comp_U0/or_l2
        set or_err_l2 [get_cells -quiet design_1_i/two_layers_0/inst/conv_comp_U0/or_l2]

        while "$n > 0" {
            set pin [expr {$n-1}]
            connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/or_net${n} -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or_l2/I${pin}]
            set n [expr {$n-1}]

            if {$n == 0} break
        }

        create_net design_1_i/two_layers_0/inst/conv_comp_U0/error_l2 
        connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/error_l2 -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or_l2/O]

        if {$odd == 1} {
            create_cell -reference or2 design_1_i/two_layers_0/inst/conv_comp_U0/or_final 
            connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/error_l2 -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or_final/I0]
            connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_1 -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or_final/I1]
            create_net design_1_i/two_layers_0/inst/conv_comp_U0/error
            connect_net -hier -net design_1_i/two_layers_0/inst/conv_comp_U0/error -objects [get_pins design_1_i/two_layers_0/inst/conv_comp_U0/or_final/O]
        }

} else {
    # Only one ShadowFF
    # design_1_i/two_layers_0/inst/conv_comp_U0/razor_error_${i} output of the xor
} 

set empty_sig [get_pins -of_objects [get_cells -hier -filter {NAME =~ "design_1_i/two_layers_0/inst/*" && NAME =~ "*empty_n_reg"}] -filter { REF_PIN_NAME == Q}]
