set ap_clk_nets [get_net_delays -of_objects [get_nets design_1_i/zynq_ultra_ps_e_0_pl_clk0]]
foreach {net} $ap_clk_nets { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_ap_clk.txt
}

set ap_rst_nets [get_net_delays -of_objects [get_nets design_1_i/rst_ps8_0_214M_peripheral_aresetn]]
foreach {net} $ap_rst_nets { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_ap_rst.txt
}


set clk1_nets [get_net_delays -of_objects [get_nets design_1_i/clk_wiz_0_clk_out1]]
foreach {net} $clk1_nets { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_clk1.txt
}

set clk2_nets [get_net_delays -of_objects [get_nets design_1_i/clk_wiz_0_clk_out2]]
foreach {net} $clk2_nets { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_clk2.txt
}


set num 0
while "$num != 64" {
    if {$num == 64} break
    
    set input_net [get_net_delays -of_objects [get_nets design_1_i/axi_dma_0_M_AXIS_MM2S_TDATA[$num]]]

    foreach {net} $input_net { 
        report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_in$num.txt
    }
    incr num
} 


set last_net [get_net_delays -of_objects [get_nets design_1_i/axi_dma_0_M_AXIS_MM2S_TLAST]]
foreach {net} $last_net { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_last.txt
}

set valid_net [get_net_delays -of_objects [get_nets design_1_i/axi_dma_0_M_AXIS_MM2S_TVALID]]
foreach {net} $valid_net { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_valid.txt
}


set ready_net [get_net_delays -of_objects [get_nets design_1_i/two_layers_0_o_outp1_TREADY]]
foreach {net} $ready_net { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_ready.txt
}


set rst1_net [get_net_delays -of_objects [get_nets design_1_i/peripheral_aresetn]]
foreach {net} $rst1_net { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_rst1.txt
}



set rst2_net [get_net_delays -of_objects [get_nets design_1_i/peripheral_aresetn1]]
foreach {net} $rst2_net { 
    report_property -all [lindex $net] -append -file /home/lavinia/Razor_KRIA_ip_simulation/delay/delay_rst2.txt
}