proc insert_latches {block_name} {
    set i 0
    set added_cell 1
    while {$added_cell} {
        set added_cell 0
        foreach cell [get_cells -of [get_pins -of [get_cells -hier -regexp ".*${block_name}.*" -filter {is_sequential && REF_NAME !~ LD.*}] -filter {direction == in && setup_slack < 0}]] { 
            set name [get_property NAME $cell]
            if {[get_cells $cell -quiet] == ""} {
                continue
            }
            set added_cell 1
            puts $name
            set D_pin [get_pins [get_property NAME ${cell}]/D]
            set D_net [get_nets -of [get_pins ${D_pin}]]
            set Q_pin [get_pins [get_property NAME ${cell}]/Q]
            set Q_net [get_nets -of [get_pins ${Q_pin}]]
            set C_pin [get_pins [get_property NAME ${cell}]/C]
            set C_net [get_nets -of [get_pins ${C_pin}]]
            set CE_pin [get_pins [get_property NAME ${cell}]/CE]
            set CE_net [get_nets -of [get_pins ${CE_pin}]]
            set R_pin [get_pins [get_property NAME ${cell}]/R]
            set R_net [get_nets -of [get_pins ${R_pin}]]
            remove_cell ${cell}
            set name ptl_n${i}
            create_cell -reference LDCE ${name}
            connect_net -hier -net $D_net -objects [get_pins ${name}/D]
            connect_net -hier -net $Q_net -objects [get_pins ${name}/Q]
            connect_net -hier -net $C_net -objects [get_pins ${name}/G]
            connect_net -hier -net $CE_net -objects [get_pins ${name}/GE]
            connect_net -hier -net $R_net -objects [get_pins ${name}/CLR]
            incr i
            set_multicycle_path 2 -to [get_cells ${name}]
        }
    }

}