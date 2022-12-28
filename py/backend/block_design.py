import os
import sys
import onnx
from onnx import numpy_helper

def write(
    additional_ports,
    additional_ports_info
):

    read_width = 128

    os.system("cp tcl/bd_design.tcl tmp/bd_design.tcl")
    with open("tmp/bd_design.tcl", "a") as fd:
        
        inv = []
        inv.append(["empty", "Network_0"])
        inv.append(["full", "MemoryManagement_0"])

        en = []
        en.append(["write", "wr_en", "MemoryManagement_0"])
        en.append(["read", "rd_en", "Network_0"])
        en.append(["dout", "dout", "Network_0"])
        en.append(["din", "din", "MemoryManagement_0"])

        for port in additional_ports:
            for index in range(additional_ports_info[port][0]):
                if (additional_ports_info[port][0] == 1):
                    fifo_name = "fifo_generator_%s" % (port)
                else:
                    fifo_name = "fifo_generator_%s_%0d" % (port, index)
                fd.write("# Create instance: %s, and set properties\n" % fifo_name)
                fd.write("set %s [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 %s ] \n" % (fifo_name, fifo_name))
                fd.write("set_property -dict [ list ")
                fd.write("CONFIG.Input_Data_Width {%0d} " % additional_ports_info[port][1])
                fd.write("CONFIG.Output_Data_Width {%0d} " % additional_ports_info[port][1])
                fd.write("] $%s\n" % fifo_name)

                for port_list in inv:
                    if (additional_ports_info[port][0] == 1):
                        weight_name = "s_%s" % (
                            port
                        )
                    else:
                        weight_name = "s_%s_%0d" % (
                            port,
                            index
                        )
                    port_name = port_list[0]
                    if (additional_ports_info[port][0] == 1):
                        inv_name = "inv_%s_%s" % (port, port_name)
                    else:
                        inv_name = "inv_%s_%0d_%s" % (port, index, port_name)
                    fd.write("# Create instance: %s, and set properties\n" % inv_name)
                    fd.write("set %s [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 %s]\n" % (inv_name, inv_name))
                    fd.write("set_property -dict [ list ")
                    fd.write("CONFIG.C_OPERATION {not} ")
                    fd.write("CONFIG.C_SIZE {1} ")
                    fd.write("CONFIG.LOGO_FILE {data/sym_notgate.png} ")
                    fd.write("] $%s\n" % inv_name)

                    # Connecting the fifo output flag to the inverter input
                    fd.write("connect_bd_net -net %s_%s [get_bd_pins %s/%s] [get_bd_pins %s/Op1]\n" % (fifo_name, port_name, fifo_name, port_name, inv_name))

                    fd.write("connect_bd_net -net %s_Res [get_bd_pins %s/%s_%s_n] [get_bd_pins %s/Res]\n" % (inv_name, port_list[1], weight_name, port_name, inv_name))

                for en_list in en:
                    if (additional_ports_info[port][0] == 1):
                        port_name = "s_%s_%s" % (
                            port,
                            en_list[0]
                        )
                    else:
                        port_name = "s_%s_%0d_%s" % (
                            port,
                            index,
                            en_list[0]
                        )
                    fd.write(
                        "connect_bd_net -net %s_%s [get_bd_pins %s/%s] [get_bd_pins %s/%s]\n" % (
                            en_list[2],
                            port_name,
                            en_list[2],
                            port_name,
                            fifo_name,
                            en_list[1]
                        )
                    )

