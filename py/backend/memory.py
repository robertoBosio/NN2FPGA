import os
import sys
import onnx
from onnx import numpy_helper

def write(
    additional_ports,
    additional_ports_info,
    read_width
):

    def write_header(fd):
        fd.write("#include \"Network.hpp\"\n")
        fd.write("#include \"MemUtils.hpp\"\n")
        fd.write("#include \"ap_int.h\"\n")
        fd.write("#include \"MemoryManagement.hpp\"\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("\n")

        fd.write("\n")

        # mem_algo.write(
        #     additional_ports,
        #     additional_ports_info,
        #     read_width,
        #     fd
        # )

        for i, name in enumerate(additional_ports):
            node_name = additional_ports_info[name][6]

            fd.write("void f_mem_algo_%s (\n" % (name))
            fd.write("\thls::stream<t_%s> &o_data,\n" % (name))
            fd.write("\tap_int<READ_WIDTH> *i_data \n")
            fd.write(") {\n")
            fd.write("\tMemAlgo< \n")
            fd.write("\t\tt_%s,\n" % (name))
            fd.write("\t\tc_%s_ich,\n" % (node_name))
            fd.write("\t\tc_%s_och,\n" % (node_name))
            fd.write("\t\tc_%s_ow,\n" % (node_name))
            fd.write("\t\tc_%s_oh,\n" % (node_name))
            fd.write("\t\tc_%s_iw,\n" % (name))
            fd.write("\t\tc_%s_ih,\n" % (name))
            fd.write("\t\tc_%s_ops,\n" % (name))
            fd.write("\t\tREAD_WIDTH,\n")
            fd.write("\t\t0\n")
            fd.write("\t>( \n")
            fd.write("\t\to_data,\n")
            fd.write("\t\ti_data \n")
            fd.write("\t); \n")
            fd.write("} \n")
            fd.write("\n")

        for i, name in enumerate(additional_ports):
            node_name = additional_ports_info[name][6]

            fd.write("void f_%s (\n" % (name))
            fd.write("\thls::stream<t_%s> &i_data,\n" % (name))
            fd.write("\thls::stream<t_%s> o_data[c_%s_index] \n" % (name, name))
            fd.write(") {\n")
            fd.write("\tProduceStream< \n")
            fd.write("\t\tap_int<READ_WIDTH>, \n")
            fd.write("\t\tt_%s,\n" % (name))
            fd.write("\t\tc_%s_ich,\n" % (node_name))
            fd.write("\t\tc_%s_och,\n" % (node_name))
            fd.write("\t\tc_%s_ow,\n" % (node_name))
            fd.write("\t\tc_%s_oh,\n" % (node_name))
            fd.write("\t\tc_%s_iw,\n" % (name))
            fd.write("\t\tc_%s_ih,\n" % (name))
            fd.write("\t\tc_%s_ops,\n" % (name))
            fd.write("\t\tREAD_WIDTH \n")
            fd.write("\t>( \n")
            fd.write("\t\ti_data,\n")
            fd.write("\t\to_data \n")
            fd.write("\t); \n")
            fd.write("} \n")
            fd.write("\n")

        fd.write("void MemoryManagement(\n")
        for name in additional_ports:
            fd.write(
                "\thls::stream<t_%s> s_%s[c_%s_index],\n" % (
                    name, 
                    name, 
                    name
                )
            )
        for i, name in enumerate(additional_ports):
            fd.write("\tap_int<READ_WIDTH> *i_data_%s" % name)
            if (i < (len(additional_ports)-1)):
                fd.write(",")
            fd.write("\n")
        fd.write(") {\n")
        fd.write("\n")
        fd.write("\t#pragma HLS inline\n")
        fd.write("\n")

        for name in additional_ports:
            fd.write(
                "\thls::stream<t_%s> s_o_%s;\n" % (
                    name,
                    name
                )
            )

            fd.write(
                "\t#pragma HLS STREAM variable=s_o_%s depth=4096/c_%s_ops type=fifo\n" % (
                    name,
                    name
                    # depth
                )
            )

        pass

    def write_body(fd):
        fd.write("\n")

        for i, name in enumerate(additional_ports):
            fd.write("\tf_mem_algo_%s (\n" % (name))
            fd.write("\t\ts_o_%s,\n" % (name))
            fd.write("\t\ti_data_%s\n" % (name))
            fd.write("\t);\n")

            fd.write("\n")

        for i, name in enumerate(additional_ports):
            node_name = additional_ports_info[name][6]
            fd.write("\tf_%s (\n" % (name))
            fd.write("\t\ts_o_%s,\n" % (name))
            fd.write("\t\ts_%s\n" % (name))
            fd.write("\t);\n")

            fd.write("\n")

        pass
    
    def write_footer(fd):
        # End of main file
        fd.write("}\n")
        pass


    with open("src/MemoryManagement.cpp", "w+") as fd:
        write_header(fd)
        write_body(fd)
        write_footer(fd)
