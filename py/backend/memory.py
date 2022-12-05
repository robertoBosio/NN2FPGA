import os
import sys
import onnx
from onnx import numpy_helper
import backend.mem_algo as mem_algo

def write(
    additional_ports
):

    read_width = 128
    def write_header(fd):
        fd.write("#include \"Network.hpp\"\n")
        fd.write("#include \"MemUtils.hpp\"\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("#include \"ap_int.h\"\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("\n")

        fd.write("#define READ_WIDTH %0d\n" % read_width)
        fd.write("\n")

        mem_algo.write(
            additional_ports,
            read_width,
            fd
        )

        fd.write("\n")
        fd.write("void MemoryManagement(\n")
        for name in additional_ports:
            fd.write(
                "\thls::stream<t_%s> o_%s[c_%s_index],\n" % (
                    name, 
                    name, 
                    name
                )
            )
        fd.write("\tap_uint<READ_WIDTH> *i_data\n")
        fd.write(") {\n")
        fd.write("\n")
        fd.write("\t#pragma HLS DATAFLOW\n")
        fd.write("\t#pragma HLS INTERFACE s_axilite port=return\n")
        fd.write("\t#pragma HLS interface m_axi port=i_data\n")
        fd.write("\n")

        for name in additional_ports:
            fd.write("\thls::stream<ap_uint<READ_WIDTH>> s_%s(\"%s\");\n" % (name, name))
            depth = 4096/(read_width/8)
            fd.write(
                "\t#pragma HLS INTERFACE mode=ap_fifo port=s_%s\n" % (
                    name
                )
            )

            fd.write(
                "\t#pragma HLS STREAM variable=s_%s depth=%0d type=fifo\n" % (
                    name,
                    depth
                )
            )

        pass

    def write_body(fd):
        fd.write("\n")
        fd.write("\tMemAlgo(\n")
        for name in additional_ports:
            fd.write("\t\ts_%s,\n" % (name))
        fd.write("\t\ti_data\n")
        fd.write("\t);\n")
        fd.write("\n")

        for name in additional_ports:
            fd.write("\tProduceStream<\n")
            fd.write("\t\tt_%s,\n" % (name))
            fd.write("\t\tc_%s_iw,\n" % (name))
            fd.write("\t\tc_%s_ih,\n" % (name))
            fd.write("\t\tc_%s_ops,\n" % (name))
            fd.write("\t\t%0d\n" % (read_width))
            fd.write("\t>(\n")
            fd.write("\t\ts_%s,\n" % (name))
            fd.write("\t\to_%s\n" % (name))
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
