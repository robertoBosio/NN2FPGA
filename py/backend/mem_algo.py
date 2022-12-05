import os
import sys
import onnx
from onnx import numpy_helper

def round_robin(
    fd,
    ports,
    read_width
):
    for name in ports:
        n_read = "\tc_%s_ih*c_%s_iw*c_%s_och*c_%s_ich/%0d" % (
            name,
            name,
            name,
            name,
            read_width/8
        )
        fd.write("\t\tFillStream <\n")
        fd.write("\t\t\t%0d,\n" % read_width)
        fd.write("\t\t%s\n" % n_read)
        fd.write("\t\t> (\n")
        fd.write("\t\t\ti_data,\n")
        fd.write("\t\t\ts_%s_address,\n" % name)
        fd.write("\t\t\to_%s\n" % name)
        fd.write("\t\t);\n")
        fd.write("\n")

def write(
    additional_ports,
    read_width,
    fd
):

    def write_header(fd):

        fd.write("void MemAlgo(\n")
        for name in additional_ports:
            fd.write(
                "\thls::stream<ap_uint<READ_WIDTH>> &o_%s,\n" % (
                    name
                )
            )
        fd.write("\tap_uint<%0d> *i_data\n" % read_width)
        fd.write(") {\n")
        fd.write("\n")

        for name in additional_ports:
            fd.write("\tuint32_t s_%s_address = 0;\n" % (name))

        fd.write("\n")
        fd.write("\tdo{\n")

        pass

    def write_body(fd):
        fd.write("\n")
        round_robin(
            fd,
            additional_ports,
            read_width
        )

        pass
    
    def write_footer(fd):
        # End of main file
        fd.write("\t}while(1);\n")
        fd.write("\n")
        fd.write("}\n")
        pass


    write_header(fd)
    write_body(fd)
    write_footer(fd)
