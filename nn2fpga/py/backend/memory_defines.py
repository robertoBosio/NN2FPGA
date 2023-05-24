import os
import sys
import onnx
from onnx import numpy_helper
import numpy as np

def write(
    additional_ports,
    additional_ports_info,
    read_width, 
    prj_root="/tmp"
):

    def write_header(fd):
        fd.write("#ifndef __MEMORY_DEFINES__\n")
        fd.write("#define __MEMORY_DEFINES__\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("#include \"ap_int.h\"\n")
        fd.write("#include \"hls_vector.h\"\n")
        fd.write("#include <stdint.h>\n")

    def write_body(fd):
        # fd.write(
        #     "\tconst uint32_t c_address_start[%0d+1] = {\n" % (
        #         len(additional_ports)
        #     )
        # )

        # for name in additional_ports:
        #     fd.write(
        #         "%0d, " % (
        #             additional_ports_info[name][2]
        #         )
        #     )
        # fd.write(
        #     "0\n"
        # )
        # fd.write(
        #     "\t};\n"
        # )

        # fd.write(
        #     "\tconst uint32_t c_address_end[%0d+1] = {\n" % (
        #         len(additional_ports)
        #     )
        # )

        # for name in additional_ports:
        #     fd.write(
        #         "%0d, " % (
        #             additional_ports_info[name][5]
        #         )
        #     )
        # fd.write(
        #     "0\n"
        # )
        # fd.write(
        #     "\t};\n"
        # )

        pass
    
    def write_footer(fd):
        # End of main file

        fd.write("void memory_management(\n")
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
        fd.write(");\n")

        # Adding prototype declaration
        fd.write("#endif")
        pass


    with open(prj_root + "/cc/include/memory_management.h", "w+") as fd:
        write_header(fd)
        write_body(fd)
        write_footer(fd)

