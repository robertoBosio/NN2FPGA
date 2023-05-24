import os
import sys
import onnx
from onnx import numpy_helper
import numpy as np

def write(
    additional_ports,
    weights_export,
    read_width, 
    prj_root="/tmp"
):

    def write_header(fd):
        fd.write("#ifndef __MEMORY_HEADER__\n")
        fd.write("#define __MEMORY_HEADER__\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("#include \"ap_int.h\"\n")
        fd.write("#include \"hls_vector.h\"\n")
        fd.write("#include <stdint.h>\n")

        fd.write("#define READ_WIDTH %0d\n" % read_width)
        fd.write("\n")


    def write_body(fd):
        fd.write("\n")

        for i, layer in enumerate(weights_export):
            weights = layer.flatten()

            shape = weights.flatten().shape[0]

            # fd.write("ap_uint<8> weights[%0d + 1] = {" % shape)
            fd.write("ap_int<8> weights_%s[%0d + 1] = {" % (additional_ports[i], shape))

            for i in range(weights.flatten().shape[0]):
                if (i % 16 == 0):
                    fd.write("\n")
                # fd.write("%0d, " % (int(weights[i]) & 0xff))
                fd.write("%0d, " % (int(weights[i])))

            fd.write("0};\n\n")

        pass
    
    def write_footer(fd):
        # End of main file
        fd.write("#endif")
        pass


    with open(prj_root + "/cc/include/memory_weights.h", "w+") as fd:
        write_header(fd)
        write_body(fd)
        write_footer(fd)

