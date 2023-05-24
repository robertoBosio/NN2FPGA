import os
import sys
import onnx
from onnx import numpy_helper

def round_robin(
    fd,
    ports,
    ports_info,
    read_width
):
    burst_size = 4096
    maximum_reads = max([ports_info[name][4] for name in ports]) 

    fd.write(
        "\tfor (uint32_t s_iter = 0; s_iter < %0d; s_iter++) {\n" %
        # maximum_reads
        1
    )

    fd.write(
        "#pragma HLS pipeline style=frp\n"
    )

    fd.write("\t\tuint8_t s_sel = round_robin <\n")
    fd.write("\t\t\t%0d,\n" % len(ports))
    fd.write("\t\t\tREAD_WIDTH\n")
    fd.write("\t\t> (\n")
    fd.write("\t\t\to_streams\n")
    fd.write("\t\t);\n")

    fd.write(
        "\t\tuint32_t s_burst_size = (c_address_end[s_sel] - s_read_address[s_sel]);\n"
    )

    fd.write(
        "\t\tif(s_burst_size > 4096)\n"
    )

    fd.write(
        "\t\t\ts_burst_size = 4096;\n"
    )

    fd.write(
        "\t\tuint32_t s_read;\n"
    )
    fd.write(
        "\t\tuint32_t s_end_value = c_address_end[s_sel] + s_burst_size;\n"
    )
    fd.write(
        "\t\tfor (s_read = s_read_address[s_sel]; s_read < s_end_value; s_read++)\n"
    )

    fd.write(
        "\t\t\to_streams[s_sel].write(i_data[s_read]);\n"
    )

    fd.write(
        "\t\ts_read_address[s_sel] = (c_address_end[s_sel] >= s_read) "
    )
    
    fd.write(
        "? s_read : c_address_start[s_sel];\n"
    )
    

    fd.write("\t};\n")



def write(
    additional_ports,
    additional_ports_info,
    read_width,
    fd
):

    def write_header(fd):

        fd.write(
            "uint32_t s_read_address[%0d+1] = {\n" % len(additional_ports)
        )
        for name in additional_ports:
            fd.write(
                "%0d, " % (
                    additional_ports_info[name][2]
                )
            )
        fd.write(
            "0\n"
        )

        fd.write(
            "};\n"
        )

        fd.write("void mem_algo(\n")
        fd.write(
            "\thls::stream<ap_uint<READ_WIDTH>> o_streams[%0d],\n" % (
                len(additional_ports)
            )
        )
        fd.write("\tap_uint<%0d> *i_data\n" % read_width)
        fd.write(") {\n")
        fd.write("\n")

        # for name in additional_ports:
        #     fd.write(
        #         "\tuint32_t s_%s_address[%0d];\n" % (
        #             name,
        #             len(additional_ports)
        #         )
        #     )

        fd.write("\n")

        pass

    def write_body(fd):
        fd.write("\n")
        round_robin(
            fd,
            additional_ports,
            additional_ports_info,
            read_width
        )

        pass
    
    def write_footer(fd):
        # End of main file
        fd.write("\n")
        fd.write("}\n")
        pass


    write_header(fd)
    write_body(fd)
    write_footer(fd)
