import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import backend.layers.conv as conv
import backend.layers.pool as pool
import backend.layers.line_buffer as line_buffer
import backend.layers.pad as pad
import backend.layers.weights as weights
import backend.layers.input_gen as input_gen
import backend.layers.output_gen as output_gen
import backend.layers.detect as detect
import backend.layers.non_max_suppression as non_max_suppression
from backend.utils import *

def init(file_name, parsed_write, object_detection=False, prj_root="/tmp"):


    libraries = [
        "params.h",
        "ap_int.h",
        "hls_stream.h",
        f"memory_management_{file_name}.h",
        "nn2fpga/packed_conv.h",
        "nn2fpga/pool_streams.h",
        "nn2fpga/utils.h",
        "nn2fpga/activations_utils.h",
        "nn2fpga/block_interface.h",
        "nn2fpga/line_buffer_utils.h",
        "nn2fpga/quantisation.h",
        "nn2fpga/weights_utils.h",
    ]

    if (object_detection):
        libraries.append("nn2fpga/detect_utils.h")
        libraries.append("nn2fpga/non_max_suppression.h")
        libraries.append("hls_np_channel.h")

    with open(prj_root + ("/cc/include/%s.h" % file_name), "w+") as fd:

        fd.write(f"#ifndef __{file_name.upper()}__H__\n")
        fd.write(f"#define __{file_name.upper()}__H__\n")
        fd.write("#include \"params.h\"\n\n")


        # Handle internal or external parameters
        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &i_%s,\n" % (name, name))

        for layer in parsed_write:
            if "memory_management" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\tconst t_%s_st *i_data_%s,\n" % (name, name))

                for name in layer["stream_input"]:
                    fd.write("\thls::stream<t_%s_stream> &i_data_%s,\n" % (name, name))

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(");\n\n")
        fd.write(f"#endif  /*__{file_name.upper()}__H__ */")
    
    # Writing also the header file. This is mandatory for the top layer as the
    # testbench need to have the definition of the function when using Vitis HLS
    with open(prj_root + ("/cc/src/%s.cc" % file_name), "w+") as fd:

        # Write header with network definitions
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")
        fd.write("#include \"params.h\"\n\n")
        fd.write("extern \"C++\" {\n\n")
        
        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &i_%s,\n" % (name, name))

        for layer in parsed_write:
            if "memory_management" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\tconst t_%s_st *i_data_%s,\n" % (name, name))

                for name in layer["stream_input"]:
                    fd.write("\thls::stream<t_%s_stream> &i_data_%s,\n" % (name, name))

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(") {\n")
        fd.write("\n")

def parse_all_main(io_dict):

    parsed_write = []
    parsed_write.append(
        weights.parse_main(io_dict)
    )

    parsed_const = []

    no_output_gen = False

    for name, node in io_dict.items():

        if 'produce' == node["type"]:
            parsed_write.append(
                input_gen.parse(name, node)
            )

        if 'depth' == node["type"]:
            continue

        if 'conv' == node["type"]:
            parsed_write = parsed_write + line_buffer.parse(name, node)
            if (node["pad"] != 0):
                parsed_write.append(
                    pad.parse(name, node)
                )
            parsed_write = parsed_write + conv.parse(name, node)

        if 'pool' == node["type"]:
            if (not node["is_adaptive"]):
                parsed_write = parsed_write + line_buffer.parse(name, node)
                if (node["pad"] != 0):
                    parsed_write.append(
                        pad.parse(name, node)
                    )
            parsed_write.append(
                pool.parse(name, node)
            )

        # Just for the sake of constant definition
        if 'const' == node["type"]:
            parsed_const = parsed_const + weights.parse(
                name,
                node
            )

        if 'detect' == node["type"]:
            parsed_write.append(
                detect.parse(name, node)
            )

        if 'non_max_suppression' == node["type"]:
            parsed_write.append(
                non_max_suppression.parse(name, node)
            )
            no_output_gen = True


        last_node_name = name

    if not no_output_gen:
        parsed_write.append(
            output_gen.parse(parsed_write, last_node_name)
        )

    return parsed_write, parsed_const

def defines(parsed_write, prj_root="/tmp"):

    libraries = [
        "ap_int.h",
        "hls_stream.h",
        "hls_vector.h",
        "stdint.h",
        "ap_axi_sdata.h",
    ]

    # Writing parameters of the network. They are written in a separate file
    # such that also the testbench can read them.
    with open(f"{prj_root}/cc/include/params.h", "w+") as fd:
        
        fd.write("#ifndef __NN2FPGA_NETWORK_PARAMS_H__\n")
        fd.write("#define __NN2FPGA_NETWORK_PARAMS_H__\n")
        fd.write("\n")
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        fd.write("\n")
        for layer in parsed_write:

            if 'defines' in layer.keys():
                write_defines(fd, layer["defines"])

        fd.write("\n")
        fd.write("#endif /*__NN2FPGA_NETWORK_PARAMS_H__ */")

def footer(file_path):

    with open(file_path, "a") as fd:
        fd.write("\n}")

def write(io_dict, file_name, ap_ctrl_chain, object_detection, prj_root="/tmp"):

    if ap_ctrl_chain:
        ap_ctrl = "ap_ctrl_chain"
    else:
        ap_ctrl = "ap_ctrl_none"

    parsed_write, parsed_const = parse_all_main(io_dict)

    file_path = f"{prj_root}/cc/src/{file_name}.cc"
    init(file_name, parsed_write, object_detection, prj_root=prj_root)
    declare(file_path, parsed_write, ap_ctrl, prj_root=prj_root)
    body(file_path, parsed_write, prj_root=prj_root)
    footer(file_path)

    parsed_write = parsed_write + parsed_const

    defines(parsed_write, prj_root=prj_root)
