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
from backend.utils import *

def init(file_name, parsed_write):


    libraries = [
        "%s.hpp" % file_name,
        "ap_int.h",
        "hls_stream.h",
        "PackedConv.hpp",
        "ActivationStreams.hpp",
        "AddStreams.hpp",
        "PoolStreams.hpp",
        "Utils.hpp",
    ]

    with open("src/%s.cpp" % file_name, "w+") as fd:
        # Write header with network definitions
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")

        # Handle internal or external parameters
        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "ProduceStream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &i_%s,\n" % (name, name))

        for layer in parsed_write:
            if "MemoryManagement" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\tap_int<READ_WIDTH> *i_data_%s,\n" % name)

        for layer in parsed_write:
            if "ConsumeStream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_%s> &%s\n" % (name, name))

        fd.write(") {\n")

        fd.write("\n")

def parse_all_main(io_dict):

    parsed_write = []
    parsed_write.append(
        weights.parse_main(io_dict)
    )

    parsed_const = []

    for name, node in io_dict.items():

        if 'produce' == node["type"]:
            parsed_write.append(
                input_gen.parse(name, node)
            )

        if 'depth' == node["type"]:
            continue

        if 'conv' == node["type"]:
            if (not node["is_1x1"]):
                parsed_write.append(
                    pad.parse(name, node)
                )
                parsed_write = parsed_write + line_buffer.parse(name, node)
            parsed_write = parsed_write + conv.parse(name, node)

        if 'pool' == node["type"]:
            parsed_write.append(
                pool.parse(name, node)
            )

        # Just for the sake of constant definition
        if 'const' == node["type"]:
            parsed_const = parsed_const + weights.parse(name, node)

    parsed_write.append(
        output_gen.parse(parsed_write)
    )

    return parsed_write, parsed_const

def write(io_dict, file_name):

    parsed_write, parsed_const = parse_all_main(io_dict)

    init(file_name, parsed_write)
    declare(file_name, parsed_write)
    body(file_name, parsed_write)

    parsed_write = parsed_write + parsed_const

    defines(file_name, parsed_write)
