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

def init(file_name, parsed_write):


    libraries = [
        "%s.hpp" % file_name,
        "hls_stream.h",
        "ap_int.h",
        "hls_stream.h",
        "PackedConv.hpp",
        "ActivationStreams.hpp",
        "AddStreams.hpp",
        "PoolStreams.hpp",
        "Utils.hpp",
        "MemoryManagement.hpp",
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
                    fd.write("\thls::stream<t_%s> &%s,\n" % (name, name))

        for layer in parsed_write:
            if "MemoryManagement" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\tap_int<READ_WIDTH> *i_data_%s,\n" % name)

        name = parsed_write[-1]["args"][-1]
        fd.write("\thls::stream<t_%s> &%s\n" % (name, name))
        fd.write(") {\n")

        fd.write("\n")

def write_func(fd, info):

    fd.write("\t%s" % info["func"])
    if "template" in info.keys():
        fd.write(" <\n")
        for i, template in enumerate(info["template"]):
            fd.write("\t\t%s" % template)
            if i < len(info["template"])-1:
                fd.write(",\n")
            else:
                fd.write("\n")
                fd.write("\t>")
    fd.write(" (\n")
    for i, arg in enumerate(info["args"]):
        fd.write("\t\t%s" %arg)
        if i < len(info["args"])-1:
            fd.write(",\n")
        else:
            fd.write("\n")
            fd.write("\t);\n")
    fd.write("\n")


def body(file_name, parsed_write):

    with open("src/%s.cpp" % file_name, "a") as fd:
        
        for layer in parsed_write:

            write_func(fd, layer)

        fd.write("}\n")
    
def write_declare(fd, stream):
    
    name = stream["name"]
    type_name = stream["type"]
    dim = stream["dim"]
    fd.write("\thls::stream<%s> %s[%0d];\n" % (type_name, name, dim))

def write_pragma(fd, pragma):
    
    name = pragma["name"]
    depth = pragma["depth"]
    fd.write(
        "\t#pragma HLS STREAM variable=%s depth=%0d type=fifo\n" %
        (name, depth)
    )

def declare(file_name, parsed_write):

    with open("src/%s.cpp" % file_name, "a") as fd:
        
        for layer in parsed_write:

            for stream in layer["declare"]:
                write_declare(fd, stream)

            for pragma in layer["pragma"]:
                write_pragma(fd, pragma)

        fd.write("\n")
    
def input_block(name, node):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "ProduceStream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s" % input_name)
    block["template"].append("c_%s_scale_shift" % name)

    block["args"] = []
    block["args"].append("i_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["input"] = ["i_%s" % input_name]

    block["declare"] = []

    block["pragma"] = []

    return block

def output_block(parsed_write):
    
    input_name  = parsed_write[-1]["args"][-1]
    input_type_name = input_name.replace("_skip", "")
    output_name = input_name.replace("s_", "o_")
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "ConsumeStream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_och" % output_name)
    block["template"].append("c_%s_ow" % output_name)
    block["template"].append("c_%s_oh" % output_name)

    block["args"] = []
    block["args"].append("%s" % input_name)
    block["args"].append("%s" % output_name)

    block["declare"] = []

    block["pragma"] = []

    return block

def parse_all_main(io_dict):

    parsed_write = []
    parsed_write.append(
        weights.parse_main(io_dict)
    )

    for name, node in io_dict.items():

        if 'produce' == node["type"]:
            parsed_write.append(
                input_block(name, node)
            )

        if 'depth' == node["type"]:
            continue

        if 'conv' == node["type"]:
            if (not node["is_1x1"]):
                parsed_write.append(
                    pad.parse(name, node)
                )
                parsed_write = parsed_write + line_buffer.parse(name, node)
            parsed_write.append(
                conv.parse(name, node)
            )

        if 'pool' == node["type"]:
            parsed_write.append(
                pool.parse(name, node)
            )

    parsed_write.append(
        output_block(parsed_write)
    )

    return parsed_write

def write(io_dict, file_name):

    parsed_write = parse_all_main(io_dict)

    init(file_name, parsed_write)
    declare(file_name, parsed_write)
    body(file_name, parsed_write)

