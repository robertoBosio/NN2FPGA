import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np


def dma_func(
    input_name,
    dim
):

    block = {}

    block["func"] = "dma_emulator"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["uram_input"] = []
    block["output"] = []

    block["template"] = []
    block["template"].append("t_%s_stream" % (input_name))
    block["template"].append("t_%s_st" % (input_name))
    block["template"].append("c_%s_dim" % (input_name))
    block["args"].append("c_%s" % input_name)
    block["args"].append("c_%s_stream" % input_name)

    block["declare"] = []

    tmp = {}
    tmp["name"] = "c_%s_stream" % input_name
    tmp["type"] = "t_%s_stream" % input_name
    tmp["is_array"] = True
    tmp["dim"] = 1

    block["declare"].append(tmp)

    block["defines"] = {}
    block["defines"]["c_%s_dim" % (input_name)] = ["const", dim]

    block["pragma"] = []

    return block

def add_uram_layer():
    
    block = {}
    block["func"] = "load_uram"
    block["args"] = []
    block["input"] = []
    block["output"] = []
    block["stream_input"] = []

    input_name = "weights"

    block["stream_input"].append("%s" % input_name)
    block["args"].append("i_data_%s" % input_name)

    block["defines"] = {}
    block["defines"]["t_%s_st" % (input_name)]    = ["type", "int8_t"]

    block["declare"] = []

    block["pragma"] = []

    # Internal mux to the block needed to provide the weights to the specific
    # layers
    block["mux_data"] = {}
    return block

def fill_uram_layer(parsed_write):
    
    for layer in parsed_write:
        if layer["func"] == "load_uram":
            block = layer

    block["mux_data"] = {}
    for layer in parsed_write:
        if layer["func"] != "load_uram":
            if "uram_input" in layer.keys():
                # block["template"] = block["template"] + [layer["template"][0]]
                block["args"] = block["args"] + layer["uram_input"]
                block["mux_data"][layer["uram_input"][0]] = layer["uram_total"]
    
    return block

def init(uram_layer, network_name, prj_root="/tmp"):


    libraries = [
        "%s.h" % network_name,
        "ap_int.h",
        "hls_stream.h",
    ]

    file_name = uram_layer['func']

    with open(prj_root + ("/cc/include/%s.h" % file_name), "w+") as fd:
        # Write header with network definitions
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")

        # Handle internal or external parameters
        fd.write("void %s(\n" % file_name)

        # URAM read handled by external DMA
        for i, name in enumerate(uram_layer["stream_input"]):
            fd.write("\thls::stream<t_%s_st_stream> &i_data_%s,\n" % (name, name))

        for i, name in enumerate(uram_layer["mux_data"]):
            type_name = name.replace("s_", "t_")
            fd.write("\thls::stream<%s> &%s" % (type_name, name))
            if i < (len(uram_layer["mux_data"])-1):
                fd.write(",")
            fd.write("\n")

        fd.write(") {\n")

        fd.write("\n")

def body(uram_layer, network_name, prj_root):

    file_name = uram_layer['func']

    total = 0

    with open(prj_root + ("/cc/include/%s.h" % file_name), "a") as fd:
        input_name = uram_layer["stream_input"][0]
        input_type_name = input_name.replace("s_", "t_")
        for output_name in uram_layer["mux_data"]:
            total = total + uram_layer["mux_data"][output_name][0]

        # By means of hls guidelines, static variable are initialized
        # to 0 by bitstream and never re-initialized by reset
        fd.write("\tstatic bool s_init;\n")

        for output_name in uram_layer["mux_data"]:
            dim = uram_layer["mux_data"][output_name][0]
            fd.write("\tfor (auto i = 0; i < %0d; i++) {\n" % dim)
            fd.write("\t\tif (~s_init) {\n")
            fd.write(
                "\t\t\tt_%s_st_stream s_data = i_data_%s.read();\n" % (
                    input_type_name,
                    input_name
                )
            )
            fd.write("\t\t\t%s << s_data.data;\n" % output_name)
            fd.write("\t\t}\n")
        
            fd.write("\t}\n\n")
            
        fd.write("\ts_init = true;\n")
        fd.write("}\n")

def write(uram_layer, network_name, prj_root="/tmp"):

    init(uram_layer, network_name, prj_root=prj_root)
    body(uram_layer, network_name, prj_root)
    # declare("memory_management", parsed_write, ap_ctrl=None, inline=True, prj_root=prj_root)