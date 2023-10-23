import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.utils import write_func

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
    block["bits_data"] = {}
    block["index_data"] = {}
    block["ops_data"] = {}

    return block

def fill_uram_layer(parsed_write):
    
    for layer in parsed_write:
        if layer["func"] == "load_uram":
            block = layer

    for layer in parsed_write:
        if layer["func"] != "load_uram":
            if "uram_input" in layer.keys():
                # block["template"] = block["template"] + [layer["template"][0]]
                block["args"] = block["args"] + layer["uram_input"]
                block["mux_data"][layer["uram_input"][0]] = layer["uram_total"]
                block["bits_data"][layer["uram_input"][0]] = layer["bits"]
                block["index_data"][layer["uram_input"][0]] = layer["index"]
                block["ops_data"][layer["uram_input"][0]] = layer["ops"]
    
    return block

def init(uram_layer, network_name, prj_root="/tmp"):


    libraries = [
        "params.h",
        "ap_int.h",
        "hls_stream.h",
        "nn2fpga/weights_utils.h"
    ]

    layer_name = uram_layer['func']
    file_name = uram_layer['func'] + "_%s" % network_name

    with open(prj_root + ("/cc/include/%s.h" % file_name), "w+") as fd:
        fd.write("#ifndef __%s__\n" % file_name.upper())
        fd.write("#define __%s__\n" % file_name.upper())
        # Write header with network definitions
        fd.write("namespace nn2fpga {\n")
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")

        # Handle internal or external parameters
        fd.write("void %s(\n" % layer_name)

        # URAM read handled by external DMA
        for i, name in enumerate(uram_layer["stream_input"]):
            fd.write("\thls::stream<t_%s_stream> &i_data_%s,\n" % (name, name))

        for i, name in enumerate(uram_layer["mux_data"]):
            type_name = name.replace("s_", "t_")
            index = uram_layer["index_data"][name]
            fd.write("\thls::stream<%s> %s[%0d]" % (type_name, name, index))
            if i < (len(uram_layer["mux_data"])-1):
                fd.write(",")
            fd.write("\n")

        fd.write(") {\n")

        fd.write("\n")

def produce_func(uram_layer, output_name):
    output_type_name = output_name.replace("s_", "t_")
    dim = uram_layer["mux_data"][output_name][0]
    bytes = int(uram_layer["bits_data"][output_name]/8)
    if (bytes == 0):
        bytes = 1
    bits = uram_layer["bits_data"][output_name]
    # Limiting bits to 8 for packing reasons
    if (bits > 8):
        bits = 8
    pack = int(8/bits)
    if (pack == 0):
        pack = 1
    index = int(uram_layer["index_data"][output_name])
    ops   = int(uram_layer["ops_data"][output_name])

    block = {}
    block["func"] = "produce_stream"
    block["args"] = []
    block["input"] = []
    block["output"] = []
    block["stream_input"] = []
    block["template"] = []

    input_name = "weights"

    data_type_name = output_type_name.replace("_init", "_st")
    block["template"].append("t_%s_stream" % input_name)
    block["template"].append("%s" % data_type_name)
    block["template"].append("%s" % output_type_name)
    block["template"].append("%0d" % dim)
    block["template"].append("%0d" % index)
    block["template"].append("%0d" % bytes)
    block["template"].append("%0d" % bits)
    block["template"].append("%0d" % pack)
    block["template"].append("%0d" % ops)

    block["args"].append("i_data_%s" % input_name)
    block["args"].append("s_init")
    block["args"].append("%s" % output_name)

    block["defines"] = {}

    block["declare"] = []

    block["pragma"] = []

    # Internal mux to the block needed to provide the weights to the specific
    # layers
    block["mux_data"] = {}
    block["bits_data"] = {}

    return block


def body(uram_layer, network_name, prj_root):

    file_name = uram_layer['func'] + "_%s" % network_name

    total = 0

    with open(prj_root + ("/cc/include/%s.h" % file_name), "a") as fd:
        input_name = uram_layer["stream_input"][0]

        # By means of hls guidelines, static variable are initialized
        # to 0 by bitstream and never re-initialized by reset
        fd.write("\tstatic bool s_init;\n")

        for output_name in uram_layer["mux_data"]:

            produce_layer = produce_func(uram_layer, output_name)
            write_func(fd, produce_layer)
            
        fd.write("\ts_init = true;\n")
        fd.write("}\n")
        fd.write("} // namespace nn2fpga\n")
        fd.write("#endif")

def write(uram_layer, network_name, prj_root="/tmp"):

    init(uram_layer, network_name, prj_root=prj_root)
    body(uram_layer, network_name, prj_root)
    # declare("memory_management", parsed_write, ap_ctrl=None, inline=True, prj_root=prj_root)