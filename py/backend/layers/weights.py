import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.graph import extract_connections
from backend.utils import *

def parse_on_chip(
    node_info,
    pre_values
):

    dich = node_info["ich"]
    dih  = node_info["ih"]
    diw  = node_info["iw"]
    doch = node_info["och"]
    dops = node_info["ops"]
    scale_factor = 2**node_info["scale_factor"]

    doch_ops = int(doch/dops)

    values = np.zeros(
        [dih*diw, dich*doch_ops, dops]
    )

    for ih in range(dih-1, -1, -1):
        for iw in range(diw-1, -1, -1):
            for ich in range(dich):
                for och in range(doch_ops):
                    off = och*dops
                    for ops in range(dops):
                        quant_value = pre_values[off+ops][ich][ih][iw]
                        quant_value = int(quant_value/scale_factor)
                        index = ih*diw+iw
                        ch = ich*doch_ops+och
                        values[dih*diw-1-index][ch][ops] = quant_value
    
    return values

def parse_off_chip(
    node_info,
    pre_values
):

    dich = node_info["ich"]
    dih  = node_info["ih"]
    diw  = node_info["iw"]
    doch = node_info["och"]
    dops = node_info["ops"]
    scale_factor = 2**node_info["scale_factor"]

    values = np.zeros(
        [dih*diw*dich*doch]
    )

    addr = 0
    for ich in range(dich):
        for och in range(int(doch/dops)):
            for ih in range(dih-1, -1, -1):
                for iw in range(diw-1, -1, -1):
                    off = och*dops
                    for ops in range(dops):
                        quant_value  = pre_values[off+ops][ich][ih][iw]
                        quant_value  = int(quant_value/scale_factor)
                        values[addr] = quant_value
                        addr         = addr + 1
    
    return values

def extract_info(
    new_node,
    weight_name,
    node_info,
    init_info,
    off_chip_memory
):
    pre_values = numpy_helper.to_array(init_info[weight_name]) 
    shape = pre_values.shape

    och = shape[0]
    if (len(shape) > 1):
        ich = shape[1]
        is_bias = False
    else:
        ich = 1
        pre_values = np.expand_dims(pre_values, axis=-1)
        is_bias = True

    if (len(shape) > 2):
        ih = shape[2]
        iw = shape[3]
    else:
        ih = 1
        iw = 1
        pre_values = np.expand_dims(pre_values, axis=-1)
        pre_values = np.expand_dims(pre_values, axis=-1)

    oh     = node_info["oh"]
    ow     = node_info["ow"]
    stride = node_info["stride"]
    pad    = node_info["pad"]

    # bias must be always uploaded in parallel
    if (is_bias):
        ops = 1
    else:
        ops = node_info["ops"]

    if (is_bias):
        new_node["data_type"] = "int16_t"
    else:
        new_node["data_type"] = "int8_t"
    new_node["ich"]    = ich
    new_node["ih"]     = ih
    new_node["iw"]     = iw
    new_node["och"]    = och
    new_node["oh"]     = oh
    new_node["ow"]     = ow
    new_node["stride"] = stride
    new_node["pad"]    = pad
    new_node["ops"]    = ops
    new_node["is_bias"] = is_bias
    new_node["type"]    = 'const'

    dim = ich*och*ih*iw

    new_node["off_chip_memory"] = off_chip_memory and (dim < 4096)

    if new_node["off_chip_memory"]:
        new_node["values"] = parse_off_chip(
            new_node,
            pre_values
        )
    else:
        new_node["values"] = parse_on_chip(
            new_node,
            pre_values
        )

    return new_node

def weights_info(
    model,
    io_dict,
    init_info,
    off_chip_memory=False
):
    new_nodes = {}
    rem_nodes = []

    io_connect = extract_connections(model, io_dict)

    for node_name, node_info in io_dict.items():
        if 'quant' in node_name.lower():
            all_constant = all(
                [
                    input in init_info.keys()
                    for input in node_info["input"]
                ]
            )
            # Must be checked that all quant layers are related to parameters
            # and that all activations quant layer have been absorbed
            assert all_constant

            conv_name = io_connect[node_info["output"][0]][1][0]

            assert 'conv' in conv_name.lower()

            conv_info = io_dict[conv_name]

            tensor_name = node_info["input"][0]
            new_node_name = "Weight%s" % tensor_name
            new_nodes[new_node_name] = {}
            new_nodes[new_node_name]["input"] = [new_node_name]
            new_nodes[new_node_name]["output"] = [node_info["output"][0]]
            new_nodes[new_node_name]["is_constant"] = True
            new_nodes[new_node_name]["scale_factor"] = node_info["scale_factor"]

            new_nodes[new_node_name] = extract_info(
                new_nodes[new_node_name],
                tensor_name,
                conv_info,
                init_info,
                off_chip_memory
            )

            rem_nodes.append(node_name)
            
    for node_name, node_info in new_nodes.items():
        io_dict[node_name] = node_info

    for node_name in rem_nodes:
        del io_dict[node_name]

    return io_dict

def parse_main(io_dict):
    
    block = {}
    block["func"] = "MemoryManagement"
    block["args"] = []
    block["input"] = []
    block["output"] = []

    block["declare"] = []
    block["pragma"] = []
    for name, node in io_dict.items():
        if 'const' == node["type"]:
            input_name = node["input"][0]
            output_name = node["output"][0]
            if node["off_chip_memory"]:
                block["input"].append("i_data_%s" % input_name)
                block["args"].append("i_data_%s" % input_name)
            block["args"].append("s_%s" % output_name)

            tmp = {}
            tmp["name"] = "s_%s" % output_name
            tmp["type"] = "t_%s" % output_name
            tmp["dim"]  = node["ih"]*node["iw"]
            tmp["is_array"] = True

            block["declare"].append(tmp)

            pragma = {}
            pragma["name"] = "stream"
            options = [
                ["variable", "s_%s" % output_name],
                ["depth", "2"],
                ["type", "fifo"],
            ]
            pragma["options"] = options
            block["pragma"].append(pragma)

    return block

def parse(name, node):
    
    input_name = node["input"][0]
    output_name = node["output"][0]

    blocks = []
    if node["off_chip_memory"]:
        block = {}
        block["func"] = "MemAlgo"
        block["args"] = []
        block["input"] = []
        block["output"] = []

        block["template"] = []
        block["template"].append("t_%s" % (output_name))
        block["template"].append("c_%s_ich" % (name))
        block["template"].append("c_%s_och" % (name))
        block["template"].append("c_%s_ow" % (name))
        block["template"].append("c_%s_oh" % (name))
        block["template"].append("c_%s_iw" % (output_name))
        block["template"].append("c_%s_ih" % (output_name))
        block["template"].append("c_%s_ops" % (output_name))
        block["template"].append("READ_WIDTH")
        block["template"].append("c_%s_bw" % (output_name))
        block["template"].append("c_%s_reuse" % (output_name))
        block["template"].append("0")

        block["input"].append("i_data_%s" % output_name)
        block["args"].append("s_o_%s" % output_name)
        block["args"].append("i_data_%s" % output_name)

        block["declare"] = []

        block["pragma"] = []

        blocks.append(block)

        block = {}

        block["func"] = "ProduceStream"
        block["args"] = []
        block["input"] = []
        block["output"] = []

        block["template"] = []
        block["template"].append("ap_int<READ_WIDTH> ")
        block["template"].append("t_%s" % (output_name))
        block["template"].append("c_%s_ich" % (name))
        block["template"].append("c_%s_och" % (name))
        block["template"].append("c_%s_ow" % (name))
        block["template"].append("c_%s_oh" % (name))
        block["template"].append("c_%s_iw" % (output_name))
        block["template"].append("c_%s_ih" % (output_name))
        block["template"].append("c_%s_ops" % (output_name))
        block["template"].append("c_%s_bw" % (output_name))
        block["template"].append("c_%s_reuse" % (output_name))
        block["template"].append("READ_WIDTH")

        block["output"].append("s_%s" % output_name)
        block["args"].append("s_o_%s" % output_name)
        block["args"].append("s_%s" % output_name)

        block["defines"] = {}
        output_type_name = "hls::vector<%s, %0d>" % (node["data_type"], node["ops"])
        block["defines"]["t_%s_st" % (name)]       = ["type",  output_type_name]
        block["defines"]["t_%s" % (name)]       = ["type",  output_type_name]
        block["defines"]["c_%s_ich" % (name)]          = ["const", node["ich"]]
        block["defines"]["c_%s_och" % (name)]          = ["const", node["och"]]
        block["defines"]["c_%s_ow" % (name)]           = ["const", node["ow"]]
        block["defines"]["c_%s_oh" % (name)]           = ["const", node["oh"]]
        block["defines"]["c_%s_iw" % (output_name)]    = ["const", node["iw"]]
        block["defines"]["c_%s_ih" % (output_name)]    = ["const", node["ih"]]
        block["defines"]["c_%s_ops" % (output_name)]   = ["const", node["ops"]]
        block["defines"]["c_%s_bw" % (output_name)]    = ["const", node["bw"]]
        block["defines"]["c_%s_reuse" % (output_name)] = ["const", node["reuse"]]

        block["declare"] = []

        block["pragma"] = []

    else:

        block = {}

        block["func"] = "ProduceStream"
        block["args"] = []
        block["input"] = []
        block["output"] = []

        block["template"] = []
        block["template"].append("t_%s_st" % (output_name))
        block["template"].append("t_%s" % (output_name))
        block["template"].append("c_%s_ich" % (name))
        block["template"].append("c_%s_och" % (name))
        block["template"].append("c_%s_ow" % (name))
        block["template"].append("c_%s_oh" % (name))
        block["template"].append("c_%s_iw" % (output_name))
        block["template"].append("c_%s_ih" % (output_name))
        block["template"].append("c_%s_ops" % (output_name))

        block["output"].append("%s" % output_name)
        block["args"].append("c_%s" % output_name)
        block["args"].append("s_%s" % output_name)

        block["declare"] = []
        tmp = {}
        tmp["name"] = "c_%s" % output_name
        tmp["type"] = "t_%s" % output_name
        tmp["is_array"] = True
        tmp["init"] = node["values"]

        block["declare"].append(tmp)

        block["defines"] = {}
        block["defines"]["t_%s_st" % (output_name)]    = ["type", node["data_type"]]
        output_type_name = "hls::vector<%s, %0d>" % (node["data_type"], node["ops"])
        block["defines"]["t_%s" % (output_name)]       = ["type",  output_type_name]
        block["defines"]["c_%s_ich" % (name)]          = ["const", node["ich"]]
        block["defines"]["c_%s_och" % (name)]          = ["const", node["och"]]
        block["defines"]["c_%s_ow" % (name)]           = ["const", node["ow"]]
        block["defines"]["c_%s_oh" % (name)]           = ["const", node["oh"]]
        block["defines"]["c_%s_iw" % (output_name)]    = ["const", node["iw"]]
        block["defines"]["c_%s_ih" % (output_name)]    = ["const", node["ih"]]
        block["defines"]["c_%s_ops" % (output_name)]   = ["const", node["ops"]]

        pragma = {}
        pragma["name"] = "array_partition"
        options = [
            ["variable", "c_%s" % (output_name)],
            ["type", "block"],
            ["factor", 1],
            ["dim", 1],
        ]
        pragma["options"] = options

        block["pragma"] = []
        block["pragma"].append(pragma)
    block["size"] = node["iw"]*node["ih"]
    block["is_const"] = True
    blocks.append(block)

    return blocks

def parse_all(io_dict):

    parsed_write = []

    for name, node in io_dict.items():

        if 'const' == node["type"]:
            parsed_write = parsed_write + parse(name, node)

    return parsed_write

def init(file_name, network_name, parsed_write):


    libraries = [
        "%s.hpp" % network_name,
        "ap_int.h",
        "MemUtils.hpp",
        "hls_stream.h",
    ]

    with open("src/%s.cpp" % file_name, "w+") as fd:
        # Write header with network definitions
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")

        # Handle internal or external parameters
        fd.write("void MemoryManagement(\n")

        for layer in parsed_write:
            for name in layer["input"]:
                fd.write("\tap_int<READ_WIDTH> *i_data_%s,\n" % (name))

        for i, layer in enumerate(parsed_write):
            for j, name in enumerate(layer["output"]):
                fd.write(
                    "\thls::stream<t_%s> s_%s[%0d]" % (
                        name,
                        name,
                        layer["size"]
                    )
                )
                if i < (len(parsed_write)-1) or j < (len(layer["output"])-1):
                    fd.write(",")
                fd.write("\n")

        fd.write(") {\n")

        fd.write("\n")

def write(io_dict, network_name):

    parsed_write = parse_all(io_dict)

    init("MemoryManagement", network_name, parsed_write)
    declare("MemoryManagement", parsed_write)
    body("MemoryManagement", parsed_write)

