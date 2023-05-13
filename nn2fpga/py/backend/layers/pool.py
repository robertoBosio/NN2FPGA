import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich    = getattr(input_shape, 'dim')[1].dim_value
    ih     = getattr(input_shape, 'dim')[2].dim_value
    iw     = getattr(input_shape, 'dim')[3].dim_value
    och    = getattr(output_shape, 'dim')[1].dim_value
    oh     = getattr(output_shape, 'dim')[2].dim_value
    ow     = getattr(output_shape, 'dim')[3].dim_value

    global_pool = ("Global" in node.op_type)
    adaptive =  global_pool or ('adaptive' in node_name)

    attr_dict = {}
    for attribute in attributes:
        name = getattr(attribute, "name")
        ints = getattr(attribute, "ints")
        attr_dict[name] = ints

    if (adaptive):
        fh     = ih
        fw     = iw
        stride = 1
        pad    = 0
    else:
        fh     = attr_dict["kernel_shape"][0]
        fw     = attr_dict["kernel_shape"][1]
        stride = attr_dict["strides"][0]
        pad    = attr_dict["pads"][0]

    in_scale_factor = 0

    if 'max' in node.op_type.lower():
        pool     = 1

    if 'average' in node.op_type.lower():
        pool     = 0

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["fh"]     = fh
    io_dict[node_name]["fw"]     = fw
    io_dict[node_name]["stride"] = stride
    io_dict[node_name]["pad"]    = pad
    io_dict[node_name]["pool"]   = pool
    io_dict[node_name]["type"]   = 'pool'
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["actscale"] = []

    return io_dict

def parse(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    signed = node["signed"]

    block = {}
    block["func"] = "pool_op"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("t_%s_acc" % name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_pad" % name)
    block["template"].append("c_%s_pool" % name)
    block["template"].append("c_%s_scale_factor" % name)
    # block["template"].append("c_%s_in_scale_factor" % name)

    block["args"] = []
    block["args"].append("s_%s[0]" % input_name)
    block["args"].append("s_%s[0]" % output_name)

    block["defines"] = {}
    if (signed):
        output_type = "int8_t"
    else:
        output_type = "uint8_t"

    block["defines"]["t_%s" % output_type_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_type_name] = [
        "struct",
        [["data", "t_%s" % output_type_name], ["last", "bool"]]
    ]

    block["defines"]["t_%s_acc" % name]            = ["type", "int32_t"]
    block["defines"]["c_%s_ich" % name]            = ["const", node["ich"]]
    block["defines"]["c_%s_och" % name]            = ["const", node["och"]]
    block["defines"]["c_%s_iw" % name]             = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % name]             = ["const", node["ih"]]
    block["defines"]["c_%s_ow" % name]             = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % name]             = ["const", node["oh"]]
    block["defines"]["c_%s_fw" % name]             = ["const", node["fw"]]
    block["defines"]["c_%s_fh" % name]             = ["const", node["fh"]]
    block["defines"]["c_%s_stride" % name]         = ["const", node["stride"]]
    block["defines"]["c_%s_pad" % name]            = ["const", node["pad"]]
    block["defines"]["c_%s_pool" % name]           = ["const", node["pool"]]
    if "scale_factor" in node.keys():
        if (len(node["actscale"]) > 0):
            off = -1*(node["actscale"][0])
        else:
            off = 0

        diff_scale = off + node["scale_factor"][0] + node["in_scale_factor"]
        block["defines"]["c_%s_scale_factor" % name] = [
            "const", 
            diff_scale
        ]

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = 1

    block["declare"].append(declare)

    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s" % output_name],
        ["depth", node["och"]],
        ["type", "fifo"],
    ]
    pragma["options"] = options

    block["pragma"] = []
    block["pragma"].append(pragma)

    return block
