import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.layers.quant import get_quant_type

def info(io_dict, node, node_name, init_info, tensors_info, enable_ws):

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

    adaptive |= (fh == iw) and (fw == ih) and (stride == 1) and (pad == 0)

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
    io_dict[node_name]["is_adaptive"] = adaptive
    io_dict[node_name]["actbits"] = []
    io_dict[node_name]["enable_ws"] = enable_ws
    io_dict[node_name]["ws"] = 1
    io_dict[node_name]["ws_out"] = 1
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["in_ops"] = 1

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
    block["template"].append("t_%s_vector" % input_type_name)
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
    block["template"].append("c_%s_ws" % name)
    block["template"].append("c_%s_ws_out" % name)
    block["template"].append("c_%s_ops" % name)
    if (node["is_adaptive"]):
        block["template"].append("c_%s_in_ops" % name)
    # block["template"].append("c_ws")
    # block["template"].append("c_%s_in_scale_factor" % name)

    block["args"] = []
    if (node["is_adaptive"]):
        block["args"].append("s_%s" % input_name)
    else:
        if node["pad"] == 0:
            block["args"].append("s_%s_pre_pad" % input_name)
        else:
            block["args"].append("s_%s_compute" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    output_type = get_quant_type(node["signed"], node["bits"][0], node["scale_factor"][0])
    block["defines"]["t_%s" % output_type_name] = ["type", output_type]
    output_vector_type = "std::array<t_%s, %s>" % (output_type_name, node["ops"])
    block["defines"]["t_%s_vector" % output_type_name] = ["type", output_vector_type]
    block["defines"]["t_%s_struct" % output_type_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, 1>" % output_type_name], ["last", "bool"]]
    ]

    if node["pool"] == 1:
        block["defines"]["t_%s_acc" % name]            = ["type", output_type]
    else:
        acc_type = get_quant_type(True, 32, node["actscale"][0], acc_reg=True)
        block["defines"]["t_%s_acc" % name]            = ["type", acc_type]
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
    block["defines"]["c_%s_ws" % name]             = ["const", node["ws"]]
    block["defines"]["c_%s_ws_out" % name]         = ["const", node["ws_out"]]
    block["defines"]["c_%s_ops" % name]            = ["const", node["ops"]]
    block["defines"]["c_%s_in_ops" % name]         = ["const", node["in_ops"]]

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
