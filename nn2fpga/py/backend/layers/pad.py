import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, node, node_name, tensors_info, init_info):

    attributes = getattr(node, "attribute" )
    inputs = getattr(node, "input" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    pads_info   = init_info[inputs[1]]
    pad        = numpy_helper.to_array(pads_info)[0]

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value
    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["pad"]    = pad

    return io_dict

def parse(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = input_name
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "pad_input"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_lb_struct" % input_type_name)
    block["template"].append("t_%s_window_struct" % input_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_pad" % name)
    block["template"].append("c_%s_ow_ops" % name)
    block["template"].append("%0d" % node["line_ops"]) #line_ops
    block["template"].append("%0d" % node["ich_ops"])

    block["args"] = []

    block["args"].append("s_%s_pre_pad" % input_name)
    block["args"].append("s_%s_compute" % input_name)

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s_compute" % input_name
    declare["type"] = "t_%s_window_struct" % input_name
    declare["is_array"] = True
    declare["dim"] = 1

    block["declare"].append(declare)

    # depth = node["och"]*int(node["och"]/node["ich"])*4
    depth = 2
    # depth = node["fh"]*node["fw"]
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s_compute" % (input_name)],
        ["depth", depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options

    block["pragma"] = []
    block["pragma"].append(pragma)

    return block
