import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )

    input_shapes = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    upsample_name   = io_dict[node_name]["input"][2]
    upsample_info   = init_info[scale_name]
    upsample_factor = numpy_helper.to_array(scale_info)[-1]

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
    io_dict[node_name]["factor"]     = upsample_factor
    io_dict[node_name]["type"]   = 'upsample'
    io_dict[node_name]["scale_factor"] = 0

    return io_dict

def parse(parsed_write, node_name):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "upsample_op"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("c_%s_ich" % node_name)
    block["template"].append("c_%s_ih" % node_name)
    block["template"].append("c_%s_iw" % node_name)
    block["template"].append("c_%s_factor" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]
    block["defines"]["c_%s_ich" % node_name] = ["const", node["ich"]]
    block["defines"]["c_%s_ih" % node_name] = ["const", node["ih"]]
    block["defines"]["c_%s_iw" % node_name] = ["const", node["iw"]]
    block["defines"]["c_%s_factor" % node_name] = ["const", node["factor"]]

    if node["scale_factor"] is list:
        scale_factor = node["scale_factor"][0]
    else:
        scale_factor = node["scale_factor"]

    block["defines"]["c_%s_scale_factor" % name] = [
        "const",
        scale_factor
    ]

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)

    block["pragma"] = []

    return block



