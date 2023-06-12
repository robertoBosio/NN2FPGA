import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )

    input_shapes = [
        tensors_info[input].tensor_type.shape in node.input
    ]
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich = []
    for input_shape in input_shapes: 
        ich.append(getattr(input_shape, 'dim')[1].dim_value)
        ih       = getattr(input_shape, 'dim')[2].dim_value
        iw       = getattr(input_shape, 'dim')[3].dim_value
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value

    feature_map = oh*ow

    io_dict[node_name]["ich"]    = np.asarray(ich)
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["type"]   = 'detect'
    io_dict[node_name]["feature_map"] = feature_map

    return io_dict

def parse(parsed_write, node_name):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "concat"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("c_%s_feature_map" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("c_%s_ich" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]
    block["defines"]["c_%s_feature_map" % node_name] = ["const", node["feature_map"]]

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)

    tmp = {}
    tmp["name"] = "c_%s_ich" % name
    tmp["type"] = "int" % name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["ich"].shape
    tmp["size"] = size
    tmp["init"] = node["ich"]

    block["declare"].append(tmp)

    block["pragma"] = []

    return block


