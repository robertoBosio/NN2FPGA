import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, tensors_info, model):


    node_name = "ProduceStream"

    graph_input_name = model.graph.input[0].name
    input_shape = tensors_info[graph_input_name].tensor_type.shape

    graph_input_name = graph_input_name.replace(".", "_")

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = [graph_input_name]
    io_dict[node_name]["output"] = [graph_input_name]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'produce'

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw

    return io_dict

def parse(name, node):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    signed = node["signed"]

    block = {}
    block["func"] = "ProduceStream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_part" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s" % input_name)
    block["template"].append("c_%s_scale_factor" % name)

    block["args"] = []
    block["args"].append("i_%s" % input_name)
    block["args"].append("s_%s[0]" % output_name)

    block["input"] = ["%s" % input_name]

    block["defines"] = {}
    block["defines"]["c_%s" % input_name] = ["const", 64]
    block["defines"]["t_%s" % input_type_name] = [
        "type",
        "ap_axiu<c_%s, 0, 0, 0>" % input_name
    ]

    if (signed):
        output_type = "int8_t"
    else:
        output_type = "uint8_t"

    block["defines"]["t_%s_part" % input_type_name] = [
        "type",
        "uint8_t"
    ]
    block["defines"]["t_%s" % output_type_name] = [
        "type",
        output_type
    ]
    block["defines"]["t_%s_struct" % output_type_name] = [
        "struct",
        [["data", output_type], ["last", "bool"]]
    ]

    block["defines"]["c_ProduceStream_ich"] = [
        "const",
        node["ich"]
    ]
    block["defines"]["c_ProduceStream_iw"] = [
        "const",
        node["iw"]
    ]
    block["defines"]["c_ProduceStream_ih"] = [
        "const",
        node["ih"]
    ]

    block["defines"]["c_%s_ich" % name] = [
        "const",
        node["ich"]
    ]
    block["defines"]["c_%s_iw" % name] = [
        "const",
        node["iw"]
    ]
    block["defines"]["c_%s_ih" % name] = [
        "const",
        node["ih"]
    ]

    scale_factor = 8+node["scale_factor"][0]

    block["defines"]["c_%s_scale_factor" % name] = [
        "const",
        scale_factor
    ]

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)

    block["pragma"] = []

    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s" % (output_name)],
        ["depth", 2],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    return block

