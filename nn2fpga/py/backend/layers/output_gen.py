import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, graph_output_name):

    node_name = "consume_stream"

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = [graph_output_name]
    io_dict[node_name]["output"] = []
    io_dict[node_name]["type"] = 'consume'
    io_dict[node_name]["ow_ops"]     = 1

    return io_dict

def parse(node, node_name):
    
    node_name = "consume_stream"
    input_name = node["output"][-1]
    input_name = input_name.replace("s_", "")
    input_type_name = input_name.replace("_skip", "")
    output_name = input_name
    output_type_name = input_type_name

    block = {}
    block["func"] = "consume_stream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_o_%s" % output_type_name)
    block["template"].append("c_%s_och" % node_name)
    block["template"].append("c_%s_ow" % node_name)
    block["template"].append("c_%s_oh" % node_name)
    block["template"].append("c_%s_ow_ops" % node_name)
    block["template"].append("c_%s_ops" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("o_outp1")

    output_type = "hls::axis<t_%s, 0, 0, 0>" % input_type_name
    block["defines"] = {}

    block["defines"]["t_o_%s" % output_type_name] = [
        "type",
        "%s" % output_type
    ]

    block["defines"]["t_out_mem"] = ["alias", "t_%s" % input_type_name]
    block["defines"]["t_o_outp1"] = ["alias", "t_o_%s" % output_type_name]
    block["defines"]["t_o_data"] = ["alias", "t_o_%s" % output_type_name]
    block["defines"]["c_%s_och" % node_name] = ["const", node["och"]]
    block["defines"]["c_%s_ow" % node_name] = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % node_name] = ["const", node["oh"]]
    block["defines"]["c_%s_ow_ops" % node_name] = ["const", 1]
    block["defines"]["c_%s_ops" % node_name] = ["const", node["ops"]]

    block["output"] = ["outp1"]
    block["declare"] = []

    block["pragma"] = []

    pragma = {}
    pragma["name"] = "interface"
    options = [
        ["port", "o_outp1"],
        ["mode", "axis"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    return block

