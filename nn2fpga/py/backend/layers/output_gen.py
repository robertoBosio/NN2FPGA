import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, graph_output_name, i):

    node_name = "consume_stream_" + str(i)

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = [graph_output_name]
    io_dict[node_name]["output"] = []
    io_dict[node_name]["index_out"] = i
    io_dict[node_name]["type"] = 'consume'
    io_dict[node_name]["ow_ops"]     = 1
    io_dict[node_name]["i"]     =  i
    return io_dict

def parse(node, node_name, i):
    
    node_name = "consume_stream_" + str(i)
    input_name = node["output"][-1]
    input_name = input_name.replace("s_", "")
    input_type_name = input_name.replace("_skip", "")
    output_name = input_name
    output_type_name = input_type_name

    block = {}
    block["func"] = "consume_stream"
    block["index_out"] = i

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
    block["args"].append("o_outp%s" % i)

    output_type = "hls::axis<t_%s, 0, 0, 0>" % input_type_name
    block["defines"] = {}

    block["defines"]["t_o_%s" % output_type_name] = [
        "type",
        "%s" % output_type
    ]

    block["defines"]["t_out_mem%s" % i] = ["alias", "t_%s" % input_type_name]
    block["defines"]["t_o_outp%s" % i] = ["alias", "t_o_%s" % output_type_name]
    block["defines"]["t_o_data%s" % i] = ["alias", "t_o_%s" % output_type_name]
    if node["type"] == "adjust":
        block["defines"]["c_%s_och" % node_name] = ["const", node["ich"]]
        block["defines"]["c_%s_ow" % node_name] = ["const", node["iw"]]
        block["defines"]["c_%s_oh" % node_name] = ["const", node["ih"]]
        block["defines"]["c_%s_ops" % node_name] = ["const", node["och_ops"]]
    else:   
        block["defines"]["c_%s_och" % node_name] = ["const", node["och"]]
        block["defines"]["c_%s_ow" % node_name] = ["const", node["ow"]]
        block["defines"]["c_%s_oh" % node_name] = ["const", node["oh"]]
        block["defines"]["c_%s_ops" % node_name] = ["const", node["ops"]]

    block["defines"]["c_%s_ow_ops" % node_name] = ["const", 1]

    block["output"] = ["outp%s" % i]
    block["declare"] = []

    block["pragma"] = []

    pragma = {}
    pragma["name"] = "interface"
    options = [
        ["port", "o_outp%s" % i],
        ["mode", "axis"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    return block

