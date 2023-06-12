import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import utils.object_detection as object_detection

def info(io_dict, graph_output_name, nl, anchors, cut_name, stride=32):

    # This is the network that is used to detect the objects
    # Parse io_dict to find the last layer checking that the output
    # is pre_detect_net

    total = 0
    for layer_name in cut_name:
        oh = io_dict[layer_name]["oh"]
        ow = io_dict[layer_name]["ow"]
        ich = io_dict[layer_name]["ich"]
        total += oh*ow*ich
        och = io_dict[layer_name]["och"]
    
    node_name = "non_max_suppression"

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = ["s_detect"]
    io_dict[node_name]["output"] = [graph_output_name]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'non_max_suppression'
    io_dict[node_name]["och"] = och
    io_dict[node_name]["total"] = total
    io_dict[node_name]["split"] = 2
    io_dict[node_name]["nl"] = nl
    io_dict[node_name]["conf_th"] = 0.001

    return io_dict

def parse(name, node):

    block = {}
    block["func"] = "non_max_suppression"

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    # Template parameters
    nl = node["nl"]

    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_total" % name)
    block["template"].append("c_%s_split" % name)
    block["template"].append("c_%s_nl" % name)

    block["args"] = []

    block["args"].append("s_%s.out" % input_name)
    block["args"].append("c_%s_conf_th" % name)
    block["args"].append("s_%s[0]" % output_name)

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    input_type = "ap_ufixed<32, 16>"
    output_type = "ap_ufixed<32, 16>"
    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]

    block["defines"]["c_%s_och" % name]     = ["const", node["och"]]
    block["defines"]["c_%s_total" % name]   = ["const", node["total"]]
    block["defines"]["c_%s_split" % name]   = ["const", node["split"]]
    block["defines"]["c_%s_nl" % name]      = ["const", node["nl"]]
    block["defines"]["c_%s_conf_th" % name] = ["const", node["conf_th"]]

    block["declare"] = []
    merge_type = "hls::merge::load_balance<t_%s_struct, %0d>" % (input_type_name, node["nl"])
    declare = {}
    declare["name"] = "s_%s" % input_name
    declare["type"] = "%s" % merge_type
    declare["is_array"] = False
    declare["is_stream"] = False
    declare["is_const"] = False
    declare["dim"] = 1
    block["declare"].append(declare)

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_type_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)
    depth = 2

    block["pragma"] = []
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s" % output_name],
        ["depth", "%0d" % depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    return block 
