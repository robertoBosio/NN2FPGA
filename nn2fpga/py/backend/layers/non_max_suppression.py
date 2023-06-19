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
        detect_layer_name = "node_%s"  % io_dict[layer_name]["output"][0]
        ih = io_dict[detect_layer_name]["ih"]
        iw = io_dict[detect_layer_name]["iw"]
        ich = io_dict[detect_layer_name]["ich"]
        total += ih*iw*ich
        och = io_dict[detect_layer_name]["och"]
    
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
    block["func"] = "consume_stream"

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    # Template parameters
    nl = node["nl"]

    if (node["nl"] == 1):
        merge_stream = False
    else:
        merge_stream = True

    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_o_outp1")
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_total" % name)
    block["template"].append("c_%s_split" % name)
    block["template"].append("c_%s_nl" % name)

    block["args"] = []

    if merge_stream:
        block["args"].append("s_%s.out" % input_name)
    else:
        block["args"].append("s_%s" % input_name)
    block["args"].append("c_%s_conf_th" % name)
    block["args"].append("o_outp1")

    block["output"] = []
    block["output"].append("outp1")

    input_type = "ap_fixed<32, 16>"
    output_type = "hls::axis<hls::vector<ap_ufixed<16, 16>, 8>, 0, 0, 0>"
    block["defines"] = {}
    block["defines"]["t_o_%s" % output_name] = ["type", output_type]

    block["defines"]["c_%s_och" % name]     = ["const", node["och"]]
    block["defines"]["c_%s_total" % name]   = ["const", node["total"]]
    block["defines"]["c_%s_split" % name]   = ["const", node["split"]]
    block["defines"]["c_%s_nl" % name]      = ["const", node["nl"]]
    block["defines"]["c_%s_conf_th" % name] = ["const_float", node["conf_th"]]

    block["defines"]["t_o_outp1"] = ["alias", "t_o_%s" % output_type_name]
    block["defines"]["t_o_data"] = ["alias", "t_o_%s" % output_type_name]

    block["declare"] = []
    if merge_stream:
        merge_type = "hls::merge::load_balance<t_%s_struct, %0d>" % (input_type_name, node["nl"])
    else:
        merge_type = "hls::stream<t_%s_struct, 300>" % input_type_name
    declare = {}
    declare["name"] = "s_%s" % input_name
    declare["type"] = "%s" % merge_type
    declare["is_array"] = False
    declare["is_stream"] = False
    declare["is_const"] = False
    declare["dim"] = 1
    block["declare"].append(declare)

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
