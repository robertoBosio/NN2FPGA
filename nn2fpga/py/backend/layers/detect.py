import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import utils.object_detection as object_detection

def info(io_dict, graph_output_name, anchors, cut_name, stride=32):

    # This is the network that is used to detect the objects
    # Parse io_dict to find the last layer checking that the output
    # is pre_detect_net

    oh = []
    ow = []
    och = []
    ich = []
    scale_factor = []
    for i, layer_name in enumerate(cut_name):
        oh.append(io_dict[layer_name]["oh"])
        ow.append(io_dict[layer_name]["ow"])
        och.append(io_dict[layer_name]["och"])
        ich.append(io_dict[layer_name]["ich"])
        scale_factor.append(io_dict[layer_name]["scale_factor"])
    
    node_name = "detect"

    detect_lut_nl = []
    grid_nl = []
    anchor_grid_nl = []
    for i, nl in enumerate(anchors):
        detect_lut, grid, anchor_grid = object_detection.detect_lut(
            nc = 7,
            anchors=[nl],
            ch=[ich[i]],
            input_shape=[1, och[i], oh[i], ow[i]],
            stride=[stride],
        )
        detect_lut_nl.append(detect_lut)
        grid_nl.append(grid)
        anchor_grid_nl.append(anchor_grid)
    
    # cast lists to numpy arrays
    detect_lut_nl = np.array(detect_lut_nl)
    grid_nl = np.array(grid_nl)
    anchor_grid_nl = np.array(anchor_grid_nl)

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = ["s_detect"]
    io_dict[node_name]["output"] = [graph_output_name]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'detect'
    io_dict[node_name]["detect_lut"] = detect_lut_nl
    io_dict[node_name]["grid"] = grid_nl
    io_dict[node_name]["anchor_grid"] = anchor_grid_nl
    io_dict[node_name]["stride"] = stride
    io_dict[node_name]["och"] = och
    io_dict[node_name]["oh"] = oh
    io_dict[node_name]["ow"] = ow
    io_dict[node_name]["ich"] = ich
    io_dict[node_name]["split"] = 2
    io_dict[node_name]["scale_factor"] = scale_factor

    return io_dict

def detect(io_dict, pre_detect_net):
    detect_lut(
        nc = 7,
        anchors=anchors,
        ch=[ich],
        input_shape=[1, och, oh, ow],
        stride=[32],
    )
    return io_dict

def parse(name, node):

    block = {}
    block["func"] = "detect"

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    # Template parameters

    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_split" % name)

    block["args"] = []

    block["args"].append("s_%s" % input_name)
    block["args"].append("c_%s_detect_lut" % input_name)
    block["args"].append("c_%s_grid" % input_name)
    block["args"].append("c_%s_anchor_grid" % input_name)
    block["args"].append("c_%s_stride" % input_name)
    block["args"].append("s_%s" % output_name)

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    input_type = "uint8_t"
    output_type = "ap_ufixed<32, 16>"
    block["defines"] = {}
    block["defines"]["t_%s" % input_name] = ["type", input_type]
    block["defines"]["t_%s_struct" % input_name] = [
        "struct",
        [["data", "t_%s" % input_name], ["last", "bool"]]
    ]
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]

    block["defines"]["t_%s_detect_lut_st" % output_name]  = ["type", "ap_fixed<32,16>"]
    block["defines"]["t_%s_grid_st" % output_name]        = ["type", "hls::vector<ap_fixed<8,8>, 3>"]
    block["defines"]["t_%s_anchor_grid_st" % output_name] = ["type", "hls::vector<ap_uint<16>, 2>"]
    block["defines"]["t_%s_stride_st" % output_name]      = ["type", "ap_uint<8>"]

    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_type_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)

    tmp = {}
    tmp["name"] = "c_%s_detect_lut" % input_name
    tmp["type"] = "t_%s_detect_lut_st" % input_name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["detect_lut"].shape
    tmp["size"] = size
    tmp["init"] = node["detect_lut"]

    block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = "c_%s_grid" % input_name
    tmp["type"] = "t_%s_grid_st" % input_name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["grid"].shape
    tmp["size"] = size
    tmp["init"] = node["grid"]

    block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = "c_%s_anchor_grid" % input_name
    tmp["type"] = "t_%s_anchor_grid_st" % input_name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["anchor_grid"].shape
    tmp["size"] = size
    tmp["init"] = node["anchor_grid"]

    block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = "c_%s_stride" % input_name
    tmp["type"] = "t_%s_stride_st" % input_name
    tmp["is_array"] = False
    tmp["is_const"] = True
    tmp["size"] = 1
    tmp["init"] = node["stride"]

    block["declare"].append(tmp)

    depth = 2
    block["pragma"] = []
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s" % input_name],
        ["depth", "%0d" % depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    return block 
