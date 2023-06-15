import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import utils.object_detection as object_detection

def info(io_dict, nl, anchor, layer_name, tot_nl=1, ws=2, nc=7, stride=32):

    # This is the network that is used to detect the objects
    # Parse io_dict to find the last layer checking that the output
    # is pre_detect_net

    ih = io_dict[layer_name]["oh"]
    iw = io_dict[layer_name]["ow"]
    och = int(nc+5)
    ich = int(io_dict[layer_name]["och"]/(nc+5))
    scale_factor = io_dict[layer_name]["scale_factor"]
    bits = io_dict[layer_name]["bits"]

    io_dict[layer_name]["output"][0] = "%s_detect" % io_dict[layer_name]["output"][0]
    
    node_name = "node_%s"  % io_dict[layer_name]["output"][0]

    detect_lut, grid_h, grid_w, anchor_grid = object_detection.detect_lut(
        nc = nc,
        anchors=[anchor],
        ch=[ich],
        input_shape=[1, och*ich, ih, iw],
        stride=[stride],
        bit_width=bits,
        scale=scale_factor,
    )
    
    # cast lists to numpy arrays
    detect_lut = np.array(detect_lut)
    grid_w = np.array(grid_w)
    grid_h = np.array(grid_h)
    # splitting anchor in ich groups of 2 elements
    anchor_grid = []
    for i in range(ich):
        anchor_grid.append(np.array(anchor[i*2:(i+1)*2]))

    anchor_grid = np.array([anchor_grid])

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = ["%s" % io_dict[layer_name]["output"][0]]
    io_dict[node_name]["output"] = ["s_detect[%0d]" % nl]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'detect'
    io_dict[node_name]["detect_lut"] = detect_lut
    io_dict[node_name]["grid_w"] = grid_w
    io_dict[node_name]["grid_h"] = grid_h
    io_dict[node_name]["anchor_grid"] = anchor_grid
    io_dict[node_name]["stride"] = stride
    io_dict[node_name]["och"] = och
    io_dict[node_name]["ih"] = ih
    io_dict[node_name]["iw"] = iw
    io_dict[node_name]["ich"] = ich
    io_dict[node_name]["nl"] = nl
    io_dict[node_name]["split"] = 2
    io_dict[node_name]["scale_factor"] = scale_factor
    io_dict[node_name]["bits"] = bits
    io_dict[node_name]["tot_nl"] = tot_nl
    io_dict[node_name]["ws"] = ws

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

    nl = node["nl"]
    # Template parameters

    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("t_%s_detect_lut_st" % name)
    block["template"].append("t_%s_grid_h_st" % name)
    block["template"].append("t_%s_grid_w_st" % name)
    block["template"].append("t_%s_anchor_grid_st" % name)
    block["template"].append("t_%s_stride_st" % name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_split" % name)
    block["template"].append("c_%s_bits" % name)
    block["template"].append("c_%s_scale_factor" % name)
    # block["template"].append("c_ws")

    block["args"] = []

    block["args"].append("s_%s[0]" % input_name)
    block["args"].append("c_%s_detect_lut" % name)
    block["args"].append("c_%s_grid_h" % name)
    block["args"].append("c_%s_grid_w" % name)
    block["args"].append("c_%s_anchor_grid" % name)
    block["args"].append("c_%s_stride" % name)
    if node["tot_nl"] > 1:
        block["args"].append("s_%s.in[%0d]" % (output_name, nl))
    else:
        block["args"].append("s_%s" % output_name)

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    output_type = "hls::vector<ap_fixed<32, 16>, %0d>" % node["och"]
    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]

    block["defines"]["t_%s_detect_lut_st" % name]  = ["type", "ap_fixed<32, 16>"]
    block["defines"]["t_%s_grid_w_st" % name]        = ["type", "ap_fixed<32, 16>"]
    block["defines"]["t_%s_grid_h_st" % name]        = ["type", "ap_fixed<32, 16>"]
    block["defines"]["t_%s_anchor_grid_st" % name] = ["type", "ap_fixed<32, 16>"]
    block["defines"]["t_%s_stride_st" % name]      = ["type", "ap_fixed<32, 16>"]

    block["defines"]["c_%s_och" % name]     = ["const", node["och"]]
    block["defines"]["c_%s_ich" % name]     = ["const", node["ich"]]
    block["defines"]["c_%s_iw" % name]      = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % name]      = ["const", node["ih"]]
    block["defines"]["c_%s_split" % name]   = ["const", node["split"]]
    block["defines"]["c_%s_scale_factor" % name]   = ["const", node["scale_factor"]]
    block["defines"]["c_%s_bits" % name]   = ["const", node["bits"]]

    block["declare"] = []

    tmp = {}
    tmp["name"] = "c_%s_detect_lut" % name
    tmp["type"] = "t_%s_detect_lut_st" % name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["detect_lut"].shape
    tmp["size"] = size
    tmp["init"] = node["detect_lut"]
    tmp["form"] = "float"

    block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = "c_%s_grid_h" % name
    tmp["type"] = "t_%s_grid_h_st" % name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["grid_h"].shape
    tmp["size"] = size
    tmp["init"] = node["grid_h"]
    tmp["form"] = "float"

    block["declare"].append(tmp)
    tmp = {}
    tmp["name"] = "c_%s_grid_w" % name
    tmp["type"] = "t_%s_grid_w_st" % name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["grid_w"].shape
    tmp["size"] = size
    tmp["init"] = node["grid_w"]
    tmp["form"] = "float"

    block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = "c_%s_anchor_grid" % name
    tmp["type"] = "t_%s_anchor_grid_st" % name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["anchor_grid"].shape
    tmp["size"] = size
    tmp["init"] = node["anchor_grid"]

    block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = "c_%s_stride" % name
    tmp["type"] = "t_%s_stride_st" % name
    tmp["is_array"] = False
    tmp["is_const"] = True
    tmp["size"] = 1
    tmp["init"] = node["stride"]

    block["declare"].append(tmp)

    block["pragma"] = []
    # depth = 2
    # pragma = {}
    # pragma["name"] = "stream"
    # options = [
    #     ["variable", "s_%s" % output_name],
    #     ["depth", "%0d" % depth],
    #     ["type", "fifo"],
    # ]
    # pragma["options"] = options
    # block["pragma"].append(pragma)
    return block 
