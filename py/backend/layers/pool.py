import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich    = getattr(input_shape, 'dim')[1].dim_value
    ih     = getattr(input_shape, 'dim')[2].dim_value
    iw     = getattr(input_shape, 'dim')[3].dim_value
    och    = getattr(output_shape, 'dim')[1].dim_value
    oh     = getattr(output_shape, 'dim')[2].dim_value
    ow     = getattr(output_shape, 'dim')[3].dim_value
    fh     = getattr(attributes[0], 'ints')[0]
    fw     = getattr(attributes[0], 'ints')[1]
    stride = getattr(attributes[2], 'ints')[0]
    pad    = getattr(attributes[1], 'ints')[0]
    adaptive = ('adaptive' in node_name) or ((fh == ih) and (fw == iw))

    if (adaptive):
        fh     = oh
        fw     = ow
        stride = 1
        pad    = 0

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["fh"]     = fh
    io_dict[node_name]["fw"]     = fw
    io_dict[node_name]["stride"] = stride
    io_dict[node_name]["pad"]    = pad
    io_dict[node_name]["type"]   = 'pool'

    return io_dict

def parse(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "PoolOp"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("t_%s_acc" % name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_pad" % name)
    block["template"].append("c_%s_scale_shift" % name)
    block["template"].append("c_%s_in_scale_shift" % name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = False
    declare["dim"] = 1

    block["declare"].append(declare)

    pragma = {}
    pragma["name"] = "s_%s" % output_name
    pragma["depth"] = 2

    block["pragma"] = []

    return block
