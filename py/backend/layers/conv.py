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

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value
    fh       = getattr(attributes[2], 'ints')[0]
    fw       = getattr(attributes[2], 'ints')[1]
    stride   = getattr(attributes[4], 'ints')[0]
    pad      = getattr(attributes[3], 'ints')[0]
    is_1x1   = (fh == 1) and (fw == 1)
    total    = 1/(oh*ow*och*ich*fh*fw)
    kernel   = fh*fw
    img_ch   = oh*ow
    reuse    = 1
    add      = False
    in_scale_factor = 0

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
    io_dict[node_name]["is_1x1"] = is_1x1
    io_dict[node_name]["total"]  = total
    io_dict[node_name]["kernel"] = kernel
    io_dict[node_name]["img_ch"] = img_ch
    io_dict[node_name]["reuse"]  = reuse
    io_dict[node_name]["add"]    = add
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["type"]   = 'conv'

    return io_dict

def parse(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    weight_name = node["input"][1]


    # If no batchnorm merge then there is no bias
    has_bias = len(node["input"]) > 2
    if (has_bias):
        bias_name   = node["input"][2]

    if (node["add"]):
        add_name = node["input"][3]
        add_type_name = add_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    if (node["has_forward"]):
        forward_name = node["output"][1]
    if (node["merge_1x1"]):
        output_1x1_name = node["output"][1]
        output_1x1_type_name = output_1x1_name.replace("_skip", "")

    block = {}
    block["func"] = "ConvOp"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s" % weight_name)
    if (has_bias):
        block["template"].append("t_%s" % bias_name)
    if (node["add"]):
        block["template"].append("t_%s_struct" % add_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    if (node["merge_1x1"]):
        block["template"].append("t_%s_struct" % output_1x1_type_name)
        block["template"].append("t_%s" % output_1x1_type_name)
    block["template"].append("t_%s_acc" % name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_relu" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_pad" % name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_%s_scale_shift" % name)
    block["template"].append("c_%s_reuse" % name)
    block["template"].append("c_%s_in_scale_shift" % name)

    block["args"] = []
    if node["is_1x1"]:
        block["args"].append("s_%s" % input_name)
    else:
        block["args"].append("s_%s_compute" % input_name)
    block["args"].append("s_%s" % weight_name)
    if (has_bias):
        block["args"].append("s_%s" % bias_name)
    if (node["add"]):
        block["args"].append("s_%s" % add_name)
    if (node["has_forward"]):
        block["args"].append("s_%s" % forward_name)
    block["args"].append("s_%s" % output_name)
    if (node["merge_1x1"]):
        block["args"].append("s_%s" % output_1x1_name)

    block["declare"] = []

    block["pragma"] = []

    return block
