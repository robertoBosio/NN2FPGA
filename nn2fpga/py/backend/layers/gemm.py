import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import backend.quant
from backend.layers.quant import get_quant_type, get_quant_constant

def info(io_dict, node, node_name, init_info, tensors_info, enable_ws):

    attributes = getattr(node, "attribute" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = 1
    iw       = 1
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = 1
    ow       = 1
    fh       = 1
    fw       = 1
    stride   = 1
    pad      = 0
    is_1x1   = (fh == 1) and (fw == 1)
    kernel   = fh*fw
    img_ch   = ich*och
    relu     = False
    add      = False
    in_scale_factor = [None]
    in_bits = [None]

    groups = 1

    if (groups == och):
        depth = 1
    else:
        depth = 0

    if depth == 0:
        total    = 1/(oh*ow*och*ich)
        total_log = 2*oh*ow*och*ich*fh*fw
    else:
        total    = 1/(oh*ow*och)
        total_log = 2*oh*ow*och*fh*fw

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
    io_dict[node_name]["total_log"]  = total_log
    io_dict[node_name]["kernel"] = kernel
    io_dict[node_name]["img_ch"] = img_ch
    # Reuse is generic
    io_dict[node_name]["enable_ws"] = enable_ws
    io_dict[node_name]["reuse"]  = 1
    # Ws are the operations in parallel
    io_dict[node_name]["ws"]     = 1
    io_dict[node_name]["ws_out"] = 1
    io_dict[node_name]["relu"]   = relu
    io_dict[node_name]["add"]    = add
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["bits"]    = 0
    io_dict[node_name]["in_bits"] = in_bits
    io_dict[node_name]["type"]   = 'conv'
    io_dict[node_name]["wbias"]  = len(node.input) > 2
    io_dict[node_name]["wbits"]  = []
    io_dict[node_name]["actbits"] = []
    io_dict[node_name]["wscale"]  = []
    io_dict[node_name]["actscale"] = []
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["in_ops"] = 1
    io_dict[node_name]["ich_ops"] = 1
    io_dict[node_name]["depth"] = depth

    return io_dict
