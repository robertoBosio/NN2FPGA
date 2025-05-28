import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.layers.quant import get_quant_type, get_quant_constant

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )
    inputs = getattr(node, "input" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    weight_name = inputs[1] 
            
    if (len(inputs) > 2):
        bias_name = inputs[2]
    
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
    kernel   = fh*fw
    img_ch   = ich*och
    relu     = False
    add      = False
    in_scale_factor = [None]
    in_bits = [None]
    in_signed = [None]

    group = 1

    # Mark depthwise convolutions
    depth = (group == och)

    # Total number of window operations
    total = oh * ow * och * (ich / group) 

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
    io_dict[node_name]["total"]  = total
    io_dict[node_name]["kernel"] = kernel
    io_dict[node_name]["img_ch"] = img_ch
    # Reuse is generic
    io_dict[node_name]["reuse"]  = 1
    io_dict[node_name]["ow_ops"]     = 1
    io_dict[node_name]["ow_ops_out"]     = 1
    io_dict[node_name]["relu"]   = relu
    io_dict[node_name]["add"]    = add
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["bits"]    = 0
    io_dict[node_name]["in_bits"] = in_bits
    io_dict[node_name]["in_signed"] = in_signed
    io_dict[node_name]["type"]   = 'conv'
    io_dict[node_name]["wbias"]  = len(node.input) > 2
    io_dict[node_name]["wbits"]  = []
    io_dict[node_name]["wsigned"]  = []
    io_dict[node_name]["actbits"] = []
    io_dict[node_name]["wscale"]  = []
    io_dict[node_name]["actscale"] = []
    io_dict[node_name]["actsigned"] = []
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["in_ops"] = 1
    io_dict[node_name]["ich_ops"] = 1
    io_dict[node_name]["depth"] = depth
    io_dict[node_name]["weights_name"] = [weight_name]
    io_dict[node_name]["merge_1x1"] = False
    io_dict[node_name]["has_bias"] = False 
    io_dict[node_name]["has_forward"] = False
    io_dict[node_name]["start_comp_layer"] = False
    
    io_dict[node_name]["input_quant"] = None
    io_dict[node_name]["conv_output_quant"] = None
    io_dict[node_name]["output_quant"] = None
    
    if 'bias_name' in locals():
        io_dict[node_name]["bias_name"] = [bias_name]
        io_dict[node_name]["has_bias"] = True

    return io_dict
