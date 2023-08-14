
import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def get_quant_constant(signed, bit_width, scale_factor, acc_reg=False):
    return int(bit_width), int(bit_width+scale_factor)

def get_quant_type(signed, bit_width, scale_factor, acc_reg=False):
    type_name = ""
    type_name += "ap_fixed" if signed else "ap_ufixed"
    type_name += "<"
    type_name += str(int(bit_width))
    type_name += ","
    type_name += str(int(bit_width+scale_factor))
    if acc_reg:
        type_name += ",AP_RND,AP_WRAP>"
    else:
        type_name += ",AP_RND,AP_SAT>"
    # type_name += ",AP_TRN,AP_SAT>"
    return type_name

def info(io_dict, node, node_name, init_info, tensors_info):

    scale_name   = io_dict[node_name]["input"][1]
    scale_info   = init_info[scale_name]
    scale_factor = numpy_helper.to_array(scale_info)
    scale_factor = np.log2(scale_factor)

    attributes = getattr(node, "attribute" )
    narrow = attributes[0].i
    signed = attributes[2].i

    bits_name   = io_dict[node_name]["input"][3]
    bits_info   = init_info[bits_name]
    bits        = numpy_helper.to_array(bits_info)

    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    if len(getattr(input_shape, 'dim')) == 4:
        ich      = getattr(input_shape, 'dim')[1].dim_value
        ih       = getattr(input_shape, 'dim')[2].dim_value
        iw       = getattr(input_shape, 'dim')[3].dim_value
        och      = getattr(output_shape, 'dim')[1].dim_value
        oh       = getattr(output_shape, 'dim')[2].dim_value
        ow       = getattr(output_shape, 'dim')[3].dim_value
    else:
        ich      = getattr(input_shape, 'dim')[0].dim_value
        och      = getattr(output_shape, 'dim')[0].dim_value
        ih       = 1
        iw       = 1
        oh       = 1
        ow       = 1

    io_dict[node_name]["scale_factor"] = scale_factor
    io_dict[node_name]["signed"] = signed
    io_dict[node_name]["narrow"] = narrow
    io_dict[node_name]["bits"] = int(bits)
    io_dict[node_name]["type"] = "quant"
    io_dict[node_name]["clip"] = scale_factor
    io_dict[node_name]["mask"] = scale_factor
    io_dict[node_name]["clip_signed"] = signed
    io_dict[node_name]["mask_signed"] = signed
    io_dict[node_name]["ich"]  = ich
    io_dict[node_name]["ih"]   = ih
    io_dict[node_name]["iw"]   = iw
    io_dict[node_name]["och"]  = och
    io_dict[node_name]["oh"]   = oh
    io_dict[node_name]["ow"]   = ow

    return io_dict