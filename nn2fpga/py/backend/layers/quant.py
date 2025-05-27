import os
import sys
import qonnx
import numpy as np
from onnx import numpy_helper
from backend.utils import sanitize_string
from backend.utils import get_shape_from_value_info
from backend.utils import extract_attrs

def get_quant_constant(signed, bit_width, scale_factor, acc_reg=False):
    return int(bit_width), int(bit_width+scale_factor)

def get_quant_type(signed, bit_width, scale_factor, acc_reg=False, narrow=False):
    """ Return the quantized type name in Vitis HLS format """

    type_name = ""
    type_name += "ap_fixed" if signed else "ap_ufixed"
    type_name += "<"
    type_name += str(int(bit_width))
    type_name += ", "
    type_name += str(int(bit_width + scale_factor))
    if acc_reg:
        type_name += ", AP_RND_ZERO, AP_WRAP>"
    else:
        if narrow:
            type_name += ", AP_RND_ZERO, AP_SAT_SYM>"
        else:
            type_name += ", AP_RND_CONV, AP_SAT>"
    return type_name

def info(io_dict, node, node_name, init_info, tensors_info):
    """ Extract information from the IntQuant node.
    From qonnx specification 0.4.0, IntQuant node has the following inputs:
    - X: the input tensor to be quantized
    - scale : a tensor containing the scale factor for quantization
    - zeropt: a tensor containing the zero-point for quantization
    - bitwidth: the bit width for quantization
    Attributes:
    - signed: whether the quantization is signed or not
    - narrow: whether the quantization is narrow or not
    - rounding_mode: the rounding mode for quantization
    Outputs:
    - Y: the quantized output tensor
    """

    input_name = node.input[0]
    input_shape = get_shape_from_value_info(tensors_info[input_name])
    output_shape = get_shape_from_value_info(tensors_info[node.output[0]])
    
    # Handle scale, zeropt, bitwidth
    scale = zeropt = bitwidth = None

    if len(node.input) > 1 and sanitize_string(node.input[1]) in init_info:
        scale = numpy_helper.to_array(init_info[sanitize_string(node.input[1])])
    if len(node.input) > 2 and sanitize_string(node.input[2]) in init_info:
        zeropt = numpy_helper.to_array(init_info[sanitize_string(node.input[2])])
    if len(node.input) > 3 and sanitize_string(node.input[3]) in init_info:
        bitwidth = numpy_helper.to_array(init_info[sanitize_string(node.input[3])])
    
    # Basic validation
    if scale is None or zeropt is None or bitwidth is None:
        raise ValueError("Scale, zeropt and bitwidth initializers must all be provided")

    # Right now only per tensor quantization is supported.
    if scale.ndim > 1:
        raise ValueError(f"Scale must be a scalar, got array with shape {scale.shape}")
    if zeropt.ndim > 1:
        raise ValueError(f"Zero-point must be a scalar, got array with shape {zeropt.shape}")
    if bitwidth.ndim > 1:
        raise ValueError(f"Bitwidth must be a scalar, got array with shape {bitwidth.shape}")

    scale_val    = float(scale.item())
    zeropt_val   = int(zeropt.item())
    bitwidth_val = int(bitwidth.item())

    # Only symmetric quantization is supported.
    if zeropt_val != 0:
        raise ValueError(f"Only symmetric quantization supported: zero-point must be 0, got {zeropt_val}")

    # Only power of two scales are supported.
    log2_scale = np.log2(scale_val)
    rounded_log2_scale = round(log2_scale)
    if not np.isclose(log2_scale, rounded_log2_scale, atol=1e-6):
        print(f"Warning: scale={scale_val} of IntQuant node {node_name} is not a power of two. Approximating from {log2_scale} to {rounded_log2_scale}.", file=sys.stderr)

    scale_factor = rounded_log2_scale
    bitwidth = min(bitwidth_val, 16)  # Limit bitwidth to 16 bits
    
    attrs  = extract_attrs(node, {"narrow", "signed", "rounding_mode"})
    narrow = attrs.get("narrow", None).i if "narrow" in attrs else 0
    signed = attrs.get("signed", None).i if "signed" in attrs else 0

    io_dict[node_name]["narrow"] = narrow
    io_dict[node_name]["type"] = "quant"
    io_dict[node_name]["scale_factor"] = scale_factor
    io_dict[node_name]["signed"] = signed
    io_dict[node_name]["bits"] = bitwidth
    io_dict[node_name]["clip"] = scale_factor
    io_dict[node_name]["mask"] = scale_factor
    io_dict[node_name]["clip_signed"] = signed
    io_dict[node_name]["mask_signed"] = signed
    io_dict[node_name]["clip_bits"] = bitwidth
    io_dict[node_name]["mask_bits"] = bitwidth
    io_dict[node_name]["ich"]  = input_shape[1]
    io_dict[node_name]["ih"]   = input_shape[2]
    io_dict[node_name]["iw"]   = input_shape[3]
    io_dict[node_name]["och"]  = output_shape[1]
    io_dict[node_name]["oh"]   = output_shape[2]
    io_dict[node_name]["ow"]   = output_shape[3]
    io_dict[node_name]["data_type"] = get_quant_type(signed, bitwidth, scale_factor, acc_reg=False, narrow=narrow)

    return io_dict