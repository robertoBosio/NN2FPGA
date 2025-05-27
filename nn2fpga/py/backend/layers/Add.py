import numpy as np
from onnx import numpy_helper
from backend.utils import sanitize_string
from backend.utils import get_shape_from_value_info

def info(io_dict, node, node_name, init_info, tensors_info):
    input_name_A = node.input[0]
    input_shape_A = get_shape_from_value_info(tensors_info[input_name_A])
    input_name_B = node.input[1]
    input_shape_B = get_shape_from_value_info(tensors_info[input_name_B])
    output_shape = get_shape_from_value_info(tensors_info[node.output[0]])

    if input_shape_A != input_shape_B:
        raise ValueError(f"Input shapes do not match: {input_shape_A} vs {input_shape_B}")
    if input_shape_A != output_shape:
        raise ValueError(f"Input and output shapes do not match: {input_shape_A} vs {output_shape}")
    
    io_dict[node_name]["ich"] = input_shape[1]
    io_dict[node_name]["ih"] = input_shape[2]
    io_dict[node_name]["iw"] = input_shape[3]
    io_dict[node_name]["och"] = output_shape[1]
    io_dict[node_name]["oh"] = output_shape[2]
    io_dict[node_name]["ow"] = output_shape[3]
    io_dict[node_name]["type"] = "upsample"
    io_dict[node_name]["input_quant"] = None
    io_dict[node_name]["output_quant"] = None