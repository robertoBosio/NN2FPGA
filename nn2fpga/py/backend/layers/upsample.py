import numpy as np
from onnx import numpy_helper

def sanitize_string(string):
    return string.replace(".", "_")

def get_shape_from_type(tensor_type):
    return [d.dim_value for d in tensor_type.shape.dim]

def info(io_dict, node, node_name, init_info, tensors_info):
    input_name = node.input[0]
    input_shape = get_shape_from_type(tensors_info[input_name].tensor_type)
    output_shape = get_shape_from_type(tensors_info[node.output[0]].tensor_type)

    # Handle roi, scales, sizes
    roi = scales = sizes = None

    if len(node.input) > 1 and sanitize_string(node.input[1]) in init_info:
        roi = numpy_helper.to_array(init_info[sanitize_string(node.input[1])])
    if len(node.input) > 2 and sanitize_string(node.input[2]) in init_info:
        scales = numpy_helper.to_array(init_info[sanitize_string(node.input[2])])
    if len(node.input) > 3 and sanitize_string(node.input[3]) in init_info:
        sizes = numpy_helper.to_array(init_info[sanitize_string(node.input[3])])
    
    # Basic validation
    if scales is not None and sizes is not None:
        raise ValueError("Resize node should provide either 'scales' or 'sizes', not both.")

    if scales is not None:
        # Only handle 2D upsampling
        assert len(scales) == 4, f"Unexpected scales length: {len(scales)}"
        assert scales[0] == 1 and scales[1] == 1, "Only spatial scaling supported"
        assert scales[2] == scales[3], "Only uniform scale supported"
        upsample_factor = int(scales[2])
    elif sizes is not None:
        # Infer scale factor from input/output shape
        assert len(sizes) == 4, f"Unexpected sizes length: {len(sizes)}"
        scale_h = sizes[2] / input_shape[2]
        scale_w = sizes[3] / input_shape[3]
        assert scale_h == scale_w, "Only uniform scaling supported"
        upsample_factor = int(scale_h)
    else:
        raise ValueError("Resize node must provide either 'scales' or 'sizes'")

    io_dict[node_name] = {
        "ich": input_shape[1],
        "ih": input_shape[2],
        "iw": input_shape[3],
        "och": output_shape[1],
        "oh": output_shape[2],
        "ow": output_shape[3],
        "factor": upsample_factor,
        "type": "upsample",
        "scale_factor": 0,
    }

    return io_dict

def parse(parsed_write, node_name):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "upsample_op"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("c_%s_ich" % node_name)
    block["template"].append("c_%s_ih" % node_name)
    block["template"].append("c_%s_iw" % node_name)
    block["template"].append("c_%s_factor" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]
    block["defines"]["c_%s_ich" % node_name] = ["const", node["ich"]]
    block["defines"]["c_%s_ih" % node_name] = ["const", node["ih"]]
    block["defines"]["c_%s_iw" % node_name] = ["const", node["iw"]]
    block["defines"]["c_%s_factor" % node_name] = ["const", node["factor"]]

    if node["scale_factor"] is list:
        scale_factor = node["scale_factor"][0]
    else:
        scale_factor = node["scale_factor"]

    block["defines"]["c_%s_scale_factor" % name] = [
        "const",
        scale_factor
    ]

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)

    block["pragma"] = []

    return block



