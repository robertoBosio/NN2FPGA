import numpy as np
from onnx import numpy_helper

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )
    axis = attributes[0].i

    input_shapes = [
        tensors_info[input].tensor_type.shape for input in node.input
    ]
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    # Changing names of the variables in input to concat
    # To organize them in a vector that will be used in the concat function
    input_vector_name = "s_%s_net" % node_name
    for i, input in enumerate(node.input):
        for name_iter, node_iter in io_dict.items():
            if input in node_iter["output"]:
                index = node_iter["output"].index(input)
                io_dict[name_iter]["output"][index] = "%s" % ( node_name + "_" + str(i))

            if input in node_iter["input"]:
                index = node_iter["input"].index(input)
                io_dict[name_iter]["input"][index] = "%s" % ( node_name + "_" + str(i))

    ich = []
    ih = 1
    iw = 1
    och = 1
    oh = 1
    ow = 1
    for input_shape in input_shapes: 
        i_shape = [dim.dim_value for dim in input_shape.dim]
        print(i_shape)
        if len(i_shape) > 1:
            ich.append(i_shape[1])
        if len(i_shape) > 2:
            ih = i_shape[2]
        if len(i_shape) > 3:
            iw = i_shape[3]
    o_shape = [dim.dim_value for dim in output_shape.dim]
    if len(o_shape) > 1:
        och = o_shape[1]
    if len(o_shape) > 2:
        oh = o_shape[2]
    if len(o_shape) > 3:
        ow = o_shape[3]

    if (axis < 0):
        axis = len(o_shape) + axis
    feature_map = np.prod([dim for i, dim in enumerate(o_shape) if i != axis])

    io_dict[node_name]["ich"]    = np.asarray(ich)
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["type"]   = 'concat'
    io_dict[node_name]["feature_map"] = feature_map
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["input_quant"] = None

    return io_dict

def get_input_name(node):
    for input_name in node["input"][0]:
            return input_name
        
def get_output_name(node):
    for output_name in node["output"][0]:
            return output_name
        
def parse(parsed_write, node_name):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "concat_op"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("c_%s_feature_map" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("c_%s_ich" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]
    block["defines"]["c_%s_feature_map" % node_name] = ["const", node["feature_map"]]

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

    tmp = {}
    tmp["name"] = "c_%s_ich" % name
    tmp["type"] = "int" % name
    tmp["is_array"] = True
    tmp["is_const"] = True
    size = node["ich"].shape
    tmp["size"] = size
    tmp["init"] = node["ich"]

    block["declare"].append(tmp)

    block["pragma"] = []

    return block


