import numpy as np
from onnx import numpy_helper
from backend.layers.quant import get_quant_type
import math
def sanitize_string(string):
    return string.replace(".", "_")

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute")

    input_shape = tensors_info[node.input[1]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["fh"]     = 1
    io_dict[node_name]["fw"]     = 1
    io_dict[node_name]["depth"]  = 1
    io_dict[node_name]["ich_ops"] = 1
    io_dict[node_name]["ow_ops"] = 1
    io_dict[node_name]["ow_ops_out"] = 1
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["adjust_line_buffer"] = False
    io_dict[node_name]["type"]   = 'gather'
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["output_quant"] = None
    io_dict[node_name]["input_quant"] = None
    io_dict[node_name]["stride"] = 1
    io_dict[node_name]["pad"] = 0

    return io_dict
def get_input_name(node):
    input_name = node["input"][1]
    return input_name
def get_output_name(node):
    return node["output"][0]

def parse(node_name, node):
    
    #substitute gather with silu in node name
    node_name = node_name.replace("gather", "silu") 
    input_name  = get_input_name(node)
    input_type_name = input_name.replace("_skip", "")
    # if node["adjust_line_buffer"]:
    #     input_type_name = input_name +"_adj"
        
    output_name = get_output_name(node)
    output_type_name = output_name.replace("_skip", "")
    
    block = {}
    block["func"] = "silu_op"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_vector" % input_type_name)
    block["template"].append("t_%s_vector" % output_type_name)
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("c_%s_ow_ops" % node_name)
    block["template"].append("c_%s_ops" % node_name)
    block["template"].append("c_%s_ich" % node_name)
    block["template"].append("c_%s_ih" % node_name)
    block["template"].append("c_%s_iw" % node_name)
    block["template"].append("c_%s_prec_bits" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    
    output_type = get_quant_type(node["output_quant"]["signed"], node["output_quant"]["bits"], node["output_quant"]["scale_factor"])
    block["defines"]["t_%s" % output_type_name] = ["type", output_type]
    output_type_vector = "std::array<t_%s, %0d>" % (output_type_name, node["ow_ops"])
    block["defines"]["t_%s_vector" % output_type_name] = ["type", output_type_vector]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, %0d>" % (output_type_name, node["ow_ops"])], ["last", "bool"]]
    ]
    block["defines"]["c_%s_ich" % node_name] = ["const", node["ich"]]
    block["defines"]["c_%s_ih" % node_name] = ["const", node["ih"]]
    block["defines"]["c_%s_iw" % node_name] = ["const", node["iw"]]
    block["defines"]["c_%s_ow_ops" % node_name] = ["const", node["ow_ops"]]
    block["defines"]["c_%s_ops" % node_name] = ["const", node["ops"]]
    block["defines"]["c_%s_prec_bits" % node_name] = ["const", int(node["output_quant"]["bits"] + node["output_quant"]["scale_factor"])]
    


    block["output"] = []
    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ow_ops"]
    block["declare"].append(declare)

    depth = node["ich"] + 1

    block["pragma"] = []
    pragma = {}
    pragma["name"] = "stream"
    pragma_name = "s_%s" % (output_name)
    options = [
        ["variable", pragma_name],
        ["depth", depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    return block



