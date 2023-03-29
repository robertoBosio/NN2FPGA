import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def info(io_dict, tensors_info, model):


    node_name = "ProduceStream"

    graph_output_name = model.graph.output[0].name
    output_shape = tensors_info[graph_output_name].tensor_type.shape

    graph_output_name = graph_output_name.replace(".", "_")

    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = [graph_input_name]
    io_dict[node_name]["output"] = [graph_input_name]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'produce'

    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow

    return io_dict

def parse(parsed_write):
    
    input_name  = parsed_write[-1]["args"][-1]
    input_type_name = input_name.replace("_skip", "")
    output_name = input_name.replace("s_", "o_")
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "ConsumeStream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_och" % input_name)
    block["template"].append("c_%s_ow" % input_name)
    block["template"].append("c_%s_oh" % input_name)

    block["args"] = []
    block["args"].append("%s" % input_name)
    block["args"].append("%s" % output_name)

    block["defines"] = {}
    block["defines"]["c_%s" % (output_name)] = ["const", 32]
    block["defines"]["t_%s" % output_type_name] = [
        "type",
        "ap_axiu<c_%s, 0, 0, 0>" % output_name
    ]

    block["output"] = ["%s" % output_name]
    block["declare"] = []

    block["pragma"] = []

    return block

