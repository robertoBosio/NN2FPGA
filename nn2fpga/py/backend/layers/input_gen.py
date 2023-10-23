import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.layers.quant import get_quant_type

def info(io_dict, tensors_info, model, ws):


    node_name = "produce_stream"

    graph_input_name = model.graph.input[0].name
    input_shape = tensors_info[graph_input_name].tensor_type.shape

    graph_input_name = graph_input_name.replace(".", "_")

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value
    # print(ich, ih, iw)
    # exit(0)

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = [graph_input_name]
    io_dict[node_name]["output"] = [graph_input_name]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'produce'

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["enable_ws"] = ws
    io_dict[node_name]["ws_out"]     = 1
    io_dict[node_name]["ops"]     = 1

    return io_dict

def parse(name, node):

     
    vitis_flow = False
    if "VITIS_FLOW" in os.environ:
        if int(os.environ.get("VITIS_FLOW")) == 1:
            vitis_flow = True
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    signed = node["signed"]

    block = {}
    block["func"] = "produce_stream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_part" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_ws_out" % name)
    block["template"].append("c_%s" % input_name)
    block["template"].append("%0d" % node["ops"])
    # block["template"].append("c_ws")

    block["args"] = []
    block["args"].append("i_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["input"] = ["%s" % input_name]

    block["defines"] = {}
    block["defines"]["c_%s" % input_name] = ["const", 64]

    block["defines"]["t_in_mem"] = [
        "type",
        "ap_uint<c_%s>" % input_name
    ]
    
    block["defines"]["t_%s" % input_type_name] = [
        "type",
        "ap_axiu<c_%s, 0, 0, 0>" % input_name
    ]

    output_type = get_quant_type(node["signed"], node["bits"][0], node["scale_factor"][0])

    block["defines"]["t_%s_part" % input_type_name] = [
        "type",
        "uint8_t"
    ]
    block["defines"]["t_%s" % output_type_name] = [
        "type",
        output_type
    ]
    output_vector_type = "std::array<t_%s, %0d>" % (output_type_name, node["ops"])
    block["defines"]["t_%s_vector" % output_type_name] = [
        "type",
        output_vector_type
    ]
    block["defines"]["t_%s_struct" % output_type_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, 1>" % output_type_name], ["last", "bool"]]
    ]

    block["defines"]["c_produce_stream_ich"] = [
        "const",
        node["ich"]
    ]
    block["defines"]["c_produce_stream_iw"] = [
        "const",
        node["iw"]
    ]
    block["defines"]["c_produce_stream_ih"] = [
        "const",
        node["ih"]
    ]

    block["defines"]["c_%s_ich" % name] = [
        "const",
        node["ich"]
    ]
    block["defines"]["c_%s_iw" % name] = [
        "const",
        node["iw"]
    ]
    block["defines"]["c_%s_ih" % name] = [
        "const",
        node["ih"]
    ]
    block["defines"]["c_%s_ws_out" % name] = [
        "const",
        node["ws_out"]
    ]

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ws_out"]
    block["declare"].append(declare)

    block["pragma"] = []

    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s" % (output_name)],
        # ["depth", node["ich"]],
        ["depth", node["ich"]],
        # ["depth", 2],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    pragma = {}
    pragma["name"] = "interface"
    options = [
        ["port", "i_%s" % (input_name)],
        ["mode", "axis"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    # Adding here dataflow pragma for top dataflow
    pragma = {}
    pragma["name"] = "dataflow"
    options = [["disable_start_propagation"]]
    pragma["options"] = options
    block["pragma"].append(pragma)

    return block

