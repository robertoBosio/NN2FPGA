import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.layers.quant import get_quant_type
from backend.layers.layer import Layer

def info(io_dict, tensors_info, graph_input_name, transform=False):
    """ Generate a specific layer block for the input layer. """
    
    node_name = "produce_stream"
    input_shape = tensors_info[graph_input_name].tensor_type.shape
    graph_input_name = graph_input_name.replace(".", "_")

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value

    io_dict[node_name] = {}
    io_dict[node_name]["input"] = [graph_input_name]
    io_dict[node_name]["output"] = [graph_input_name]
    io_dict[node_name]["is_constant"] = False
    io_dict[node_name]["type"] = 'produce'

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = ich
    io_dict[node_name]["oh"]     = ih
    io_dict[node_name]["ow"]     = iw
    io_dict[node_name]["ops"]    = 1
    io_dict[node_name]["ow_ops"] = 1
    io_dict[node_name]["ow_ops_out"] = 1
    io_dict[node_name]["transform"] = transform

    return io_dict

def parse(name, node):
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "produce_stream"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s_part" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    # if node["transform"]:
    #     block["template"].append("t_%s" % output_type_name)
    # else:
    #     block["template"].append("ap_ufixed<8,0,AP_RND_CONV,AP_SAT>")
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s" % input_name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_act_width")
    if node["transform"]:
        block["template"].append({"name":"true", "comment": "transform flag"})
    else:
        block["template"].append({"name":"false", "comment": "transform flag"})


    block["args"] = []
    block["args"].append("i_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    block["input"] = ["%s" % input_name]

    block["defines"] = {}

    ### Defines for testbench 
    block["defines"]["c_act_width"] = [
        "const",
        node["output_quant"]["bits"]
    ]

    #### Version with fixed width but using only part of the input
    # Computing how many ops data it can be inserted in a packet from DMA.
    # width_stream = 64
    # ops_packet = width_stream // (node["bits"][0] * node["ops"])
    # useful_bits = node["bits"][0] * node["ops"] * ops_packet

    #### Version with size of the stream equal to the useful bits
    # width_stream = node["bits"][0] * node["ops"]
    # ops_packet = 1
    # useful_bits = width_stream
    
    #### Version with fixed width and using all the input with modulo index
    width_stream = 64
    useful_bits = width_stream
    act_per_packet = width_stream // node["output_quant"]["bits"]

    block["defines"]["c_data_per_packet"] = [
        "const",
        act_per_packet
    ]
    block["defines"]["c_width_act_stream"] = [
        "const",
        width_stream
    ]
    ### End of defines for testbench

    block["defines"]["c_%s" % input_name] = ["const", useful_bits]
    if input_name != "inp_1":
        block["defines"]["c_inp_1"] = [
            "const", useful_bits]
        block["defines"]["t_inp_1"] = [
            "type",
            "t_%s" % input_name
        ]

    block["defines"]["t_in_mem"] = [
        "type",
        f"ap_uint<{width_stream}>"
    ]
    
    block["defines"]["t_%s" % input_type_name] = [
        "type",
        f"ap_axiu<{width_stream}, 0, 0, 0>"
    ]

    output_type = get_quant_type(node["output_quant"]["signed"], node["output_quant"]["bits"], node["output_quant"]["scale_factor"])

    block["defines"]["t_%s_part" % input_type_name] = [
        "type",
            # "uint8_t"
        output_type
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
    # block["defines"]["c_%s_ow_ops" % name] = [
    #     "const",
    #     node["ow_ops"]
    # ]
    # block["defines"]["c_%s_ow_ops_out" % name] = [
    #     "const",
    #     node["ow_ops_out"]
    # ]

    block["defines"]["c_%s_ops" % name] = [
        "const",
        node["ops"]
    ]

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    # declare["dim"] = node["ow_ops_out"]
    declare["dim"] = 1
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

class InputGenerator(Layer):

    def __init__(self, name, dma_bitwidth):
        super().__init__(name)
        self.foldable = True
        self.dma_bitwidth = dma_bitwidth

