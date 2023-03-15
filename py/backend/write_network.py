import os
import sys
#import onnx
import backend.main as main
import backend.defines as defines
import backend.memory as memory
import backend.block_design as block_design
import qonnx
from qonnx.transformation import infer_shapes
import backend.memory_header as memory_header
import backend.memory_defines as memory_defines

from backend.quant import *
from backend.graph import *
from backend.opt import *

from onnx import numpy_helper
import numpy as np

def graph_quant_info(model, quant_info):

    skip_connections_info = {}

    skip_connections = []

    general_info = {}

    bias_info = {}

    split_info = {}

    connection_level, diff_level = evaluate_connection_level(model)

    reordered_layers = reorder_layers(model, connection_level)

    # The dictionary reports all the layers which have a different input output
    # structure with respect to the original 1 input stream - 1 output stream
    # architecture
    
    io_dict = {}
    
    inv_quant_info = {}

    for output_name, input_name in quant_info.items():
        inv_quant_info[input_name] = output_name

    # Listing for each layer the input and outputs taking into account the
    # quantization process

    graph_input_name = model.graph.input[0].name

    # Declaring input stream management as a specific node
    # of the network
    io_dict["ProduceStream"] = {}

    io_dict["ProduceStream"]["input"] = [
        graph_input_name
    ]

    if graph_input_name in quant_info.keys():
        io_dict["ProduceStream"]["output"] = [
            quant_info[graph_input_name]
        ]

    # Analyzing the graph to create connections removing quantizations
    for node_level in reordered_layers:
        for node in node_level:

            if 'quant' not in node.name.lower():
                io_dict[node.name] = {}
                io_dict[node.name]["input"] = []
                io_dict[node.name]["output"] = []

                for input in node.input:

                    input_name = input

                    if input in quant_info.keys():
                        input_name = quant_info[input_name]

                    io_dict[node.name]["input"].append(input_name)
                    # Not including the weights parameters
                    if 'conv' in node.name.lower():
                        break

                output_name = node.output[0]
                if node.output[0] in quant_info.keys():
                    output_name = quant_info[output_name]

                io_dict[node.name]["output"].append(output_name)

    io_connect = extract_connections(model, io_dict)

    print(io_dict)
    print(io_connect)
    #Fixing not correct connections
    for net_name, layers in io_connect.items():
        if layers[0] == []:
            print(net_name)

    sys.exit(0)
    return io_dict

def extracts_tensors_info(model):

    tensors_info = {}

    graph_input = model.graph.input
    for input in graph_input:
        tensors_info[input.name] = input.type

    for info in model.graph.value_info:
        tensors_info[info.name] = info.type

    graph_output = model.graph.output
    for output in graph_output:
        tensors_info[output.name] = output.type

    return tensors_info

def extracts_weights_info(model):

    weights_info = {}

    input_info = {}
    for node in model.graph.node:
        # Associating input weight to quant output
        for input in node.input:
            input_info[input] = node.output[0]

    for info in model.graph.initializer:
        if not input_info[info.name] in weights_info.keys():
            weights_info[input_info[info.name]] = {}

        weights_info[input_info[info.name]][info.name] = info
    return weights_info

# Expects an ONNX model
def write_network(
    model,
    off_chip_storage=False
):

    read_width = 8

    inferred_model = model.transform(infer_shapes.InferShapes())

    init_info = {}

    for info in model.graph.initializer:
        init_info[info.name] = info

    io_dict = graph_info(
        inferred_model
    )

    for i in range(10):
        for i in range(10):
            quant_info = extract_quant_info(
                inferred_model,
                io_dict,
                init_info
            )

            io_dict = merge_quant(
                io_dict,
                quant_info
            )

            quant_info = extract_quant_info(
                inferred_model,
                io_dict,
                init_info
            )

            io_dict = opt_quant(
                model,
                io_dict,
                quant_info
            )

            io_dict = opt_relu(
                model,
                io_dict,
            )

        io_dict = opt_add(
            model,
            io_dict,
        )

        io_dict = opt_relu(
            model,
            io_dict,
        )

        io_dict = opt_skip(
            model,
            io_dict,
        )

    # TODO: Check that quantizations are correct and especially why there is
    # a remaining quant module for the downsample
    for name, node in io_dict.items():
        print(name, node)

    sys.exit(0)
    weights_info = extracts_weights_info(inferred_model)

    tensors_info = extracts_tensors_info(inferred_model)

    conv_relu, additional_ports, additional_ports_info, parallel_ops, weights_export, reuse = main.write(
        inferred_model,
        tensors_info,
        quant_info,
        weights_info,
        skip_connections_info,
        bias_info,
        relu_info,
        split_info,
        flatten_info,
        reordered_layers,
        off_chip_storage,
        read_width
    )

    # TODO: export tensors and weight info
    # TODO: assign correct values on tensors to defines
    # TODO: weights define
    
    defines.write(
        inferred_model,
        tensors_info,
        quant_info,
        weights_info,
        skip_connections_info,
        bias_info,
        relu_info,
        conv_relu,
        flatten_info,
        split_info,
        off_chip_storage,
        additional_ports,
        parallel_ops,
        read_width,
        reuse
    )

    if off_chip_storage:

        memory.write(
            additional_ports,
            additional_ports_info,
            read_width
        )

        memory_defines.write(
            additional_ports,
            additional_ports_info,
            read_width
        )

        memory_header.write(
            additional_ports,
            weights_export,
            read_width
        )

        # block_design.write(
        #     additional_ports,
        #     additional_ports_info,
        #     read_width
        # )
