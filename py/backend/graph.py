import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def evaluate_connection_level(model):
    connection_level = {}
    connection_level[model.graph.input[0].name] = 0

    diff_level = {}
    diff_level[model.graph.input[0].name] = [0, ""]

    for node in model.graph.node:
        node_level = 0
        for input in node.input:
            if input in connection_level.keys():
                if connection_level[input] > node_level:
                    node_level = connection_level[input]

        for input in node.input:
            if input in connection_level.keys():
                diff_level[input] = [node_level - connection_level[input], node]
            else:
                diff_level[input] = 0

        connection_level[node.output[0]] = node_level + 1

    return connection_level, diff_level 

def reorder_layers(model, connection_level):

    # Reordering nodes for optimized residual layer
    reordered_layers = [[]]
    for node in model.graph.node:
        node_level = connection_level[node.output[0]]

        while node_level >= len(reordered_layers):
            reordered_layers.append([])

        reordered_layers[node_level].append(node)


    # Done for skip connection and stride management
    reordered_layers_skip = []
    for node_level in reordered_layers:
        reordered_level = node_level
        if len(node_level) != 0: 
            node_name = node_level[0].name.lower()
            reordered_level = []
            cycles = []
            for node in node_level:
                attributes = getattr(node, "attribute")
                if ('conv' not in node_name):
                    c_kernel = 1
                else:
                    c_kernel = getattr(attributes[2], 'ints')[0]**2
                cycles.append([node, c_kernel])
            reordered_level = list(sorted(cycles, key=lambda i: i[1], reverse=True))
            reordered_level = [elem[0] for elem in reordered_level]
            reordered_layers_skip.append(reordered_level)

    return reordered_layers_skip

def extract_connections(model, io_dict):

    io_connect = {}

    graph_input_name = model.graph.input[0].name
    graph_output_name = model.graph.output[0].name

    # This list is storing metadata of the connections between the output 
    # streams and the producers
    for node_name, io_info in io_dict.items():
        output_name = io_info["output"][0]

        is_graph_input = output_name == graph_input_name
        io_connect.setdefault(output_name, [[], []])

        io_connect[output_name][0].append(node_name)

    # Adding to the previous list the connections to the consumers
    # This is done to isolate the connections which should be optimized in
    # terms of skip connections
    for node_name, io_info in io_dict.items():
        for input_name in io_info["input"]:
        
            is_produce_stream = "ProduceStream" == node_name
            if (not is_produce_stream) and (input_name in io_connect.keys()):
                    io_connect[input_name][1].append(node_name)

    io_connect[graph_output_name][1] = "ConsumeStream"

    return io_connect


def graph_info(model):

    connection_level, diff_level = evaluate_connection_level(model)

    reordered_layers = reorder_layers(model, connection_level)

    # The dictionary reports all the layers which have a different input output
    # structure with respect to the original 1 input stream - 1 output stream
    # architecture
    
    io_dict = {}
    
    # Listing for each layer the input and outputs taking into account the
    # quantization process

    graph_input_name = model.graph.input[0].name

    # Declaring input stream management as a specific node
    # of the network
    io_dict["ProduceStream"] = {}
    io_dict["ProduceStream"]["input"] = [graph_input_name]
    io_dict["ProduceStream"]["output"] = [graph_input_name]
    io_dict["ProduceStream"]["is_constant"] = False

    for node_level in reordered_layers:
        for node in node_level:

            io_dict[node.name] = {}
            io_dict[node.name]["input"] = []
            io_dict[node.name]["output"] = []

            for input in node.input:

                input_name = input

                io_dict[node.name]["input"].append(input_name)
                node_name_comp = node.name.lower()
                # Not including the weights parameters
                # if ('conv' in node_name_comp):
                #     break

            output_name = node.output[0]

            io_dict[node.name]["output"].append(output_name)

    return io_dict

