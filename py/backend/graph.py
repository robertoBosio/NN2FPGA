import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.layers_info import *

def net_distance(io_dict, io_connect):

    # Finding if the branch has reconvergent fanout, i.e is a skip connection

    net_levels = {}
    max_level = 0
    for net_name, net_info in io_connect.items():
        level = 0
        for node_name in net_info[0]:
            node_level = 0
            for input in io_dict[node_name]["input"]:
                if input not in net_levels.keys():
                    net_levels[input] = [0, []]
                else:
                    node_level = max([node_level, net_levels[input][0] + 1])
            level = max([level, node_level])

        net_levels[net_name] = [level, []]
        if level > max_level:
            max_level = level
    
    for net_name, net_info in io_connect.items():
        for node_name in net_info[1]: 
            if 'ConsumeStream' not in node_name:
                node_level = 0
                for input in io_dict[node_name]["input"]:
                    node_level = max([node_level, net_levels[input][0]])
                net_levels[net_name][1].append([node_name, node_level])
        
    return net_levels 

# def reorder_layers(io_dict, connection_level):

#     # Reordering nodes for optimized residual layer
#     reordered_layers = [[]]
#     for node_name, node in io_dict:
#         node_level = connection_level[node["output"][0]]

#         while node_level >= len(reordered_layers):
#             reordered_layers.append([])

#         reordered_layers[node_level].append(node)


#     # Done for skip connection and stride management
#     reordered_layers_skip = []
#     for node_level in reordered_layers:
#         reordered_level = node_level
#         if len(node_level) != 0: 
#             node_name = node_level[0].name.lower()
#             reordered_level = []
#             cycles = []
#             for node in node_level:
#                 attributes = getattr(node, "attribute")
#                 if ('conv' not in node_name):
#                     c_kernel = 1
#                 else:
#                     c_kernel = getattr(attributes[2], 'ints')[0]**2
#                 cycles.append([node, c_kernel])
#             reordered_level = list(
#                 sorted(cycles, key=lambda i: i[1], reverse=True)
#             )
#             reordered_level = [elem[0] for elem in reordered_level]
#             reordered_layers_skip.append(reordered_level)

#     return reordered_layers_skip

def extract_connections(model, io_dict):

    io_connect = {}

    graph_input_name = model.graph.input[0].name
    graph_output_name = model.graph.output[0].name

    # This list is storing metadata of the connections between the output 
    # streams and the producers
    for node_name, io_info in io_dict.items():
        for output_name in io_info["output"]:

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

    io_connect[graph_output_name][1] = ["ConsumeStream"]

    return io_connect


def extract_tensors_info(model):

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

def graph_info(model, init_info):

    tensors_info = extract_tensors_info(
        model
    )

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

    for node in model.graph.node:

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
        io_dict[node.name]["has_forward"] = False
        io_dict[node.name]["merge_1x1"] = False

        if 'conv' in node.op_type.lower():
            io_dict = conv_info(
                io_dict,
                node,
                init_info,
                tensors_info
            )

        if 'pool' in node.op_type.lower():
            io_dict = pool_info(
                io_dict,
                node,
                init_info,
                tensors_info
            )

        if 'quant' in node.op_type.lower():
            scale_name   = io_dict[node.name]["input"][1]
            scale_info   = init_info[scale_name]
            scale_factor = numpy_helper.to_array(scale_info)
            scale_factor = np.log2(scale_factor)

            io_dict[node.name]["scale_factor"] = scale_factor

    return io_dict

