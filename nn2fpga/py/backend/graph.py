import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import backend.layers.conv as conv
import backend.layers.pool as pool
import backend.layers.input_gen as input_gen

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
            if 'consume_stream' not in node_name:
                node_level = 0
                for input in io_dict[node_name]["input"]:
                    node_level = max([node_level, net_levels[input][0]])
                net_levels[net_name][1].append([node_name, node_level])
        
    return net_levels 

def extract_connections(model, io_dict):

    io_connect = {}

    graph_input_name = model.graph.input[0].name
    graph_input_name = graph_input_name.replace(".", "_")

    graph_output_name = model.graph.output[0].name
    graph_output_name = graph_output_name.replace(".", "_")

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
        
            is_produce_stream = "produce_stream" == node_name
            if (not is_produce_stream) and (input_name in io_connect.keys()):
                    io_connect[input_name][1].append(node_name)

    io_connect[graph_output_name][1] = ["consume_stream"]

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
    
    # Declaring input stream management as a specific node
    # of the network
    io_dict = input_gen.info(
        io_dict,
        tensors_info,
        model
    )

    for node in model.graph.node:

        node_name = node.name
        node_name = node_name.replace(".", "_")
        io_dict[node_name] = {}
        io_dict[node_name]["input"] = []
        io_dict[node_name]["output"] = []

        for input in node.input:

            input_name = input
            input_name = input_name.replace(".", "_")
            io_dict[node_name]["input"].append(input_name)
            node_name_comp = node_name.lower()
            # Not including the weights parameters
            # if ('conv' in node_name_comp):
            #     break

        for output in node.output:
            output_name = output
            output_name = output_name.replace(".", "_")

            io_dict[node_name]["output"].append(output_name)

        io_dict[node_name]["has_forward"] = False
        io_dict[node_name]["merge_1x1"] = False

        if 'conv' in node.op_type.lower():
            io_dict = conv.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info
            )
            continue

        if 'pool' in node.op_type.lower():
            io_dict = pool.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info
            )
            continue

        if 'relu' in node.op_type.lower():
            io_dict[node_name]["type"] = "relu"
            continue

        if 'add' in node.op_type.lower():
            io_dict[node_name]["type"] = "add"
            continue

        if 'quant' in node.op_type.lower():
            scale_name   = io_dict[node_name]["input"][1]
            scale_info   = init_info[scale_name]
            scale_factor = numpy_helper.to_array(scale_info)
            scale_factor = np.log2(scale_factor)

            attributes = getattr(node, "attribute" )
            narrow = attributes[0].i
            signed = attributes[2].i

            bits_name   = io_dict[node_name]["input"][3]
            bits = init_info[bits_name]

            io_dict[node_name]["scale_factor"] = scale_factor
            io_dict[node_name]["signed"] = signed
            io_dict[node_name]["narrow"] = narrow
            io_dict[node_name]["bits"] = numpy_helper.to_array(bits)
            io_dict[node_name]["type"] = "quant"
            io_dict[node_name]["clip"] = scale_factor
            io_dict[node_name]["mask"] = scale_factor
            continue

    return io_dict

def rename_nodes(io_dict):

    new_io_dict = {}
    
    n_node = 0
    for node_name, node in io_dict.items():
            
        new_node_name = "node_" + node["type"] + "_%0d" % n_node
        new_io_dict[new_node_name] = node

        n_node += 1

    return new_io_dict

def rename_edges(model, io_dict):

    io_connect = extract_connections(model, io_dict)

    n_net = 0
    rename_dict = {}
    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        in_type = io_dict[layer_in_name]["type"]

        no_skip_name = net_name.replace("_skip", "")
        if no_skip_name in rename_dict.keys():
            new_net_name = rename_dict[no_skip_name] + "_skip"
        else:
            new_net_name = "net_" + in_type + "_%0d" % n_net

        rename_dict[net_name] = new_net_name

        in_pos = io_dict[layer_in_name]["output"].index(net_name)
        io_dict[layer_in_name]["output"][in_pos] = new_net_name 

        if layer_out_name != "consume_stream":
            out_pos = io_dict[layer_out_name]["input"].index(net_name)
            io_dict[layer_out_name]["input"][out_pos] = new_net_name 

        n_net += 1

    return io_dict
