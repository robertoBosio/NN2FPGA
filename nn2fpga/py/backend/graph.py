import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import backend.layers.conv as conv
import backend.layers.gemm as gemm
import backend.layers.pool as pool
import backend.layers.input_gen as input_gen
import backend.layers.quant as quant
import backend.layers.detect as detect
import backend.layers.non_max_suppression as non_max_suppression
import backend.layers.concat as concat
import backend.layers.upsample as upsample

def compute_branch_length(io_dict, io_connect, layer_name, forward=False):
    branch_length = 0
    branch_found = False
    analyze_layer = layer_name
    if forward:
        print("////////////////////// FORWARD PROPAGATION //////////////////////")
    else:
        print("////////////////////// BACKWARD PROPAGATION //////////////////////")
    print("Checking conv layers", layer_name)

    include_types = ["conv", "add"]

    while not(branch_found) and (io_dict[analyze_layer]["type"] != "produce"):
        # Search backward to find branching layers (i.e. layers with multiple outputs)
        # The previous layer to look is always the first input layer

        # If forward is True, we search forward to find merging layers (i.e. layers with multiple input)
        if forward:
            analyze_net = io_dict[analyze_layer]["output"][0]
            # Then we extract the layer receiving the output from io_connect
            analyze_layer = io_connect[analyze_net][1][0]
            print("Analyze layer", analyze_layer)

            if analyze_layer == "consume_stream":
                branch_length = -1
                break
        else:
            if (len(io_dict[layer_name]["output"]) > 1):
                branch_length = 0
                break
            analyze_net = io_dict[analyze_layer]["input"][0]
            # Then we extract the layer producing the input from io_connect
            analyze_layer = io_connect[analyze_net][0][0]

        if forward:
            # If the analyzed layer has add, we found a merging layer
            if "add" in io_dict[analyze_layer]["type"]:
                branch_found = True
            elif 'add' in io_dict[analyze_layer].keys():
                branch_found = io_dict[analyze_layer]['add']
        else:
            # If the analyzed net has multiple outputs, we found a branching layer
            merged_layer = len(io_dict[analyze_layer]["output"]) > 1
            split_net = len(io_connect[analyze_net][1]) > 1
            print("analyze_layer", analyze_layer)
            if (merged_layer or split_net):
                branch_found = True

        # Only incrementing on conv,add layers
        if io_dict[analyze_layer]["type"] not in include_types:
            continue

        if not branch_found:
            branch_length += 1

    if not branch_found:
        branch_length = -1

    print("Branch length", branch_length, layer_name)
    return branch_length

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

            # Done to recognize vector connections
            output_name = output_name.split("[")[0]

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

def graph_info(model, init_info, object_detection=False, anchors=None, enable_ws=True):

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
        model,
        enable_ws
    )

    graph_output_name = model.graph.output[0].name
    graph_output_name = graph_output_name.replace(".", "_")

    cut_name = []
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
                tensors_info,
                enable_ws
            )
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue

        if 'gemm' in node.op_type.lower():
            io_dict = gemm.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info,
                enable_ws
            )
            last_layer_name = node_name
            continue

        if 'pool' in node.op_type.lower():
            io_dict = pool.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info,
                enable_ws
            )
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue

        if 'relu' in node.op_type.lower():
            io_dict[node_name]["type"] = "relu"
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue

        if 'add' in node.op_type.lower() and cut_name == []:
            io_dict[node_name]["type"] = "add"
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue

        if 'quant' in node.op_type.lower():
            io_dict = quant.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info
            )
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue
        
        if 'flatten' in node.op_type.lower():
            io_dict[node_name]["type"] = "flatten"
            last_layer_name = node_name
            continue

        if 'concat' in node.op_type.lower() and cut_name == []:
            io_dict = concat.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info
            )
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue
        
        if 'resize' in node.op_type.lower() and cut_name == []:
            io_dict = upsample.info(
                io_dict,
                node,
                node_name,
                init_info,
                tensors_info
            )
            # Save last layer name if it is a recognized layer
            last_layer_name = node_name
            continue
        
        if cut_name == []:
            cut_name.append(last_layer_name)
            # Assign the graph output name to the cut layer
            io_dict[last_layer_name]["output"][0] = graph_output_name
        else:
            io_connect = extract_connections(model, io_dict)
            # append the last layer if the input is in the io_connect
            if io_dict[node_name]["input"][0] in io_connect.keys():
                cut_name.append(last_layer_name)
                # Assign the graph output name to the cut layer
                io_dict[last_layer_name]["output"][0] = graph_output_name

        # If the node is not recognized, it is not included in the dictionary
        # and it is not considered in the optimization process
        io_dict.pop(node_name)

    # check if the last layer output is the graph output
    print(cut_name)
    assert (len(cut_name) < 2 or object_detection)
    if len(cut_name) > 0:
        assert (len(cut_name) == len(anchors))

    if object_detection:
        concat_net = None
        for i, layer_name in enumerate(cut_name):
            io_dict = detect.info(io_dict, i, anchors[i], layer_name, len(cut_name), ws)
        io_dict = non_max_suppression.info(io_dict, graph_output_name, len(cut_name), anchors, cut_name)
    elif len(cut_name) == 1:
        graph_output_name = model.graph.output[0].name
        graph_output_name = graph_output_name.replace(".", "_")
        if io_dict[last_layer_name]["output"][0] != graph_output_name:
            io_dict[last_layer_name]["output"][0] = graph_output_name
    
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

        rename_dict[net_name.split("[")[0]] = new_net_name

        # Done to recognize vector connections
        vector_less_name = []
        for output_name in io_dict[layer_in_name]["output"]:
            vector_less_name.append(output_name.split("[")[0])

        in_pos = vector_less_name.index(net_name)
        io_dict[layer_in_name]["output"][in_pos] = new_net_name 
        # In this way splits are handled with the layer merging them

        if len(net_name.split("[")) > 1:
            io_dict[layer_in_name]["output"][in_pos] += "[" + net_name.split("[")[1]

        if layer_out_name != "consume_stream":
            out_pos = io_dict[layer_out_name]["input"].index(net_name)
            io_dict[layer_out_name]["input"][out_pos] = new_net_name 

        n_net += 1

    return io_dict
