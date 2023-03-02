import os
import sys
#import onnx
import backend.main as main
import backend.defines as defines
import backend.memory as memory
import backend.block_design as block_design
import qonnx
from qonnx.transformation import infer_shapes


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

    return reordered_layers

def extracts_skip_connections_info(model):

    skip_connections_info = {}

    skip_connections = []

    general_info = {}

    bias_info = {}

    split_info = {}

    connection_level, diff_level = evaluate_connection_level(model)

    reordered_layers = reorder_layers(model, connection_level)

    for node in model.graph.node:
        for input in node.input:
            if input not in general_info.keys():
                general_info[input] = []
            general_info[input].append(node)
    # DONT USE SKIP IN MODEL
    for input_name, nodes in general_info.items():
        if len(nodes) > 1:
            #print(nodes)
            if diff_level[input_name] != 0:
                suffix = ""
                for i, node in enumerate(nodes):
                    skip_connections_info[node.name] = []
                    connection_name = input_name + suffix
                    skip_connections_info[node.name].append(connection_name)
                    if i < (len(nodes) - 1):
                        suffix += "_skip"
                        connection_name = input_name + suffix
                        skip_connections_info[node.name].append(connection_name)
                        skip_connections.append(connection_name)
            else:
                split_info[input_name] = []
                for node in nodes:
                    split_info[input_name].append(node.name)

    for node in model.graph.node:
        if 'add' == node.op_type.lower():
            for input in node.input:
                if diff_level[input][0] > 0:
                    skip_name = input + "_skip"
                    if skip_name not in skip_connections:
                        skip_name = input
            for input in node.input:
                if diff_level[input][0] == 0:
                    no_skip_name = input
            for output in node.output:
                out_name = output
            # NO_SKIP_NAME is the output of the branch which has diff_level 0
            # SKIP_NAME is the name of the BIAS that should be applied
            # OUT_NAME is the name of the output to the relu function
            bias_info[no_skip_name] = [skip_name, out_name]

    return skip_connections_info, bias_info, split_info, reordered_layers

def extracts_relu_info(model):

    relu_info = {}

    for node in model.graph.node:
        if 'relu' in node.op_type.lower():
            relu_info[node.input[0]] = [node.name, node.output[0]]

    return relu_info

def extracts_flatten_info(model):

    flatten_info = {}

    for node in model.graph.node:
        if 'flatten' in node.op_type.lower():
            flatten_info[node.input[0]] = [node.name, node.output[0]]

    return flatten_info


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

    print("------------------------------------------------------")
    
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

    inferred_model = model.transform(infer_shapes.InferShapes())

    skip_connections_info, bias_info, split_info, reordered_layers = extracts_skip_connections_info(inferred_model)

    weights_info = extracts_weights_info(inferred_model)

    relu_info = extracts_relu_info(inferred_model)

    flatten_info = extracts_flatten_info(inferred_model)

    tensors_info = extracts_tensors_info(inferred_model)

    conv_relu, additional_ports, additional_ports_info, parallel_ops = main.write(
        inferred_model,
        tensors_info,
        weights_info,
        skip_connections_info,
        bias_info,
        relu_info,
        split_info,
        flatten_info,
        reordered_layers,
        off_chip_storage
    )

    # TODO: export tensors and weight info
    # TODO: assign correct values on tensors to defines
    # TODO: weights define
    
    defines.write(
        inferred_model,
        tensors_info,
        weights_info,
        skip_connections_info,
        bias_info,
        relu_info,
        conv_relu,
        flatten_info,
        split_info,
        off_chip_storage,
        additional_ports,
        parallel_ops
    )

    if off_chip_storage:
        memory.write(
            additional_ports,
            additional_ports_info
        )

        block_design.write(
            additional_ports,
            additional_ports_info
        )
