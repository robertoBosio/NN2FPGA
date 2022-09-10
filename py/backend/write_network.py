import os
import sys
import onnx
import backend.main as main
import backend.defines as defines

def extracts_skip_connections_info(model):

    skip_connections_info = {}

    general_info = {}

    for node in model.graph.node:
        for input in node.input:
            if input not in general_info.keys():
                general_info[input] = []
            general_info[input].append(node.name)

    # DONT USE SKIP IN MODEL
    for input_name, nodes in general_info.items():
        if len(nodes) > 1:
            suffix = ""
            for i, node in enumerate(nodes):
                skip_connections_info[node] = []
                connection_name = input_name + suffix
                skip_connections_info[node].append(connection_name)
                if i < (len(nodes) - 1):
                    suffix += "_skip"
                    connection_name = input_name + suffix
                    skip_connections_info[node].append(connection_name)

    return skip_connections_info

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

    for info in model.graph.initializer:
        weights_info[info.name] = info

    return weights_info

# Expects an ONNX model
def write_network(model):

    inferred_model = onnx.shape_inference.infer_shapes(model)

    skip_connections_info = extracts_skip_connections_info(model)

    weights_info = extracts_weights_info(model)

    main.write(
        inferred_model,
        weights_info,
        skip_connections_info
    )

    # TODO: export tensors and weight info
    # TODO: assign correct values on tensors to defines
    # TODO: weights define
    
    tensors_info = extracts_tensors_info(inferred_model)

    defines.write(
        inferred_model,
        tensors_info,
        weights_info,
        skip_connections_info
    )
