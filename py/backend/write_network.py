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

def extracts_skip_connections_info(model, quant_info):

    skip_connections_info = {}

    skip_connections = []

    general_info = {}

    bias_info = {}

    split_info = {}

    connection_level, diff_level = evaluate_connection_level(model)

    reordered_layers = reorder_layers(model, connection_level)

    last_conv = None
    for node_level in reordered_layers:
        for node in node_level:
            if 'conv' in node.name.lower():
                last_conv = node
            for input in node.input:
                if input not in general_info.keys():
                    general_info[input] = []
                if 'quant' in node.name.lower():
                    general_info[input].append(last_conv)
                else:
                    general_info[input].append(node)

    
    print(general_info)

    # DONT USE SKIP IN MODEL
    for input_name, nodes in general_info.items():
        if len(nodes) > 1:
            if input_name in quant_info.keys():
                input_name = quant_info[input_name]
            # print(nodes)
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
                split_info[input_name] = [[], []]
                tensor_name = input_name.replace(".", "_")
                tensor_name = input_name.lower().replace("onnx::", "")
                for i, node in enumerate(nodes):
                    split_info[input_name][0].append(tensor_name + "_skip%0d" % i)
                    split_info[input_name][1].append(node.name)
                    if (i < (len(nodes)-1)):
                        skip_connections_info[node.name] = []
                        connection_name = input_name + "_skip%0d" % (i+1)
                        skip_connections_info[node.name].append(tensor_name)
                        skip_connections_info[node.name].append(connection_name)

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

    print(skip_connections_info)
    print(sys.exit(0))

    return skip_connections_info, bias_info, split_info, reordered_layers

def extracts_skip_connections_info(model, quant_info):

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
    
    for node_level in reordered_layers:
        for node in node_level:
            # if 'quant' in node.name.lower():
                # if node.input[0] == model.graph.input[0].name:
                #     io_dict[node.name] = [[], []]
                #     input_name = node.input[0]
                #     output_name = node.output[0]
                #     io_dict[node.name][0].append(input_name)
                #     io_dict[node.name][1].append(output_name)

            if 'quant' not in node.name.lower():
                io_dict[node.name] = [[], []]

                input_name = node.input[0]
                io_dict[node.name][0].append(input_name)

                output_name = node.output[0]
                if node.output[0] in quant_info.keys():
                    output_name = quant_info[output_name]

                io_dict[node.name][1].append(output_name)

    io_connect = {}

    # This list is storing metadata of the connections between the output 
    # streams and the producers
    for node_name, io_info in io_dict.items():
        output_name = io_info[1][0]

        io_connect.setdefault(output_name, [[], []])

        io_connect[output_name][1].append(node_name)

    inv_quant_info = {}

    for output_name, input_name in quant_info.items():
        inv_quant_info[input_name] = output_name

    print(quant_info)
    print(io_connect)

    # Adding to the previous list the connections to the consumers
    for node_name, io_info in io_dict.items():
        input_name = io_info[0][0]
        
        if input_name in inv_quant_info.keys():
            input_name = inv_quant_info[input_name]

        if input_name in io_connect.keys():
            io_connect[input_name][0].append(node_name)


    opt_skip_info = {}

    # for node_name, io_info in io_dict.items():

    print(io_connect)
    print(sys.exit(0))


    return skip_connections_info, bias_info, split_info, reordered_layers

def extracts_relu_info(model):

    relu_info = {}

    for node in model.graph.node:
        if 'relu' in node.op_type.lower():
            relu_info[node.input[0]] = [node.name, node.output[0]]

    return relu_info

def extracts_quant_info(model):

    quant_info = {}

    last_input = None
    last_output = None
    for node in model.graph.node:
        if 'quant' in node.op_type.lower():
            quant_info[node.input[0]] = node.output[0]
            if last_output is not None:
                if last_output == node.input[0]:
                    # Assuming equal quantization between consecutive quant
                    # nodes
                    quant_info[last_input] = node.output[0]
            last_input = node.input[0]
            last_output = node.output[0]

    return quant_info

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

    read_width = 8

    quant_info = extracts_quant_info(inferred_model)

    skip_connections_info, bias_info, split_info, reordered_layers = extracts_skip_connections_info(inferred_model, quant_info)

    weights_info = extracts_weights_info(inferred_model)

    relu_info = extracts_relu_info(inferred_model)

    flatten_info = extracts_flatten_info(inferred_model)

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
