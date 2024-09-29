import numpy as np
from onnx import numpy_helper
from backend.quant import *
from backend.graph import *

def opt_flatten(io_dict, model, log=False):
    """ Optimize flatten layers by removing them and connecting the input and output layers. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the flatten layers
    flatten_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "flatten"]

    for layer_name in flatten_layers:
            
        remove_and_bypass_layer(io_dict, io_connect, layer_name)

        if log:
            print(f"Flatten layer \"{layer_name}\" removed.")

    return io_dict

def opt_pad(model, io_dict, log=False):
    """ Optimize padding layers by merging them with the next computation layer. """
    
    # Layers that support padding
    comp_layers = ["conv", "pool"]
    
    io_connect = extract_connections(model, io_dict)

    # Retrieve all the padding layers
    pad_layers = [layer_name for layer_name, layer_info in io_dict.items() if "pad" in layer_info["type"]]

    for layer_name in pad_layers:

        input_layer_name = prev_layers(io_dict, io_connect, layer_name)
        output_layer_name = next_layers(io_dict, io_connect, layer_name)

        if input_layer_name is None or output_layer_name is None or len(input_layer_name) > 1 or len(output_layer_name) > 1:
            print(f"Error in opt_pad: padding layer \"{layer_name}\" with multiple inputs or outputs.")
            exit(1)

        if io_dict[output_layer_name]["type"].lower() in comp_layers:

            io_dict[output_layer_name]["pad"] = io_dict[input_layer_name]["pad"]
            remove_and_bypass_layer(io_dict, io_connect, layer_name)
            
            if log:
                print(f"Padding layer \"{layer_name}\" merged with \"{output_layer_name}\".")

    return io_dict

def opt_relu(model, io_dict, log=False):
    """ Optimize relu layers by merging them with the previous computation layer. """
    
    # Layers that supports activation functions
    comp_layers = ["conv"]

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the relu layers
    relu_layers = [layer_name for layer_name, layer_info in io_dict.items() if "relu" in layer_info["type"]]

    for layer_name in relu_layers:
        
        input_layer_name = prev_layers(io_dict, io_connect, layer_name)

        if input_layer_name is None or len(input_layer_name) > 1:
            print(f"Error in opt_relu: relu layer \"{layer_name}\" with multiple inputs.")
            exit(1)

        input_layer_name = input_layer_name[0]
        if io_dict[input_layer_name]["type"].lower() in comp_layers:

            io_dict[input_layer_name]["relu"] = True
            if io_dict[layer_name]["output_quant"] is not None:
                io_dict[input_layer_name]["output_quant"] = io_dict[layer_name]["output_quant"]
            remove_and_bypass_layer(io_dict, io_connect, layer_name)
            
            if log:
                print(f"Relu layer \"{layer_name}\" merged with \"{input_layer_name}\".")

    return io_dict

def opt_add(model, io_dict, log=False):
    """ Optimize add layers by merging them with the previous computation layer. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the add layers
    add_layers = [layer_name for layer_name, layer_info in io_dict.items() if "add" in layer_info["type"]]
    add_layers.sort(key=lambda x: io_dict[x]["layer_index"], reverse=True)

    for layer_name in add_layers:

        input_net_names = io_dict[layer_name]["input"]

        if input_net_names is None or len(input_net_names) != 2:
            print(f"Error in opt_add: add layer \"{layer_name}\" without exactly two inputs.")
            exit(-1)

        first_operand_net = input_net_names[0]
        first_operand_name = io_connect[first_operand_net][0][0]
        first_operand = io_dict[first_operand_name]
        second_operand_net = input_net_names[1]
        second_operand_name = io_connect[second_operand_net][0][0]
        second_operand = io_dict[second_operand_name]

        # Detach the short branch input from the add layer and attach it to the merged conv
        first_operand_index = first_operand["layer_index"]
        second_operand_index = second_operand["layer_index"]

        # Merge with the higher index layer
        layer_merge = first_operand
        layer_merge_net = second_operand_net
        layer_merge_name = first_operand_name
        if first_operand_index < second_operand_index:
            layer_merge = second_operand
            layer_merge_net = first_operand_net
            layer_merge_name = second_operand_name
        
        if layer_merge["type"].lower() == "conv":
            layer_merge["add"] = True
            layer_merge["add_node"] = io_dict[layer_name]
            layer_merge["add_node"]["name"] = layer_name

            # Substitute output quant with add quant
            if io_dict[layer_name]["output_quant"] is not None:
                layer_merge["output_quant"] = io_dict[layer_name]["output_quant"]

            layer_merge["input"].append(layer_merge_net)
            io_dict[layer_name]["input"].remove(layer_merge_net)
            io_connect = extract_connections(model, io_dict)
            remove_and_bypass_layer(io_dict, io_connect, layer_name)
            
            if log:
                print(f"Add layer \"{layer_name}\" merged with \"{layer_merge_name}\".")

    return io_dict

def fold_params_quant(model, io_dict, init_info, log=False):
    """ Fold weights/biases quantization layers inside convolution layers."""

    io_connect = extract_connections(model, io_dict)

    conv_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "conv"]
    for layer_name in conv_layers:

        # Retrieve nets name connected in input to conv layers
        input_nets = io_dict[layer_name]["input"].copy()

        if len(input_nets) > 1:
            weight_net_name = input_nets[1]
            weight_quant_name = io_connect[weight_net_name][0][0]
            weight_input_name = io_dict[weight_quant_name]["input"][0]
            values = numpy_helper.to_array(init_info[weight_input_name]) 
            if values.ndim == 2:
                values = np.expand_dims(values, axis=-1)
                values = np.expand_dims(values, axis=-1)
            io_dict[layer_name]["weight_quant"] = io_dict[weight_quant_name]
            io_dict[layer_name]["weight_quant"]["name"] = weight_quant_name
            io_dict[layer_name]["weight_quant"]["values"] = values
            io_dict[layer_name]["input"].remove(weight_net_name)

            del io_dict[weight_quant_name]
            if log:
                print(f"Folding weight quant {weight_quant_name} into layer {layer_name} {np.prod(values.shape)}")
        
        if len(input_nets) > 2:
            bias_net_name = input_nets[2]
            bias_quant_name = io_connect[bias_net_name][0][0]
            bias_input_name = io_dict[bias_quant_name]["input"][0]
            values = numpy_helper.to_array(init_info[bias_input_name]) 
            if values.ndim == 1:
                values = np.expand_dims(values, axis=-1)
            io_dict[layer_name]["bias_quant"] = io_dict[bias_quant_name]
            io_dict[layer_name]["bias_quant"]["name"] = bias_quant_name
            io_dict[layer_name]["bias_quant"]["values"] = values
            io_dict[layer_name]["input"].remove(bias_net_name)
            del io_dict[bias_quant_name]
            if log:
                print(f"Folding bias quant {bias_quant_name} into layer {layer_name} {np.prod(values.shape)}")
    
    return io_dict

def fold_output_quant(model, io_dict, log=False):
    """ Fold quantization layers inside input layers. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the quantization layers, excluding weights and biases
    quant_layers = [layer_name for layer_name, layer_info in io_dict.items() if "quant" == layer_info["type"]]
    quant_layers_folding = []

    for layer_name in quant_layers:

        input_layer_name = prev_layers(io_dict, io_connect, layer_name)

        if input_layer_name is None:
            if log:
                print(f"Quantization layer \"{layer_name}\" has no input, skipping folding.")
            continue

        if len(input_layer_name) > 1:
            if log:
                print(f"Quantization layer \"{layer_name}\" has multiple inputs, skipping folding.")
            continue

        input_layer_name = input_layer_name[0]
        if io_dict[input_layer_name]["type"].lower() in ["conv", "pool", "add", "relu", "produce"]:

            quant_layers_folding.append((layer_name, input_layer_name))
            if log:
                print(f"Quantization layer \"{layer_name}\" folded with \"{input_layer_name}\".")

    for layer_name, input_layer_name in quant_layers_folding:
        io_dict[input_layer_name]["output_quant"] = io_dict[layer_name]
        io_dict[input_layer_name]["output_quant"]["name"] = layer_name
        if io_dict[input_layer_name]["type"].lower() == "conv":
            io_dict[input_layer_name]["conv_output_quant"] = io_dict[layer_name]
        print(f"Output quant {layer_name} folded into {input_layer_name}")
        remove_and_bypass_layer(io_dict, io_connect, layer_name)
            
    return io_dict

def fold_input_quant(model, io_dict, log=False):
    """ Fold quantization layers inside output layers. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the quantization layers, excluding weights and biases
    quant_layers = [layer_name for layer_name, layer_info in io_dict.items() if "quant" == layer_info["type"]]
    quant_layers_folding = []

    for layer_name in quant_layers:

        output_layer_name = next_layers(io_dict, io_connect, layer_name)

        if output_layer_name is None:
            if log:
                print(f"Quantization layer \"{layer_name}\" has no output, skipping folding.")
            continue

        if len(output_layer_name) > 1:
            if log:
                print(f"Quantization layer \"{layer_name}\" has multiple outputs, skipping folding.")
            continue

        output_layer_name = output_layer_name[0]
        if io_dict[output_layer_name]["type"].lower() in ["conv", "pool", "produce"]:

            quant_layers_folding.append((layer_name, output_layer_name))
            if log:
                print(f"Quantization layer \"{layer_name}\" folded with \"{output_layer_name}\".")

    for layer_name, output_layer_name in quant_layers_folding:
        io_dict[output_layer_name]["input_quant"] = io_dict[layer_name]
        io_dict[output_layer_name]["input_quant"]["name"] = layer_name
        remove_and_bypass_layer(io_dict, io_connect, layer_name, False)
            
    return io_dict

def check_dangling_quant(io_dict):
    """ Check for dangling quantization layers. """
    
    # Retrieve all the quantization layers, excluding weights and biases
    quant_layers = [layer_name for layer_name, layer_info in io_dict.items() if "quant" in layer_info["type"]]

    for layer_name in quant_layers:
        print(f"Error in check_dangling_quant: quantization layer \"{layer_name}\" has not been folded.")

    return len(quant_layers) > 0

def check_dangling_relu(io_dict):
    """ Check for dangling relu layers. """

    # Retrieve all the relu layers
    relu_layers = [layer_name for layer_name, layer_info in io_dict.items() if "relu" in layer_info["type"]]

    for layer_name in relu_layers:
        print(f"Error in check_dangling_relu: relu layer \"{layer_name}\" has not been folded.")

    return len(relu_layers) > 0

def check_dangling_add(io_dict):
    """ Check for dangling add layers. """
    
    # Retrieve all the add layers
    add_layers = [layer_name for layer_name, layer_info in io_dict.items() if "add" in layer_info["type"]]

    for layer_name in add_layers:
        print(f"Error in check_dangling_add: add layer \"{layer_name}\" has not been folded.")

    return len(add_layers) > 0

def check_unquantized_layers(io_dict):
    """ Check for layers that have not been quantized. """
    
    for layer_name, layer_info in io_dict.items():

        # Checking layers with only output quantization
        if (layer_info["type"].lower() in ["produce"]):
            if layer_info["output_quant"] is None:
                print(f"Error in check_unquantized_layers: layer \"{layer_name}\" has not been quantized.")
                return True

        # Checking layers with input and output quantization
        if (layer_info["type"].lower() in ["conv", "pool", "add"]):
            if layer_info["input_quant"] is None or layer_info["output_quant"] is None:
                print(f"Error in check_unquantized_layers: layer \"{layer_name}\" has not been quantized.")
                return True
    
    return False

def dag_sorting(model, io_dict):
    """ Sort the layers of the model in a Directed Acyclic Graph (DAG). """

    io_connect = extract_connections(model, io_dict)

    start_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "produce"]

    if len(start_layers) > 1:
        print("Error in dag_sorting: multiple start layers.")
        exit(1)
    else:
        start_layer = start_layers[0]
    
    # Initializing layer index
    accepted_layers = ["conv", "pool", "produce", "consume", "add", "relu", "duplicate", "upsample", "concat"]
    start_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] in accepted_layers]
    for layer_name in start_layers:
        io_dict[layer_name]["layer_index"] = 0

    node_list = [(start_layer, 0)]
    while len(node_list) != 0:
        current_node, level = node_list.pop(0)
        output_nodes = next_layers(io_dict, io_connect, current_node)
        # print(f"Analizing {current_node} at level {level}")

        if output_nodes is not None:
            for node in output_nodes:
                if (level + 1 > io_dict[node]["layer_index"]):
                    io_dict[node]["layer_index"] = level + 1
                    node_list.append((node, level + 1))

    # Assigning to layers attached to produce nodes a boolean flag
    # Will be used in balance computation to recognize layers that 
    # have a produce node to consider.
    for layer_name, layer_info in io_dict.items():
        if "layer_index" in layer_info.keys():
            if layer_info['layer_index'] == 1:
                layer_info['start_comp_layer'] = True

    return io_dict

def opt_merge_pointwise(model, io_dict, log=False):
    """ Merge pointwise layers in skip connections with computation layer acting on the same tensor. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the pointwise layers 
    convadd_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "conv" and layer_info["add"]]
    merge_candidate_layers = [io_connect[io_dict[layer_name]["input"][1]][0][0] for layer_name in convadd_layers]
    pointwise_layers = [layer_name for layer_name in merge_candidate_layers if io_dict[layer_name]["fh"] == 1 and io_dict[layer_name]["fw"] == 1]

    for layer_name in pointwise_layers:

        # Retrieve the consumers of the input tensor
        input_tensor_net = io_dict[layer_name]["input"][0]
        tensor_consumers = io_connect[input_tensor_net][1]

        # Removing from the consumers the pointwise layer
        tensor_consumers.remove(layer_name)
        
        if len(tensor_consumers) == 0:
            continue
        else:
            merge_layer_name = tensor_consumers[0]

        # Check if the consumers have the same layer index
        if io_dict[merge_layer_name]["layer_index"] == io_dict[layer_name]["layer_index"]:
            if io_dict[merge_layer_name]["type"].lower() == "conv":

                io_dict[merge_layer_name]["merge_1x1"] = True
                io_dict[merge_layer_name]["merge_node"] = io_dict[layer_name]
                io_dict[merge_layer_name]["merge_node"]["name"] = layer_name

                print(f'Extending {io_dict[merge_layer_name]["output"]} with {io_dict[layer_name]["output"]}')
                io_dict[merge_layer_name]["output"].extend(io_dict[layer_name]["output"])
                
                del io_dict[layer_name]
                if log:
                    print(f"Pointwise layer \"{layer_name}\" merged with \"{merge_layer_name}\".")

    return io_dict

def opt_merge_skip(model, io_dict, log=False):
    """ Merge skip connections with computation layer acting on the same tensor. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all nets with multiple consumers
    skip_layers = [net_name for net_name, net_info in io_connect.items() if len(net_info[1]) > 1]
    
    for net_name in skip_layers:

        # Retrieve the consumers of the input tensor
        consumers = io_connect[net_name][1]
        
        # Check if the consumers are computation layers, and on different levels
        if len(consumers) == 2 :
            if all("conv" in io_dict[consumer]["type"] for consumer in consumers):
                if io_dict[consumers[0]]["layer_index"] != io_dict[consumers[1]]["layer_index"]:
                
                    merge_layer_name = consumers[0]
                    end_layer_name = consumers[1]

                    if io_dict[merge_layer_name]["layer_index"] > io_dict[end_layer_name]["layer_index"]:
                        merge_layer_name = consumers[1]
                        end_layer_name = consumers[0]
                    
                    io_dict[merge_layer_name]["has_forward"] = True
                    io_dict[end_layer_name]["input"].remove(net_name)

                    new_net_name = f"{net_name}_skip"

                    io_dict[merge_layer_name]["output"].append(new_net_name)
                    io_dict[end_layer_name]["input"].append(new_net_name)

                    if log:
                        print(f"Skip connection \"{net_name}\" merged with \"{merge_layer_name}\".")

    return io_dict

def propagate_quant(model, io_dict, log=False):
    """ Propagate output quantization to input quantization of the next layer, if not already present. """

    io_connect = extract_connections(model, io_dict)
    
    start_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "produce"]

    if len(start_layers) > 1:
        print("Error in propagate_quant: multiple start layers.")
        exit(1)
    else:
        start_layer = start_layers[0]
    
    accepted_layers = ["conv", "pool", "add","upsample", "concat"]
    node_list = [start_layer]
    mark_set = set()
    mark_set.add(start_layer)
    while len(node_list) != 0:
        current_node = node_list.pop()
        output_nodes = next_layers(io_dict, io_connect, current_node)
        print(f"Analizing {current_node}")

        if output_nodes is not None:
            for node in output_nodes:
                if io_dict[node]["type"] in accepted_layers:
                    if io_dict[current_node]["output_quant"] is None and io_dict[current_node]["input_quant"] is None:
                        print(f"Error in propagate_quant: no input/output quantization for layer \"{current_node}\".")
                        exit(1)
                    

                    if io_dict[current_node]["output_quant"] is None and io_dict[current_node]["input_quant"] is not None:
                        io_dict[current_node]["output_quant"] = io_dict[current_node]["input_quant"]
                        if log:
                            print(f"\tPropagating input quantization to output quantization for \"{current_node}\".")

                    if io_dict[node]["input_quant"] is None:
                        io_dict[node]["input_quant"] = io_dict[current_node]["output_quant"]
                        if log:
                            print(f"\tPropagating output quantization from \"{current_node}\" to input quantization for \"{node}\".")

                if node not in mark_set:
                    node_list.append(node)
                    mark_set.add(node)
            
    # Retrieve all the layers without input quantization
    # accepted_layers = ["conv", "pool", "add"]
    # no_input_quant_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] in accepted_layers and layer_info["input_quant"] is None]

    # for layer_name in no_input_quant_layers:
            
    #     # Retrieve the producers of the input tensor
    #     input_tensor_nets = io_dict[layer_name]["input"]
    #     for net in input_tensor_nets:
           
    #         produce_node = io_connect[net][0][0]
    #         if io_dict[produce_node]["output_quant"] is not None:
    #             io_dict[layer_name]["input_quant"] = io_dict[produce_node]["output_quant"]
    #             if log:
    #                 print(f"Propagating quantization from \"{produce_node}\" to \"{layer_name}\".")
    #             else:
    #                 print(f"Error in propagate_quant: no quantization propagated from \"{produce_node}\" to \"{layer_name}\".")
    
    return io_dict

def duplicate_tensor(model, io_dict, log=False):
    """ Duplicate a tensor in the model, in cases it is used by multiple computation layers. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the layers with multiple consumers
    multiple_consumer = [net_name for net_name, net_info in io_connect.items() if len(net_info[1]) > 1]
    print(f"Multiple: {multiple_consumer}")

    for net_name in multiple_consumer:

        # Retrieve the consumers of the tensor
        consumers = io_connect[net_name][1]

        # Retrieve the producer of the tensor
        producers = io_connect[net_name][0]

        # Check that the producer is a convolution layer and it is the only producer
        # Right now only convolution layers are supported.
        if not (len(producers) == 1 and io_dict[producers[0]]["type"].lower() in ["conv", "produce"]):
            continue

        # Create a new layer duplicating the tensor
        layer_name = f"dup_{net_name}"
        io_dict[layer_name] = {}
        io_dict[layer_name]["type"] = "duplicate"
        io_dict[layer_name]["name"] = layer_name
        io_dict[layer_name]["output"] = []
        io_dict[layer_name]["input"] = [net_name]
        io_dict[layer_name]["factor"] = len(consumers)
        if io_dict[producers[0]]["type"].lower() == "conv":
            io_dict[layer_name]["C"] = io_dict[producers[0]]["och"]
            io_dict[layer_name]["H"] = io_dict[producers[0]]["oh"]
            io_dict[layer_name]["W"] = io_dict[producers[0]]["ow"]
            io_dict[layer_name]["ops"] = io_dict[producers[0]]["ops_out"]
            io_dict[layer_name]["ow_ops"] = io_dict[producers[0]]["ow_ops_out"]
        else: 
            io_dict[layer_name]["C"] = io_dict[producers[0]]["ich"]
            io_dict[layer_name]["H"] = io_dict[producers[0]]["ih"]
            io_dict[layer_name]["W"] = io_dict[producers[0]]["iw"]
            io_dict[layer_name]["ops"] = io_dict[producers[0]]["ops"]
            io_dict[layer_name]["ow_ops"] = 1
        print(f"Creating duplicate layer {layer_name} with producer {producers[0]} and parallelism {io_dict[producers[0]]['ops_out']}, {io_dict[producers[0]]['ow_ops_out']}")
        
        #Divide the the nets to compute depth
        id = 0
        io_dict[producers[0]]["output"].remove(net_name)
        for consumer in consumers:
            new_net_name = f"{net_name}_dup_{id}"
            io_dict[consumer]["input"].remove(net_name)
            io_dict[consumer]["input"].append(new_net_name)
            io_dict[producers[0]]["output"].append(new_net_name)
            id += 1

            if log:
                print(f"Tensor \"{net_name}\" duplicated to \"{new_net_name}\".")
    
        # Not working for merge 1x1
        io_connect = extract_connections(model, io_dict)
        id = 0
        start_layer = io_dict[producers[0]]["layer_index"]
        for consumer in consumers:
            new_net_name = f"{net_name}_dup_{id}"
            end_layer = io_dict[consumer]["layer_index"]
            if (end_layer - start_layer) > 1:
                io_dict[layer_name][f"depth_{new_net_name}"] = compute_depth_stream(io_dict, io_connect, new_net_name, producers[0])
            id += 1
        
        id = 0
        io_dict[producers[0]]["output"].append(net_name)
        io_dict[layer_name]["input"] = [net_name]
        for consumer in consumers:
            new_net_name = f"{net_name}_dup_{id}"
            io_dict[producers[0]]["output"].remove(new_net_name)
            io_dict[layer_name]["output"].append(new_net_name)
            id += 1

    return io_dict

def opt_step(
    inferred_model,
    io_dict,
    init_info,
    log=False
):

    io_dict = opt_flatten(
        io_dict,
        inferred_model,
        log
    )

    io_dict = opt_pad(
        inferred_model,
        io_dict,
        log
    )

    # Folding quantization layers

    io_dict = fold_params_quant(
        inferred_model,
        io_dict,
        init_info,
        log
    )

    io_dict = fold_output_quant(
        inferred_model,
        io_dict,
        log
    )
    
    io_dict = fold_input_quant(
        inferred_model,
        io_dict,
        log
    )

    if (check_dangling_quant(io_dict)):
        exit(-1)

    io_dict = propagate_quant(
        inferred_model,
        io_dict,
        log
    )

    # Layer merging optimizations

    io_dict = dag_sorting(inferred_model, io_dict)
    
    io_dict = opt_add(
        inferred_model,
        io_dict,
        log
    )

    if (check_dangling_add(io_dict)):
        exit(-1)

    io_dict = opt_relu(
        inferred_model,
        io_dict,
        log
    )

    if (check_dangling_relu(io_dict)):
        exit(-1)

    io_dict = dag_sorting(inferred_model, io_dict)

    io_dict = opt_merge_pointwise(
        inferred_model,
        io_dict,
        log
    )

    io_dict = opt_merge_skip(
        inferred_model,
        io_dict,
        log
    )

    for layer_name, layer_info in io_dict.items():
        print(f"Layer: {layer_name} {next_layers(io_dict, extract_connections(inferred_model, io_dict), layer_name)}")
        if "input_quant" in layer_info.keys() and layer_info["input_quant"] is not None:
            quant_node = layer_info["input_quant"]
            quant_type = quant.get_quant_type(quant_node["signed"], quant_node["bits"], quant_node["scale_factor"])
            print(f"Input quant: {quant_type}")
        if "conv_output_quant" in layer_info.keys() and layer_info["conv_output_quant"] is not None:
            quant_node = layer_info["conv_output_quant"]
            quant_type = quant.get_quant_type(quant_node["signed"], quant_node["bits"], quant_node["scale_factor"])
            print(f"Conv output quant: {quant_type}")
        if "add" in layer_info.keys() and layer_info["add"] and layer_info["add_node"]["output_quant"] is not None:
            quant_node = layer_info["add_node"]["output_quant"]
            quant_type = quant.get_quant_type(quant_node["signed"], quant_node["bits"], quant_node["scale_factor"])
            print(f"Add quant: {quant_type}")
        if "output_quant" in layer_info.keys() and layer_info["output_quant"] is not None:
            quant_node = layer_info["output_quant"]
            quant_type = quant.get_quant_type(quant_node["signed"], quant_node["bits"], quant_node["scale_factor"])
            print(f"Output quant: {quant_type}")
        # if "weight_quant" in layer_info.keys() and layer_info["weight_quant"] is not None:
        #     quant_node = layer_info["weight_quant"]
        #     quant_type = quant.get_quant_type(quant_node["signed"], quant_node["bits"], quant_node["scale_factor"])
        #     print(f"Weight quant: {quant_type}")
        # if "bias_quant" in layer_info.keys() and layer_info["bias_quant"] is not None:
        #     quant_node = layer_info["bias_quant"]
        #     quant_type = quant.get_quant_type(quant_node["signed"], quant_node["bits"], quant_node["scale_factor"])
        #     print(f"Bias quant: {quant_type}")

    if (check_unquantized_layers(io_dict)):
        exit(-1)
    
    return io_dict
