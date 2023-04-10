import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.quant import *
from backend.graph import *
from backend.opt import *

def opt_relu(model, io_dict):
    
    io_connect = extract_connections(model, io_dict)

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        start_conv = 'conv' in layer_in_name.lower()
        end_relu = 'relu' in layer_out_name.lower()

        # If true the relu can be absorbed into convolution
        if start_conv and end_relu:
            out_name = io_dict[layer_out_name]["output"][0]
            io_dict[layer_in_name]["relu"] = True
            io_dict[layer_in_name]["output"][0] = out_name
            del io_dict[layer_out_name]

    return io_dict

def opt_add(model, io_dict):
    
    no_break = False
    # Compute for each tensor the distance between producer and consumer
    while not no_break:

        io_connect = extract_connections(model, io_dict)

        no_break = True

        for net_name, layers in io_connect.items():

            # By construction if the add layer is in the first position of the 
            # output layers of a net it means it can be merged
            layer_in_name = layers[0][0]
            layer_out_name = layers[1][0]

            start_conv = 'conv' in layer_in_name.lower()
            end_add = 'add' in layer_out_name.lower()

            # If true the add can be absorbed into convolution
            if start_conv and end_add:

                # The add operation should be not merged with a pointwise layer
                fh = io_dict[layer_in_name]["fh"] 
                fw = io_dict[layer_in_name]["fw"] 
                is_not_pointwise =  (fh != 1) or (fw != 1)

                if is_not_pointwise:

                    io_dict[layer_in_name]["add"] = True

                    # Bypassing add layer changing conv output
                    out_name = io_dict[layer_out_name]["output"][0]
                    io_dict[layer_in_name]["output"][0] = out_name

                    # Adding other outputs in input to the conv layer
                    for input in io_dict[layer_out_name]["input"]:
                        if input != net_name:
                            io_dict[layer_in_name]["input"].append(input)

                    # Removing layer
                    del io_dict[layer_out_name]
                    no_break = False
                    break

    return io_dict

def opt_skip(model, io_dict):

    # If the hop for the skip connection is greater than 1 then it means that
    # the skip connection buffering size can be optimized reusing the buffering
    # already provided by the first consumer

    io_connect = extract_connections(model, io_dict)

    net_levels = net_distance(io_dict, io_connect)

    for net_name, net_info in io_connect.items():
        layer_out_names = net_info[1]

        if len(layer_out_names) > 1:
            levels = net_levels[net_name]
            not_same_level = not(all(
                [
                    levels[0] == comp
                    for node_name, comp in levels[1]
                ]
            ))

            all_conv = all(
                [
                    'conv' in layer_name.lower()
                    for layer_name in layer_out_names
                ]
            )

            if not_same_level and all_conv:
                # Reordering the layers to have correct actiations passthrough
                ordered_layers = list(
                    sorted(
                        levels[1], key=lambda i: i[1], reverse=False
                    )
                )

                layer_base_name = ordered_layers[0][0]
                io_dict[layer_base_name]["has_forward"] = True
                for i, layer_info in enumerate(ordered_layers):
                    if i != 0:
                        layer_name = layer_info[0]
                        skip_name = net_name + "_skip"
                        io_dict[layer_base_name]["output"].append(
                            skip_name
                        )
                        skip_index = io_dict[layer_name]["input"].index(
                            net_name
                        )
                        io_dict[layer_name]["input"][skip_index] = skip_name 

    return io_dict

def opt_merge_conv(model, io_dict):

    # Merging 3x3 and pointwise convolutions acting on the same data
    
    no_break = False

    while not no_break:

        io_connect = extract_connections(model, io_dict)

        net_levels = net_distance(io_dict, io_connect)

        no_break = True

        for layer_name, layer_info in io_dict.items():

            # Check if there are multiple layer connected to the net
            layer_out_len = len(layer_info["output"])
            if layer_out_len > 1:

                # Checking that the net level is the same
                output_names = layer_info["output"]

                is_same_level = all(
                    [
                        net_levels[output_names[0]][0] == net_levels[comp_name][0]
                        for comp_name in output_names
                    ]
                )

                layer_out_names = [
                    io_connect[output][1][0] for output in output_names
                ]

                all_conv = all(
                    [
                        'conv' in layer_name.lower()
                        for layer_name in layer_out_names
                    ]
                )

                not_same_conv = all(
                    [
                        layer_out_names[0] != layer_name
                        for i, layer_name in enumerate(layer_out_names)
                        if i != 0
                    ]
                )

                # If they are all convolutions and the tensors are at the same
                # level, the operations can be merged in one layer
                if all_conv and is_same_level and not_same_conv:
                    layer_base_name = layer_out_names[0]

                    rem_layer = []
                    input_tensor = []
                    output_tensor = []
                    for layer_merge_name in layer_out_names:
                        if layer_base_name != layer_merge_name:
                            rem_layer.append(layer_merge_name)
                            for input in io_dict[layer_merge_name]["input"]:
                                if input not in output_names:
                                    input_tensor.append(input)
                            for output in io_dict[layer_merge_name]["output"]:
                                output_tensor.append(output)
                            
                    io_dict[layer_base_name]["merge_1x1"] = True

                    for input in input_tensor:
                        io_dict[layer_base_name]["input"].append(input)

                    for output in output_tensor:
                        io_dict[layer_base_name]["output"].append(output)

                    # Removing merged layer
                    for rem_name in rem_layer:
                        del io_dict[rem_name]

                    # Removing output stream going to merged layer
                    io_dict[layer_name]["output"] = [
                        io_dict[layer_name]["output"][0]
                    ]
                    no_break = False

                    break

    return io_dict

def opt_quant(model, io_dict, quant_info):
    
    io_connect = extract_connections(model, io_dict)

    removed_layers = []

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        search_layers = [
            'conv',
            'pool',
            'produce',
        ]

        if layer_in_name in removed_layers:
            continue

        start_merge = any(
            search_name in io_dict[layer_in_name]["type"]
            for search_name in search_layers
        )

        if (layer_out_name == "ConsumeStream"):
            continue

        end_quant = 'quant' == io_dict[layer_out_name]["type"]
        single_quant = len(layers[1]) < 2

        # If true the relu can be absorbed into convolution
        if start_merge and end_quant and single_quant:

            others_quant = len(quant_info[net_name]["others"]) > 0

            if not others_quant:
                out_names = io_dict[layer_out_name]["output"]

                # Scale factor is equal to the one of the quantization in 
                # output In case there are multiple output nodes, all of the 
                # must be saved

                scale_factor = quant_info[net_name]["seq_scale"]
                clip_factor = quant_info[net_name]["seq_clip"]
                signed = quant_info[net_name]["signed"]

                in_index = io_dict[layer_in_name]["output"].index(net_name)

                if "quant" in io_dict[layer_in_name].keys():
                    old_clip = io_dict[layer_in_name]["clip_factor"][in_index]
                    if (old_clip < clip_factor):
                        io_dict[layer_in_name]["clip_factor"][in_index] = old_clip
                else:
                    io_dict[layer_in_name]["clip_factor"] = clip_factor

                io_dict[layer_in_name]["quant"] = True
                io_dict[layer_in_name]["scale_factor"] = scale_factor
                io_dict[layer_in_name]["signed"] = signed
                io_dict[layer_in_name]["output"] = out_names

                removed_layers.append(layer_out_name)

                del io_dict[layer_out_name]

    return io_dict

def opt_skip_quant(model, io_dict, quant_info, init_info):
    
    # If there is a quant node that is blocking a skip connection opitmization
    # It must be absorbed by the convolution layer that is consuming its
    # output instead of merging with the one producing its input

    change = True

    while change:
        io_connect = extract_connections(model, io_dict)
        change = False
        for net_name, layers in io_connect.items():
            layer_in_name = layers[0][0]
            layer_out_name = layers[1][0]

            start_quant = 'quant' in layer_in_name.lower()
            end_conv = 'conv' in layer_out_name.lower()

            quant_input = io_dict[layer_in_name]["input"][0]
            not_constant = quant_input not in init_info.keys()

            # If true the relu can be absorbed into convolution
            if end_conv and start_quant and not_constant:

                is_split = len(io_connect[quant_input][1]) > 1

                if is_split and not_constant:
                    # This opt is only possible when the output conv has already
                    # a quantization
                    assert "quant" in io_dict[layer_out_name].keys()
                    assert io_dict[layer_out_name]["quant"]

                    # This opt is only possible when the input tensor has already
                    # a quantization
                    pre_quant_layer = io_connect[quant_input][0][0]
                    assert "quant" in io_dict[pre_quant_layer].keys()
                    assert io_dict[pre_quant_layer]["quant"]
                else:
                    continue

                # A regular split has only quant layers as consumers of the net
                # and the same quantization
                all_quant = all(
                    [
                        'quant' in name.lower() 
                        for name in io_connect[quant_input][1]
                    ]
                )

                is_regular = all_quant

                if all_quant:
                    parallel_quant_layers = io_connect[quant_input][1]
                    pre_scale_factor = [
                        io_dict[quant_layer_name]["scale_factor"] 
                        for quant_layer_name in parallel_quant_layers
                    ]
                    first_scale_factor = pre_scale_factor[0]
                    count_first_scale_factor = pre_scale_factor.count(
                        first_scale_factor
                    )

                    len_scale_factor = len(pre_scale_factor)

                    is_regular &= count_first_scale_factor == len_scale_factor

                if is_split and not_constant and (not is_regular):

                    # To do a forward merge of the quantization the scale factor
                    # change of the input must be tracked to use it when quantizing
                    # the output and when taking into account the bias
                    pre_scale_factor = io_dict[pre_quant_layer]["quant"]
                    quant_dict = quant_info[quant_input]
                    scale_index = quant_dict["seq_out"].index(net_name)
                    scale_factor = quant_dict["seq_scale"][scale_index]
                    diff_scale_factor = scale_factor - pre_scale_factor

                    io_dict[layer_out_name]["in_scale_factor"] = diff_scale_factor
                    io_dict[layer_out_name]["input"][0] = quant_input
                    del io_dict[layer_in_name]

                if is_split and not_constant and is_regular:

                    # If the scale factors of the quantization are regular, the
                    # branch can be moved after the quantization allowing a better
                    # optimization merging the pre-conv layer with the merged quant
                    
                    merging_layer = parallel_quant_layers[0]
                    for i, rem_name in enumerate(parallel_quant_layers):
                        if i > 0:
                            new_output = io_dict[rem_name]["output"][0]
                            io_dict[merging_layer]["output"].append(new_output)

                    for i, rem_name in enumerate(parallel_quant_layers):
                        if i > 0:
                            del io_dict[rem_name]
                    change = True
                    break

    return io_dict

def opt_steps(
    inferred_model,
    io_dict,
    init_info
):
    for i in range(2):

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
                    inferred_model,
                    io_dict,
                    quant_info
                )

                io_dict = opt_relu(
                    inferred_model,
                    io_dict,
                )

            io_dict = opt_add(
                inferred_model,
                io_dict,
            )

            io_dict = opt_relu(
                inferred_model,
                io_dict,
            )

        quant_info = extract_quant_info(
            inferred_model,
            io_dict,
            init_info
        )

        for name, info in io_dict.items():
            print(name, info)

        io_dict = opt_skip_quant(
            inferred_model,
            io_dict,
            quant_info,
            init_info
        )

    io_dict = opt_merge_conv(
        inferred_model,
        io_dict,
    )

    io_dict = opt_skip(
        inferred_model,
        io_dict,
    )

    return io_dict

