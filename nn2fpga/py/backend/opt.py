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

            # The merge should be done with the layer on the longest chain of the branch
            layer_in_name = ""

            layer_out_name = layers[1][0]

            end_add = False
            if layer_out_name != "consume_stream":
                end_add = 'add' in io_dict[layer_out_name]["type"].lower()

            if (end_add):

                max_length = -1
                temp_start_conv = False
                for layer_name in layers[0]:
                    # look if there is at least a conv layer in the branching path
                    if ('conv' in io_dict[layer_name]["type"].lower()):
                        temp_start_conv = True 
                        length = compute_branch_length(io_dict, io_connect, layer_name)
                        if length > max_length:
                            max_length = length
                            max_index = layers[0].index(layer_name)

                if temp_start_conv and (max_length > 0):
                    layer_in_name = layers[0][max_index]

            if layer_in_name == "":
                continue

            start_conv = 'conv' in io_dict[layer_in_name]["type"].lower()

            # If true the add can be absorbed into convolution
            if start_conv and end_add:

                print("Merged add in:", layer_in_name)
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
        # Annotating produce layer to forward the quantization
        layer_in_names  = net_info[0]
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

                # Annotating quantization from previous layer
                in_layer = io_dict[layer_in_names[0]]
                in_index = in_layer["output"].index(net_name)
                in_scale = in_layer["scale_factor"][in_index]
                in_bits  = in_layer["bits"][in_index]
                in_clip  = in_layer["clip_factor"][in_index]
                if in_clip is list:
                    in_clip = in_clip[0]
                in_mask  = in_layer["mask_factor"][in_index]
                if in_mask is list:
                    in_mask = in_mask[0]
                in_clip_signed = in_layer["clip_signed"][in_index]
                if in_clip_signed is list:
                    in_clip_signed = in_clip_signed[0]
                in_mask_signed = in_layer["mask_signed"][in_index]
                if in_mask_signed is list:
                    in_mask_signed = in_mask_signed[0]

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
                        io_dict[layer_base_name]["scale_factor"].append(
                            in_scale
                        )
                        io_dict[layer_base_name]["bits"].append(
                            in_bits
                        )
                        io_dict[layer_base_name]["clip_factor"].append(
                            in_clip
                        )
                        io_dict[layer_base_name]["mask_factor"].append(
                            in_mask
                        )
                        io_dict[layer_base_name]["clip_signed"].append(
                            in_clip_signed
                        )
                        io_dict[layer_base_name]["mask_signed"].append(
                            in_mask_signed
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
            output_name = layer_info["output"][0]

            # Done to recognize vector connections
            output_name = output_name.split("[")[0]

            layer_out_len = len(io_connect[output_name][1])
            if layer_out_len > 1:

                # Checking that the net level is the same
                # output_names = layer_info["output"]
                output_names = []
                for output_layer in io_connect[output_name][1]:
                    output_names += io_dict[output_layer]["output"]

                is_same_level = all(
                    [
                        net_levels[output_names[0]][0] == net_levels[comp_name][0]
                        for comp_name in output_names
                    ]
                )

                layer_out_names = io_connect[output_name][1]

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
                    merging_length = [
                        compute_branch_length(io_dict, io_connect, layer_name, forward=True)
                        for layer_name in layer_out_names
                    ]
                    # exit()
                    max_length = max(merging_length)
                    max_index = merging_length.index(max_length)
                    layer_base_name = layer_out_names[max_index]

                    rem_layer = []
                    input_tensor = []
                    output_tensor = []
                    scale_factor = []
                    bits = []
                    clip_factor = []
                    mask_factor = []
                    clip_signed = []
                    mask_signed = []
                    in_scale_factor = []
                    in_bits = []
                    for layer_merge_name in layer_out_names:
                        if layer_base_name != layer_merge_name:
                            layer_merge = io_dict[layer_merge_name]
                            rem_layer.append(layer_merge_name)
                            for input in layer_merge["input"]:
                                if input not in output_names:
                                    input_tensor.append(input)
                            for i, output in enumerate(layer_merge["output"]):
                                output_tensor.append(output)
                                scale_value = layer_merge["scale_factor"][i]
                                scale_factor.append(scale_value)
                                bits_value = layer_merge["bits"][i]
                                bits.append(bits_value)
                                clip_value = layer_merge["clip_factor"][i]
                                if clip_value is list:
                                    clip_value = clip_value[0]
                                clip_factor.append(clip_value)
                                mask_value = layer_merge["mask_factor"][i]
                                if mask_value is list:
                                    mask_value = mask_value[0]
                                mask_factor.append(mask_value)
                                clip_signed_value = layer_merge["clip_signed"][i]
                                if clip_signed_value is list:
                                    clip_signed_value = clip_signed_value[0]
                                clip_signed.append(clip_signed_value)
                                mask_signed_value = layer_merge["mask_signed"][i]
                                if mask_signed_value is list:
                                    mask_signed_value = mask_signed_value[0]
                                mask_signed.append(mask_signed_value)
                                in_scale_value = layer_merge["in_scale_factor"][i]
                                in_scale_factor.append(in_scale_value)
                                in_bits_value = layer_merge["in_bits"][i]
                                in_bits.append(in_bits_value)
                            
                    io_dict[layer_base_name]["merge_1x1"] = True

                    for i, input in enumerate(input_tensor):
                        if (i > 0):
                          io_dict[layer_base_name]["input"].append(input)

                    for output in output_tensor:
                        io_dict[layer_base_name]["output"].append(output)

                    io_dict[layer_base_name]["scale_factor"] += scale_factor
                    io_dict[layer_base_name]["bits"] += bits
                    io_dict[layer_base_name]["clip_factor"] += clip_factor
                    io_dict[layer_base_name]["mask_factor"] += mask_factor
                    io_dict[layer_base_name]["clip_signed"] += clip_signed
                    io_dict[layer_base_name]["mask_signed"] += mask_signed
                    io_dict[layer_base_name]["in_scale_factor"] += in_scale_factor
                    io_dict[layer_base_name]["in_bits"] += in_bits

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

# Creating quantization layers if add/conv layers are without it
def assign_quant(model, io_dict):

    io_connect = extract_connections(model, io_dict)

    removed_layers = []

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        search_layers = [
            'conv',
            'pool',
            'produce',
            'add'
        ]

        if layer_in_name in removed_layers:
            continue

        start_merge = any(
            search_name in io_dict[layer_in_name]["type"]
            for search_name in search_layers
        )

        if (layer_out_name == "consume_stream"):
            continue

        end_quant = 'quant' == io_dict[layer_out_name]["type"]
        multiple_quant = len(layers[1]) > 1

        if start_merge and end_quant and multiple_quant:
            # TODO: optimize it
            io_dict[layer_in_name]["scale_factor"] = [0]
            io_dict[layer_in_name]["quant"] = True
            io_dict[layer_in_name]["signed"] = [1]
            io_dict[layer_in_name]["narrow"] = [0]
            io_dict[layer_in_name]["bits"] = [32]
            io_dict[layer_in_name]["actscale"] = [0]
            io_dict[layer_in_name]["clip"] = [0]
            io_dict[layer_in_name]["mask"] = [0]
            io_dict[layer_in_name]["clip_factor"] = [0]
            io_dict[layer_in_name]["mask_factor"] = [0]
            io_dict[layer_in_name]["clip_signed"] = [0]
            io_dict[layer_in_name]["mask_signed"] = [0]

    return io_dict

def opt_quant(model, io_dict, quant_info):
    
    change = True
    while change:
        change = False
        io_connect = extract_connections(model, io_dict)

        removed_layers = []

        for net_name, layers in io_connect.items():
            layer_in_name = layers[0][0]
            layer_out_name = layers[1][0]

            search_layers = [
                'conv',
                'pool',
                'produce',
                'add',
            ]

            if layer_in_name in removed_layers:
                continue

            start_merge = any(
                search_name in io_dict[layer_in_name]["type"]
                for search_name in search_layers
            )

            if (layer_out_name == "consume_stream"):
                continue

            end_quant = 'quant' == io_dict[layer_out_name]["type"]
            single_quant = len(layers[1]) < 2

            # If true the relu can be absorbed into convolution
            if start_merge and end_quant and single_quant:

                others_quant = len(quant_info[net_name]["others"]) > 0

                if not others_quant:
                    out_names = io_dict[layer_out_name]["output"]

                    # Scale factor is equal to the one of the quantization in 
                    # output 

                    seq_scale = io_dict[layer_out_name]["scale_factor"]
                    seq_bits = io_dict[layer_out_name]["bits"]
                    # seq_scale = quant_info[net_name]["seq_scale"]
                    if isinstance(seq_scale, list):
                        scale_factor = seq_scale[0]
                    else:
                        scale_factor = seq_scale

                    if isinstance(seq_bits, list):
                        bits = seq_bits[0]
                    else:
                        bits = seq_bits

                    clip_factor = quant_info[net_name]["seq_clip"][0]
                    if isinstance(clip_factor, list):
                        clip_factor = clip_factor[0]
                    mask_factor = quant_info[net_name]["seq_mask"][0]
                    if isinstance(mask_factor, list):
                        mask_factor = mask_factor[0]
                    clip_signed = quant_info[net_name]["seq_clip_signed"][0]
                    if isinstance(clip_signed, list):
                        clip_signed = clip_signed[0]
                    mask_signed = quant_info[net_name]["seq_mask_signed"][0]
                    if isinstance(mask_signed, list):
                        mask_signed = mask_signed[0]
                    signed = quant_info[net_name]["signed"]

                    in_index = io_dict[layer_in_name]["output"].index(net_name)

                    if "quant" in io_dict[layer_in_name].keys():
                        # The old clip must be saved to have coherent behavior
                        # If a merged quantization has lower scaling factor then
                        # quantization is clipping to a lower max value
                        old_clip = io_dict[layer_in_name]["clip_factor"][in_index]
                        old_clip_signed = io_dict[layer_in_name]["clip_signed"][in_index]
                        if (old_clip < clip_factor):
                            io_dict[layer_in_name]["clip_factor"][in_index] = old_clip
                            io_dict[layer_in_name]["clip_signed"][in_index] = old_clip_signed
                        else:
                            io_dict[layer_in_name]["clip_factor"][in_index] = clip_factor
                            io_dict[layer_in_name]["clip_signed"][in_index] = clip_signed
                    else:
                        io_dict[layer_in_name]["clip_factor"] = [clip_factor]
                        io_dict[layer_in_name]["clip_signed"] = [clip_signed]

                    if "quant" in io_dict[layer_in_name].keys():
                        # The old mask must be saved to have coherent behavior
                        # If a merged quantization has higher scaling factor then
                        # quantization is masking the LSBs
                        old_mask = io_dict[layer_in_name]["mask_factor"][in_index]
                        old_mask_signed = io_dict[layer_in_name]["mask_signed"][in_index]
                        if (old_mask > mask_factor):
                            io_dict[layer_in_name]["mask_factor"][in_index] = old_mask
                            io_dict[layer_in_name]["mask_signed"][in_index] = old_mask_signed
                        else:
                            io_dict[layer_in_name]["mask_factor"][in_index] = mask_factor
                            io_dict[layer_in_name]["mask_signed"][in_index] = mask_signed
                    else:
                        io_dict[layer_in_name]["mask_factor"] = [mask_factor]
                        io_dict[layer_in_name]["mask_signed"] = [mask_signed]

                    io_dict[layer_in_name]["quant"] = True
                    io_dict[layer_in_name]["scale_factor"] = [scale_factor]
                    io_dict[layer_in_name]["bits"] = [bits]
                    io_dict[layer_in_name]["signed"] = signed
                    io_dict[layer_in_name]["output"] = out_names

                    removed_layers.append(layer_out_name)

                    del io_dict[layer_out_name]
                    change = True

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
                    if not "quant" in io_dict[layer_out_name].keys():
                        continue
                    if not io_dict[layer_out_name]["quant"]:
                        continue

                    # This opt is only possible when the input tensor has already
                    # a quantization
                    pre_quant_layer = io_connect[quant_input][0][0]
                    if not "quant" in io_dict[pre_quant_layer].keys():
                        continue
                    if not io_dict[pre_quant_layer]["quant"]:
                        continue
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

                    # no_quant_dict = False
                    # for quant_layer_name in parallel_quant_layers:
                    #     if quant_layer_name not in io_dict.keys():
                    #         no_quant_dict = True

                    # if no_quant_dict:
                    #     break

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
                    bits = quant_dict["seq_bits"][scale_index]

                    # The scale factor is stored and the activation is 
                    # requantized before performing convolution
                    io_dict[layer_out_name]["in_scale_factor"] = [scale_factor]
                    io_dict[layer_out_name]["in_bits"] = [bits]
                    io_dict[layer_out_name]["input"][0] = quant_input
                    change = True
                    del io_dict[layer_in_name]
                    break

                if is_split and not_constant and is_regular:

                    # If the scale factors of the quantization are regular, the
                    # branch can be moved after the quantization allowing a better
                    # optimization merging the pre-conv layer with the merged quant
                    
                    merging_layer = parallel_quant_layers[0]
                    for i, rem_name in enumerate(parallel_quant_layers):
                        if i > 0:
                            # Keeping just one tensor
                            rem_output = io_dict[rem_name]["output"][0]
                            layer_out = io_connect[rem_output][1][0]
                            io_dict[layer_out]["input"][0] = new_output
                        else:
                            new_output = io_dict[rem_name]["output"][0]
                            io_dict[merging_layer]["output"].append(new_output)

                    for i, rem_name in enumerate(parallel_quant_layers):
                        if i > 0:
                            del io_dict[rem_name]
                    change = True
                    break

    return io_dict

def share_reuse(model, io_dict):
    
    io_connect = extract_connections(model, io_dict)

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        start_const = 'const' == io_dict[layer_in_name]["type"]

        # If true the relu can be absorbed into convolution
        if start_const:
            reuse0 = io_dict[layer_in_name]["reuse"]
            reuse1 = io_dict[layer_out_name]["reuse"]
            reuse = max([reuse0, reuse1])
            io_dict[layer_in_name]["reuse"] = reuse
            io_dict[layer_out_name]["reuse"] = reuse

    return io_dict

def opt_flatten(io_dict, model):

    io_connect = extract_connections(model, io_dict)

    # cycling on io_dict to find flatten layers
    for layer_name, layer_info in io_dict.items():
            
        if layer_info["type"] == "flatten":

            # getting input and output of flatten layer
            input = layer_info["input"][0]
            output = layer_info["output"][0]

            # getting input and output layer before flatten
            input_layer = io_connect[input][0][0]

            # getting input and output of the layer after flatten
            io_dict[input_layer]["output"][0] = output

            # removing flatten layer
            del io_dict[layer_name]
            break

    return io_dict

def opt_steps(
    inferred_model,
    io_dict,
    init_info
):
    for i in range(3):

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

                # io_dict = assign_quant(
                #     inferred_model,
                #     io_dict
                # )

                io_dict = opt_flatten(
                    io_dict,
                    inferred_model
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

            io_dict = opt_relu(
                inferred_model,
                io_dict,
            )

            quant_info = extract_quant_info(
                inferred_model,
                io_dict,
                init_info
            )

            io_dict = opt_skip_quant(
                inferred_model,
                io_dict,
                quant_info,
                init_info
            )

            io_dict = opt_add(
                inferred_model,
                io_dict,
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

