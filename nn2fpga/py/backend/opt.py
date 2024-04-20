import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.quant import *
from backend.graph import *
from backend.opt import *

def sanity_check(model, io_dict, log="", debug=False):
    # Check that all the signed are coherent
    if debug:
        print("########## Sanity check", log)
    for name, layer in io_dict.items():
        check_quant = False
        if "quant" in layer.keys():
            if layer["quant"]:
                check_quant = True
        if "quant" == layer["type"]:
            check_quant = True
        if check_quant:
            if debug:
                print("########", name, layer["signed"])
            if "signed" in layer.keys():
                if isinstance(layer["signed"], list):
                    for elem in layer["signed"]:
                        if isinstance(elem, list):
                            print("Error in", name, "signed is a list", log)
                            raise Exception

def opt_pad(model, io_dict, flag_modified=False):
    
    local_flag_modified = False
    io_connect = extract_connections(model, io_dict)

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        comp_layers = ["conv", "pool"]
        start_pad = 'pad' in io_dict[layer_in_name]["type"]
        if start_pad:
            end_comp = io_dict[layer_out_name]["type"].lower() in comp_layers

            # If true the relu can be absorbed into convolution
            if start_pad and end_comp:
                in_name = io_dict[layer_in_name]["input"][0]
                io_dict[layer_out_name]["pad"] = io_dict[layer_in_name]["pad"]
                io_dict[layer_out_name]["input"][0] = in_name
                local_flag_modified = True
                del io_dict[layer_in_name]

    return io_dict, (flag_modified or local_flag_modified)

def opt_relu(model, io_dict, flag_modified=False):
    
    local_flag_modified = False
    io_connect = extract_connections(model, io_dict)

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        start_conv = 'conv' in layer_in_name.lower()
        end_relu = 'relu' in layer_out_name.lower()

        # If true the relu can be absorbed into convolution
        if start_conv and end_relu:
            print(f"Merged relu {layer_out_name} in: {layer_in_name}")
            out_name = io_dict[layer_out_name]["output"][0]
            io_dict[layer_in_name]["relu"] = True
            io_dict[layer_in_name]["output"][0] = out_name
            local_flag_modified = True
            del io_dict[layer_out_name]

    return io_dict, (flag_modified or local_flag_modified)

def opt_add(model, io_dict, flag_modified=False):
    
    no_break = False
    local_flag_modified = False
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
                local_flag_modified = True
                no_break = False
                break

    return io_dict, (flag_modified or local_flag_modified)

def opt_skip(model, io_dict, flag_modified=False):

    # If the hop for the skip connection is greater than 1 then it means that
    # the skip connection buffering size can be optimized reusing the buffering
    # already provided by the first consumer
    local_flag_modified = False
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
                in_signed  = in_layer["signed"][in_index]
                print("##################### opt_skip in_signed ", in_signed)
                if in_signed is list:
                    in_signed = in_signed[0]
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
                        local_flag_modified = True
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
                        io_dict[layer_base_name]["signed"].append(
                            in_signed
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
                print("##################### in_signed ", io_dict[layer_base_name]["signed"])

    return io_dict, (flag_modified or local_flag_modified)

def opt_merge_conv(model, io_dict, flag_modified=False):
    """Merge convolutions that are acting on the same data"""
    
    # Merging 3x3 and pointwise convolutions acting on the same data
    local_flag_modified = False    
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
                    signed = []
                    clip_factor = []
                    mask_factor = []
                    clip_signed = []
                    mask_signed = []
                    in_scale_factor = []
                    in_bits = []
                    in_signed = []
                    weights_name = []
                    print("##################### opt_merge_conv signed ", io_dict[layer_base_name]["signed"])
                    for layer_merge_name in layer_out_names:
                        if layer_base_name != layer_merge_name:
                            layer_merge = io_dict[layer_merge_name]
                            print(layer_merge_name, "merged in", layer_base_name)
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
                                signed_value = layer_merge["signed"][i]
                                if signed_value is list:
                                    signed_value = signed_value[0]
                                signed.append(signed_value)
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
                                in_signed_value = layer_merge["in_signed"][i]
                                in_signed.append(in_signed_value)
                                weights_name.append(layer_merge["weights_name"][i])
                            
                    io_dict[layer_base_name]["merge_1x1"] = True
                    io_dict[layer_base_name]["och_1x1"] = layer_merge["och"]

                    for i, input in enumerate(input_tensor):
                        if (i > 0):
                          io_dict[layer_base_name]["input"].append(input)

                    for output in output_tensor:
                        io_dict[layer_base_name]["output"].append(output)

                    io_dict[layer_base_name]["scale_factor"] += scale_factor
                    io_dict[layer_base_name]["bits"] += bits
                    io_dict[layer_base_name]["signed"] += signed
                    io_dict[layer_base_name]["clip_factor"] += clip_factor
                    io_dict[layer_base_name]["mask_factor"] += mask_factor
                    io_dict[layer_base_name]["clip_signed"] += clip_signed
                    io_dict[layer_base_name]["mask_signed"] += mask_signed
                    io_dict[layer_base_name]["in_scale_factor"] += in_scale_factor
                    io_dict[layer_base_name]["in_bits"] += in_bits
                    io_dict[layer_base_name]["in_signed"] += in_signed
                    io_dict[layer_base_name]["weights_name"] += weights_name
                    print("##################### in_signed ", io_dict[layer_base_name]["signed"])

                    # Removing merged layer
                    for rem_name in rem_layer:
                        local_flag_modified = True
                        del io_dict[rem_name]

                    # Removing output stream going to merged layer
                    io_dict[layer_name]["output"] = [
                        io_dict[layer_name]["output"][0]
                    ]
                    no_break = False

                    break

    return io_dict, (flag_modified or local_flag_modified)

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
            io_dict[layer_in_name]["actsigned"] = [0]
            io_dict[layer_in_name]["clip"] = [0]
            io_dict[layer_in_name]["mask"] = [0]
            io_dict[layer_in_name]["clip_factor"] = [0]
            io_dict[layer_in_name]["mask_factor"] = [0]
            io_dict[layer_in_name]["clip_signed"] = [0]
            io_dict[layer_in_name]["mask_signed"] = [0]
            io_dict[layer_in_name]["clip_bits"] = [0]
            io_dict[layer_in_name]["mask_bits"] = [0]

    return io_dict

def opt_quant(model, io_dict, init_info, flag_modified, debug=False):

    io_connect = extract_connections(model, io_dict)

    removed_layers = []

    change = False
    sanity_check(model, io_dict, "during opt_quant")

    print("Another round of opt_quant!!!!!!!!!!!!!!!!")
    for net_name, layers in io_connect.items():

        quant_info = extract_quant_info(
            model,
            io_dict,
            init_info
        )

        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        search_layers = [
            'conv',
            'pool',
            'produce',
            # 'add',
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
        # print(f"Considering {layer_in_name} -> {layer_out_name} for opt_quant, start_merge: {start_merge}, end_quant: {end_quant}, single_quant: {single_quant}")

        # If true the relu can be absorbed into convolution
        if start_merge and end_quant and single_quant:

            others_quant = len(quant_info[net_name]["others"]) > 0
            # print(f"others_quant: {others_quant}")

            if not others_quant:
                print(
                    f"Merging {layer_in_name} [{io_dict[layer_in_name]['type']}] -> {layer_out_name} [{io_dict[layer_out_name]['type']}] for opt_quant")
                out_names = io_dict[layer_out_name]["output"]

                # Scale factor is equal to the one of the quantization in 
                # output 
                if "signed" in io_dict[layer_in_name].keys():
                    signed = io_dict[layer_in_name]["signed"]
                else:
                    signed = None
                    
                seq_scale = io_dict[layer_out_name]["scale_factor"]
                if isinstance(seq_scale, list):
                    scale_factor = seq_scale[0]
                else:
                    scale_factor = seq_scale

                seq_bits = io_dict[layer_out_name]["bits"]
                if isinstance(seq_bits, list):
                    bits = seq_bits[0]
                else:
                    bits = seq_bits

                signed = quant_info[net_name]["seq_signed"]
                if isinstance(signed, list):
                    signed = signed[0]
                else:
                    signed = signed

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

                clip_bits = quant_info[net_name]["seq_clip_bits"][0]
                if isinstance(clip_bits, list):
                    clip_bits = clip_bits[0]

                mask_bits = quant_info[net_name]["seq_mask_bits"][0]
                if isinstance(mask_bits, list):
                    mask_bits = mask_bits[0]

                in_index = io_dict[layer_in_name]["output"].index(net_name)

                if "quant" in io_dict[layer_in_name].keys():

                    # The old clip must be saved to have coherent behavior
                    # If a merged quantization has lower scaling factor then
                    # quantization is clipping to a lower max value
                    old_clip = io_dict[layer_in_name]["clip_factor"][in_index]
                    old_clip_signed = io_dict[layer_in_name]["clip_signed"][in_index]
                    old_clip_bits = io_dict[layer_in_name]["clip_bits"][in_index]
                    
                    if (old_clip < clip_factor):
                        io_dict[layer_in_name]["clip_factor"][in_index] = old_clip
                        io_dict[layer_in_name]["clip_signed"][in_index] = old_clip_signed
                        io_dict[layer_in_name]["clip_bits"][in_index] = old_clip_bits
                    else:
                        io_dict[layer_in_name]["clip_factor"][in_index] = clip_factor
                        io_dict[layer_in_name]["clip_signed"][in_index] = clip_signed
                        io_dict[layer_in_name]["clip_bits"][in_index] = clip_bits
                else:
                    io_dict[layer_in_name]["clip_factor"] = [clip_factor]
                    io_dict[layer_in_name]["clip_signed"] = [clip_signed]
                    io_dict[layer_in_name]["clip_bits"] = [clip_bits]

                if "quant" in io_dict[layer_in_name].keys():
                    print("Already have quant in {layer_name}")
                    # The old mask must be saved to have coherent behavior
                    # If a merged quantization has higher scaling factor then
                    # quantization is masking the LSBs
                    old_mask = io_dict[layer_in_name]["mask_factor"][in_index]
                    old_mask_signed = io_dict[layer_in_name]["mask_signed"][in_index]
                    old_mask_bits = io_dict[layer_in_name]["mask_bits"][in_index]
                    if (old_mask > mask_factor):
                        io_dict[layer_in_name]["mask_factor"][in_index] = old_mask
                        io_dict[layer_in_name]["mask_signed"][in_index] = old_mask_signed
                        io_dict[layer_in_name]["mask_bits"][in_index] = old_mask_bits
                    else:
                        io_dict[layer_in_name]["mask_factor"][in_index] = mask_factor
                        io_dict[layer_in_name]["mask_signed"][in_index] = mask_signed
                        io_dict[layer_in_name]["mask_bits"][in_index] = mask_bits
                else:
                    io_dict[layer_in_name]["mask_factor"] = [mask_factor]
                    io_dict[layer_in_name]["mask_signed"] = [mask_signed]
                    io_dict[layer_in_name]["mask_bits"] = [mask_bits]

                io_dict[layer_in_name]["quant"] = True
                io_dict[layer_in_name]["scale_factor"] = [scale_factor]
                io_dict[layer_in_name]["bits"] = [bits]
                if isinstance(signed, list):
                    io_dict[layer_in_name]["signed"] = signed
                else:
                    io_dict[layer_in_name]["signed"] = [signed]
                if debug:
                    print("##################### signed ", signed)
                io_dict[layer_in_name]["output"] = out_names

                removed_layers.append(layer_out_name)

                del io_dict[layer_out_name]
                change = True
                io_dict, change = opt_quant(model, io_dict, init_info, change, True)
                return io_dict, True

    return io_dict, (flag_modified or change)

def opt_skip_quant(model, io_dict, init_info, flag_modified=False):

    
    # If there is a quant node that is blocking a skip connection opitmization
    # It must be absorbed by the convolution layer that is consuming its
    # output instead of merging with the one producing its input
    local_flag_modified = False
    change = True

    while change:
        io_connect = extract_connections(model, io_dict)
        change = False
        quant_info = extract_quant_info(
            model,
            io_dict,
            init_info
        )
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
                    signed = quant_dict["seq_signed"][scale_index]

                    # The scale factor is stored and the activation is 
                    # requantized before performing convolution
                    io_dict[layer_out_name]["in_scale_factor"] = [scale_factor]
                    io_dict[layer_out_name]["in_bits"] = [bits]
                    if isinstance(signed, list):
                        io_dict[layer_out_name]["in_signed"] = signed
                    else:
                        io_dict[layer_out_name]["in_signed"] = [signed]
                    io_dict[layer_out_name]["input"][0] = quant_input
                    change = True
                    local_flag_modified = True
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
                            local_flag_modified = True
                            del io_dict[rem_name]
                    change = True
                    break

    return io_dict, (flag_modified or local_flag_modified)

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

def opt_flatten(io_dict, model, flag_modified):
    """Optimize flatten layers by removing them and connecting the input and output layers"""

    local_flag_modified = False
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
            local_flag_modified = True
            del io_dict[layer_name]
            break

    return io_dict, (flag_modified or local_flag_modified)

def opt_flatten_singlepass(io_dict, model, log=False):
    """ Optimize flatten layers by removing them and connecting the input and output layers. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the flatten layers
    flatten_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "flatten"]

    for layer_name in flatten_layers:
            
        remove_and_bypass_layer(io_dict, io_connect, layer_name)

        if log:
            print(f"Flatten layer \"{layer_name}\" removed.")

    return io_dict

def opt_pad_singlepass(model, io_dict, log=False):
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
            print(f"Error in opt_pad_singlepass: padding layer \"{layer_name}\" with multiple inputs or outputs.")
            exit(1)

        if io_dict[output_layer_name]["type"].lower() in comp_layers:

            io_dict[output_layer_name]["pad"] = io_dict[input_layer_name]["pad"]
            remove_and_bypass_layer(io_dict, io_connect, layer_name)
            
            if log:
                print(f"Padding layer \"{layer_name}\" merged with \"{output_layer_name}\".")

    return io_dict

def opt_relu_singlepass(model, io_dict, log=False):
    """ Optimize relu layers by merging them with the previous computation layer. """
    
    # Layers that supports activation functions
    comp_layers = ["conv"]

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the relu layers
    relu_layers = [layer_name for layer_name, layer_info in io_dict.items() if "relu" in layer_info["type"]]

    for layer_name in relu_layers:
        
        input_layer_name = prev_layers(io_dict, io_connect, layer_name)

        if input_layer_name is None or len(input_layer_name) > 1:
            print(f"Error in opt_relu_singlepass: relu layer \"{layer_name}\" with multiple inputs.")
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

def opt_add_singlepass(model, io_dict, log=False):
    """ Optimize add layers by merging them with the previous computation layer. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the add layers
    add_layers = [layer_name for layer_name, layer_info in io_dict.items() if "add" in layer_info["type"]]
    add_layers.sort(key=lambda x: io_dict[x]["layer_index"], reverse=True)

    for layer_name in add_layers:

        input_net_names = io_dict[layer_name]["input"]

        if input_net_names is None or len(input_net_names) != 2:
            print(f"Error in opt_add_singlepass: add layer \"{layer_name}\" without exactly two inputs.")
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

def dag_sorting(model, io_dict):
    """ Sort the layers of the model in a Directed Acyclic Graph (DAG). """

    io_connect = extract_connections(model, io_dict)

    start_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "produce"]

    if len(start_layers) > 1:
        print("Error in dag_sorting: multiple start layers.")
        exit(1)
    else:
        start_layer = start_layers[0]
    
    DAG = {}
    node_list = [start_layer]
    mark_set = set()
    mark_set.add(start_layer)
    while len(node_list) != 0:
        current_node = node_list.pop()
        output_nodes = next_layers(io_dict, io_connect, current_node)
        input_nodes = prev_layers(io_dict, io_connect, current_node)

        # Save node in DAG
        if current_node not in DAG.keys():
            if input_nodes is None:
                input_nodes = []
            if output_nodes is None:
                output_nodes = []
            DAG[current_node] = {"input":input_nodes, "output":output_nodes}
            print(f"Adding {current_node} to DAG with input {input_nodes} and output {output_nodes}")

        if output_nodes is not None:
            for node in output_nodes:
                node_list.append(node)
                mark_set.add(node)

    for node in DAG.keys():
        print(f"{node}: {DAG[node]}")

    node_list = [start_layer]
    level = 0
    io_dict[current_node]["layer_index"] = level
    while len(node_list) != 0:
        current_node = node_list.pop()
        used = False

        for node in DAG[current_node]["output"]:
            print(f"Removing {current_node} from {node} in {DAG[node]['input']}")
            DAG[node]["input"].remove(current_node)

            if len(DAG[node]["input"]) == 0:
                node_list.append(node)
                io_dict[node]["layer_index"] = level + 1
                used = True

        if used:
            level += 1
    
    for layer_name, layer_info in io_dict.items():
        if "layer_index" in layer_info.keys():
            print(f"{layer_name}: {layer_info['layer_index']}")

    return io_dict

def opt_merge_pointwise(model, io_dict, log=False):
    """ Merge pointwise layers in skip connections with computation layer acting on the same tensor. """

    io_connect = extract_connections(model, io_dict)

    # Retrieve all the pointwise layers
    pointwise_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] == "conv" and layer_info["fh"] == 1 and layer_info["fw"] == 1]

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

    # Retrieve all the layers without input quantization
    accepted_layers = ["conv", "pool", "add"]
    no_input_quant_layers = [layer_name for layer_name, layer_info in io_dict.items() if layer_info["type"] in accepted_layers and layer_info["input_quant"] is None]

    for layer_name in no_input_quant_layers:
            
        # Retrieve the producers of the input tensor
        input_tensor_nets = io_dict[layer_name]["input"]
        for net in input_tensor_nets:
           
            print(f"Propagating quantization from \"{net}\" to \"{layer_name}\".")
            produce_node = io_connect[net][0][0]
            if io_dict[produce_node]["output_quant"] is not None:
                io_dict[layer_name]["input_quant"] = io_dict[produce_node]["output_quant"]
                if log:
                    print(f"Propagating quantization from \"{produce_node}\" to \"{layer_name}\".")
                else:
                    print(f"Error in propagate_quant: no quantization propagated from \"{produce_node}\" to \"{layer_name}\".")

    return io_dict
                
def opt_steps(
    inferred_model,
    io_dict,
    init_info
):
    flag_modified0 = False
    flag_modified1 = False
    flag_modified2 = False
    while True:
        flag_modified0 = False

        while True:
            flag_modified1 = False

            while True:
                flag_modified2 = False

                # io_dict, flag_modified2 = merge_quant(
                #     inferred_model,
                #     io_dict,
                #     init_info,
                #     flag_modified2
                # )
                
                sanity_check(inferred_model, io_dict, "after merge_quant")

                # io_dict = assign_quant(
                #     inferred_model,
                #     io_dict
                # )

                io_dict, flag_modified2 = opt_flatten(
                    io_dict,
                    inferred_model,
                    flag_modified2
                )

                sanity_check(inferred_model, io_dict, "after flatten")

                io_dict, flag_modified2 = opt_quant(
                    inferred_model,
                    io_dict,
                    init_info,
                    flag_modified2,
                    True
                )

                sanity_check(inferred_model, io_dict, "after opt_quant")

                io_dict, flag_modified2 = opt_relu(
                    inferred_model,
                    io_dict,
                    flag_modified2
                )

                io_dict, flag_modified2 = opt_pad(
                    inferred_model,
                    io_dict,
                    flag_modified2
                )
                
                sanity_check(inferred_model, io_dict, "after opt_pad")

                if (not flag_modified2):
                    break
            
            flag_modified1 = flag_modified1 or flag_modified2

            io_dict, flag_modified1 = opt_relu(
                inferred_model,
                io_dict,
                flag_modified1
            )

            quant_info = extract_quant_info(
                inferred_model,
                io_dict,
                init_info
            )

            io_dict, flag_modified1 = opt_skip_quant(
                inferred_model,
                io_dict,
                init_info,
                flag_modified1
            )

            io_dict, flag_modified1 = opt_add(
                inferred_model,
                io_dict,
                flag_modified1
            )

            if (not flag_modified1):
                break
                
        flag_modified0 = flag_modified0 or flag_modified1

        io_dict, flag_modified0 = opt_merge_conv(
            inferred_model,
            io_dict,
            flag_modified0
        )

        io_dict, flag_modified0 = opt_skip(
            inferred_model,
            io_dict,
            flag_modified0
        )

        if (not flag_modified0):
            break

    return io_dict

def opt_step_singlepass(
    inferred_model,
    io_dict,
    init_info,
    log=False
):

    # Layer optimizations without quantizations
    io_connect = extract_connections(inferred_model, io_dict)
    for layer_name, layer_info in io_dict.items():
        print(f"Layer: {layer_name} next: {next_layers(io_dict, io_connect, layer_name)} prev: {prev_layers(io_dict, io_connect, layer_name)}")
        print(f"Layer: {layer_name} next: {io_dict[layer_name]['output']} prev: {io_dict[layer_name]['input']}")
    
    io_dict = opt_flatten_singlepass(
        io_dict,
        inferred_model,
        log
    )

    io_dict = opt_pad_singlepass(
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
    
    io_dict = opt_add_singlepass(
        inferred_model,
        io_dict,
        log
    )

    if (check_dangling_add(io_dict)):
        exit(-1)

    io_dict = opt_relu_singlepass(
        inferred_model,
        io_dict,
        log
    )

    if (check_dangling_relu(io_dict)):
        exit(-1)

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
    
    return io_dict
