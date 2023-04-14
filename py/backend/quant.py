import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.graph import *

def compute_out_quant(
        actscale=None,
        wscale=None,
        scale_factor=None,
        clip_factor=None,
        mask_factor=None,
        signed=False
    ):

        if (actscale is not None):
            off = -1*(actscale + wscale)
        else:
            off = 0

        diff_scale = off + scale_factor

        reduced_clip = -1 * (clip_factor - actscale)
        reduced_clip = reduced_clip + 7
        reduced_clip = int(2**reduced_clip)-1
        # TODO: Add bit width to generalize this computation
        if signed and reduced_clip > 127:
            reduced_clip = 127

        reduced_mask = (mask_factor - scale_factor)
        reduced_mask = 0xff-(int(2**reduced_mask)-1)

        return diff_scale, reduced_clip, reduced_mask

def compute_in_quant(
        actscale=None,
        in_scale_factor=None
    ):

        diff_scale =  in_scale_factor - actscale

        reduced_clip = diff_scale
        reduced_clip = reduced_clip + 7
        reduced_clip = int(2**reduced_clip)-1

        return diff_scale, reduced_clip

def hw_quant(model, io_dict):
    
    io_connect = extract_connections(model, io_dict)

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        is_quant0 = 'quant' in io_dict[layer_in_name].keys()
        is_weight = 'quant' in layer_in_name.lower()
        is_not_bias = 'bias' not in io_dict[layer_in_name]["input"][0].lower()

        if layer_out_name != "ConsumeStream":
            is_quant1 = 'quant' in io_dict[layer_out_name].keys()
            if is_quant0 and is_quant1:
                in_scale_factors = io_dict[layer_in_name]["scale_factor"]

                # Multiple outputs from previous layer in case of skip 
                # connections
                in_net_names = io_dict[layer_in_name]["output"]
                scale_index0 = in_net_names.index(net_name)
                scale_factor0 = in_scale_factors[scale_index0]

                io_dict[layer_out_name]["actscale"].append(scale_factor0)
            elif is_weight and is_not_bias:
                scale_factor = io_dict[layer_in_name]["scale_factor"]
                io_dict[layer_out_name]["wscale"].append(scale_factor)


    return io_dict

def merge_quant(io_dict, quant_info, inherit_quant=False):

    # Merging consecutive quantizations
    new_quant_info = {}
    while True:

        remove_node = []

        for name, node in quant_info.items():

            keep_elem = []

            # Creating new node for the quantization info
            new_node = {}
            new_node.setdefault("seq", [])
            new_node.setdefault("seq_scale", [])
            new_node.setdefault("seq_out", [])
            new_node.setdefault("seq_clip", [])
            new_node.setdefault("seq_mask", [])
            new_node.setdefault("changed", False)
            new_node.setdefault("removed", [])
            new_node.setdefault("clip", [])
            new_node.setdefault("mask", [])

            # Looking for node with in input the previous output
            for i, output in enumerate(node["seq_out"]):
                if output in quant_info.keys():
                    # Avoid splits with unbalanced quantizations, i.e
                    # not merging in case of branches
                    single_quant = len(quant_info[output]["seq_out"]) < 2
                    if (len(quant_info[output]["others"]) == 0) and single_quant:
                        for j, new_output in enumerate(quant_info[output]["seq_out"]):
                            new_node_name = quant_info[output][new_output]
                            new_scale = quant_info[output]["seq_scale"][j]
                            new_node[new_output] = new_node_name
                            new_node["seq"].append(new_node_name)
                            new_node["seq_scale"].append(new_scale)
                            new_node["seq_out"].append(new_output)
                            new_node["changed"] = True

                            # Propagating clip, the smaller one must be kept
                            new_clip = quant_info[output]["seq_clip"][j]
                            clip_index = node["seq_out"].index(output)
                            old_clip = node["seq_clip"][clip_index]
                            if old_clip < new_clip:
                                new_clip = old_clip
                            new_node["seq_clip"].append(new_clip)

                            # Propagating mask, the higher one must be kept
                            new_mask = quant_info[output]["seq_mask"][j]
                            mask_index = node["seq_out"].index(output)
                            old_mask = node["seq_mask"][mask_index]
                            if old_mask > new_mask:
                                new_mask = old_mask
                            new_node["seq_mask"].append(new_mask)

                        new_node["removed"].append(quant_info[output]["seq"][0])
                        remove_node.append(output)
                    else:
                        keep_elem.append(i)
                else:
                    keep_elem.append(i)

            if len(remove_node) == 0:
                continue

            for i in keep_elem:
                output = node["seq_out"][i]
                new_node[output] = quant_info[name][output]
                new_node["seq"].append(quant_info[name]["seq"][i])
                new_node["seq_scale"].append(quant_info[name]["seq_scale"][i])
                new_node["seq_out"].append(quant_info[name]["seq_out"][i])
                new_node["seq_clip"].append(quant_info[name]["seq_clip"][i])
                new_node["seq_mask"].append(quant_info[name]["seq_mask"][i])

            quant_info[name] = new_node

            # Every time there is a change the dependencies must be evaluated
            # from sratch

            if len(remove_node) > 0:
                break
        
        # If there are other nodes attached to this quantization it should not 
        # be removed from database
        if len(remove_node) > 0:
            for output in remove_node:
                del quant_info[output]
        else:
            break

    remove_node = []
    for name, node in io_dict.items():
        if "quant" in name.lower():
            new_output  = []
            new_scale   = []
            new_clip   = []
            new_mask   = []
            # The quant_info database is already selecting the point where 
            # quant layers can be merged, for this reason only the changed layers
            # should be updated
            input_name = node["input"][0]
            if input_name in quant_info.keys():
                if quant_info[input_name]["changed"]:
                    seq = quant_info[input_name]["seq"]
                    seq_out = quant_info[input_name]["seq_out"]
                    seq_scale = quant_info[input_name]["seq_scale"]
                    seq_clip = quant_info[input_name]["seq_clip"]
                    seq_mask = quant_info[input_name]["seq_mask"]
                    for i, next_output in enumerate(seq_out):
                        new_output.append(next_output)
                        new_scale.append(seq_scale[i])
                        new_clip.append(seq_clip[i])
                        new_mask.append(seq_mask[i])
                    for rem_name in quant_info[input_name]["removed"]:
                        remove_node.append(rem_name)

            if len(new_output) > 0:
                io_dict[name]["output"] = new_output
                io_dict[name]["scale"] = new_scale
                io_dict[name]["clip"] = new_clip
                io_dict[name]["mask"] = new_mask

    # print(remove_node)
    for name in remove_node:
        del io_dict[name]

    return io_dict

def extract_quant_info(model, io_dict, init_info):

    quant_info = {}

    last_input = None
    last_output = None
    quant_dict = {}

    for node_name, node in io_dict.items():
        if 'quant' == node["type"]:

            # Saving connections and scale factors for all the quantizations
            scale_name   = io_dict[node_name]["input"][1]
            scale_info   = init_info[scale_name]
            scale_factor = numpy_helper.to_array(scale_info)
            scale_factor = np.log2(scale_factor)
            clip = node["clip"]
            mask = node["mask"]

            signed = node["signed"]

            quant_info.setdefault(node["input"][0], {})
            quant_info[node["input"][0]].setdefault("seq", [])
            quant_info[node["input"][0]].setdefault("seq_scale", [])
            quant_info[node["input"][0]].setdefault("seq_out", [])
            quant_info[node["input"][0]].setdefault("others", [])
            quant_info[node["input"][0]].setdefault("others_scale", [])
            quant_info[node["input"][0]].setdefault("changed", False)
            quant_info[node["input"][0]].setdefault("removed", [])
            quant_info[node["input"][0]].setdefault("signed", signed)
            quant_info[node["input"][0]].setdefault("seq_clip", [])
            quant_info[node["input"][0]].setdefault("seq_mask", [])

            quant_info[node["input"][0]][node["output"][0]] = node_name
            quant_info[node["input"][0]]["seq"].append(node_name)
            quant_info[node["input"][0]]["seq_scale"].append(scale_factor)
            quant_info[node["input"][0]]["seq_out"].append(node["output"][0])
            quant_info[node["input"][0]]["seq_clip"].append(clip)
            quant_info[node["input"][0]]["seq_mask"].append(mask)

        else:

            # Adding direct connections to other layers from the quant
            # If the quant output is connected with other layers the
            # quantization cannot be completely pruned
            for input in node["input"]:
                if input in quant_info.keys():
                    scale_factor = quant_info[input]["seq_scale"][0]
                    quant_info[input]["others"].append(node_name)

    return quant_info

