import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.graph import extract_connections

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
    
    no_break = False
    # Compute for each tensor the distance between producer and consumer
    while not no_break:

        io_connect = extract_connections(model, io_dict)
        print(io_connect)

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

def opt_quant(model, io_dict, quant_info):
    
    io_connect = extract_connections(model, io_dict)

    for net_name, layers in io_connect.items():
        layer_in_name = layers[0][0]
        layer_out_name = layers[1][0]

        start_conv = 'conv' in layer_in_name.lower()
        end_quant = 'quant' in layer_out_name.lower()

        # If true the relu can be absorbed into convolution
        if start_conv and end_quant:
            out_name = io_dict[layer_out_name]["output"][0]

            # Scale factor is equal to the one of the quantization in output
            # In case there are multiple output nodes, all of the must be saved

            scale_factor = quant_info[net_name]["seq_scale"]

            io_dict[layer_in_name]["quant"] = True
            io_dict[layer_in_name]["scale_factor"] = scale_factor
            io_dict[layer_in_name]["output"][0] = out_name
            del io_dict[layer_out_name]

    return io_dict

