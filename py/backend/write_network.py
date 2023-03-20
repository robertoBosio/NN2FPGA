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

from backend.quant import *
from backend.graph import *
from backend.opt import *

from onnx import numpy_helper
import numpy as np

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

    read_width = 8

    inferred_model = model.transform(infer_shapes.InferShapes())

    init_info = {}

    for info in model.graph.initializer:
        init_info[info.name] = info

    io_dict = graph_info(
        inferred_model,
        init_info
    )

    # TODO: Check that merged quantizations are correct

    io_dict = opt_steps(
        inferred_model,
        io_dict,
        init_info
    )

    for name, info in io_dict.items():
        print(name, info)

    # conv_relu, additional_ports, additional_ports_info, parallel_ops, weights_export, reuse = main.write(
    #     inferred_model,
    #     tensors_info,
    #     quant_info,
    #     weights_info,
    #     skip_connections_info,
    #     bias_info,
    #     relu_info,
    #     split_info,
    #     flatten_info,
    #     reordered_layers,
    #     off_chip_storage,
    #     read_width
    # )

    # # TODO: export tensors and weight info
    # # TODO: assign correct values on tensors to defines
    # # TODO: weights define
    
    # defines.write(
    #     inferred_model,
    #     tensors_info,
    #     quant_info,
    #     weights_info,
    #     skip_connections_info,
    #     bias_info,
    #     relu_info,
    #     conv_relu,
    #     flatten_info,
    #     split_info,
    #     off_chip_storage,
    #     additional_ports,
    #     parallel_ops,
    #     read_width,
    #     reuse
    # )

    # if off_chip_storage:

    #     memory.write(
    #         additional_ports,
    #         additional_ports_info,
    #         read_width
    #     )

    #     memory_defines.write(
    #         additional_ports,
    #         additional_ports_info,
    #         read_width
    #     )

    #     memory_header.write(
    #         additional_ports,
    #         weights_export,
    #         read_width
    #     )

