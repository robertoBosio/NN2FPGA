import os
import sys
#import onnx
import qonnx
from qonnx.transformation import infer_shapes

from backend.quant import *
from backend.graph import *
from backend.opt import *
import backend.layers.weights as weights
import backend.balance_computations as balance_computations
import backend.balance_reuse as balance_reuse
import backend.main as main
import backend.sim as sim

from onnx import numpy_helper
import numpy as np

# Expects an ONNX model
def write_network(
    model,
    file_name="Network",
    off_chip_storage=False,
    board="ULTRA96v2",
    uram_storage=False,
):

    read_width = 8

    inferred_model = model.transform(infer_shapes.InferShapes())

    init_info = {}

    for info in model.graph.initializer:
        info_name = info.name.replace(".", "_")
        init_info[info_name] = info

    io_dict = graph_info(
        inferred_model,
        init_info
    )

    io_dict = opt_steps(
        inferred_model,
        io_dict,
        init_info
    )

    io_dict = balance_computations.ilp(
        io_dict,
        off_chip_storage,
        board
    )

    io_dict = hw_quant(
        model,
        io_dict
    )

    io_dict = weights.weights_info(
        inferred_model,
        io_dict,
        init_info,
        off_chip_storage
    )

    if off_chip_storage:
        io_dict = balance_reuse.ilp(
            io_dict
        )

    # 2 times to be sure that both weights and conv are updated
    io_dict = share_reuse(
        inferred_model,
        io_dict
    )

    io_dict = share_reuse(
        inferred_model,
        io_dict
    )

    io_dict = rename_nodes(
        io_dict
    )

    io_dict = rename_edges(
        model,
        io_dict
    )

    main.write(
        io_dict,
        file_name,
        off_chip_storage,
        uram_storage
    )

    weights.write(
        io_dict,
        file_name,
        uram_storage
    )

    sim.write(
        io_dict,
        file_name,
        uram_storage
    )

