import os
import sys
import time
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
    file_name="network",
    off_chip_storage=False,
    board="ULTRA96v2",
    dynamic_init=False,
    uram_storage=False,
    object_detection=False,
    anchors=[],
    prj_root="/tmp",
    transform=False,
    generate_report_file="tmp.rpt"
):


    # Cases in which a master axi interface is needed
    ap_ctrl_chain = off_chip_storage

    read_width = 8

    inferred_model = model.transform(infer_shapes.InferShapes())

    init_info = {}

    for info in model.graph.initializer:
        info_name = info.name.replace(".", "_")
        init_info[info_name] = info

    time_list = []
    time_list.append((time.time(), "Start"))

    io_dict = graph_info(
        inferred_model,
        init_info,
        object_detection,
        anchors,
        transform=transform
    )

    time_list.append((time.time(), "Graph info"))

    io_dict = opt_steps(
        inferred_model,
        io_dict,
        init_info
    )

    time_list.append((time.time(), "Opt steps"))

    io_dict = weights_quant(
        model,
        io_dict
    )

    time_list.append((time.time(), "Weights quant"))

    io_dict = balance_computations.ilp(
        io_dict,
        off_chip_storage,
        inferred_model,
        file_name,
        board,
        generate_report_file,
        prj_root=prj_root
    )

    time_list.append((time.time(), "Balance computations"))

    io_dict = compute_buffers(
        inferred_model,
        io_dict
    )

    time_list.append((time.time(), "Compute buffers"))

    io_dict = hw_quant(
        model,
        io_dict
    )

    time_list.append((time.time(), "HW quant"))

    io_dict = weights.weights_info(
        inferred_model,
        io_dict,
        init_info,
        off_chip_storage,
        dynamic_init,
        uram_storage
    )

    time_list.append((time.time(), "Weights info"))

    if off_chip_storage:
        io_dict = balance_reuse.ilp(
            io_dict
        )

    # 2 times to be sure that both weights and conv are updated
    io_dict = share_reuse(
        inferred_model,
        io_dict
    )

    time_list.append((time.time(), "Share reuse 1"))

    io_dict = share_reuse(
        inferred_model,
        io_dict
    )

    time_list.append((time.time(), "Share reuse 2"))

    io_dict = rename_nodes(
        io_dict
    )

    time_list.append((time.time(), "Rename nodes"))

    io_dict = rename_edges(
        model,
        io_dict
    )
    
    time_list.append((time.time(), "Rename edges"))

    main.write(
        io_dict,
        file_name,
        ap_ctrl_chain,
        object_detection=object_detection,
        dynamic_init=dynamic_init,
        prj_root=prj_root,
    )

    time_list.append((time.time(), "Main"))

    weights.write(
        io_dict,
        file_name,
        board,
        uram_storage,
        generate_report_file,
        prj_root=prj_root
    )
    
    time_list.append((time.time(), "Weights"))

    sim.write(
        io_dict,
        file_name,
        dynamic_init=dynamic_init,
        prj_root=prj_root
    )

    time_list.append((time.time(), "Sim"))

    for i in range(1, len(time_list)):
        print(f"{time_list[i][1]}: {(time_list[i][0] - time_list[i-1][0]):.2f}s")

