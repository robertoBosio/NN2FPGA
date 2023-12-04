import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from backend.layers.quant import get_quant_type
import math

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich    = getattr(input_shape, 'dim')[1].dim_value
    ih     = getattr(input_shape, 'dim')[2].dim_value
    iw     = getattr(input_shape, 'dim')[3].dim_value
    och    = getattr(output_shape, 'dim')[1].dim_value
    oh     = getattr(output_shape, 'dim')[2].dim_value
    ow     = getattr(output_shape, 'dim')[3].dim_value

    global_pool = ("Global" in node.op_type)
    adaptive =  global_pool or ('adaptive' in node_name)

    attr_dict = {}
    for attribute in attributes:
        name = getattr(attribute, "name")
        ints = getattr(attribute, "ints")
        attr_dict[name] = ints

    if (adaptive):
        fh     = ih
        fw     = iw
        stride = 1
        pad    = 0
    else:
        fh     = attr_dict["kernel_shape"][0]
        fw     = attr_dict["kernel_shape"][1]
        stride = attr_dict["strides"][0]
        pad    = attr_dict["pads"][0]

    adaptive |= (fh == iw) and (fw == ih) and (pad == 0)

    in_scale_factor = 0

    if 'max' in node.op_type.lower():
        pool     = 1

    if 'average' in node.op_type.lower():
        pool     = 0

    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["fh"]     = fh
    io_dict[node_name]["fw"]     = fw
    io_dict[node_name]["stride"] = stride
    io_dict[node_name]["pad"]    = pad
    io_dict[node_name]["pool"]   = pool
    io_dict[node_name]["type"]   = 'pool'
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["actscale"] = []
    io_dict[node_name]["is_adaptive"] = adaptive
    io_dict[node_name]["actbits"] = []
    io_dict[node_name]["actsigned"] = []
    io_dict[node_name]["ow_ops"] = 1
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["in_ops"] = 1
    io_dict[node_name]["ich_ops"] = 1
    io_dict[node_name]["total"] = 1/(oh*ow*och)
    io_dict[node_name]["total_log"] = 2*oh*ow*och*fh*fw

    return io_dict

def parse(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")

    signed = node["signed"]

    block = {}
    block["func"] = "pool_op"

    # Template parameters
    block["template"] = []
    if (node["is_adaptive"]):
        block["template"].append("t_%s_struct" % input_type_name)
        block["template"].append("t_%s" % input_type_name)
    # elif (node["pad"] == 0):
    #     block["template"].append("t_%s_lb_struct" % input_type_name)
    #     block["template"].append("t_%s_lb" % input_type_name)
    else:
        block["template"].append("t_%s_window_struct" % input_type_name)
        block["template"].append("t_%s_window" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("t_%s_acc" % name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_pad" % name)
    block["template"].append("c_%s_pool" % name)
    block["template"].append("c_%s_ow_ops" % name)
    block["template"].append("c_%s_ow_ops_out" % name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_%s_in_ops" % name)

    block["args"] = []
    if (node["is_adaptive"]):
        block["args"].append("s_%s" % input_name)
    else:
        # if node["pad"] == 0:
        #     block["args"].append("s_%s_pre_pad" % input_name)
        # else:
        block["args"].append("s_%s_compute" % input_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}
    output_type = get_quant_type(node["signed"], node["bits"][0], node["scale_factor"][0])
    block["defines"]["t_%s" % output_type_name] = ["type", output_type]
    output_vector_type = "std::array<t_%s, %s>" % (output_type_name, node["ops"])
    block["defines"]["t_%s_vector" % output_type_name] = ["type", output_vector_type]
    block["defines"]["t_%s_struct" % output_type_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, 1>" % output_type_name], ["last", "bool"]]
    ]

    input_reduce_type = "std::array<t_%s, %0d>" % (input_name, node["ich_ops"])
    block["defines"]["t_%s_reduce" % input_name] = ["type", input_reduce_type]
    input_window_type = "std::array<t_%s_reduce, %0d>" % (input_name, node["fh"]*(node["fw"]+(node["ow_ops"]-1)*node["stride"]))
    block["defines"]["t_%s_window" % input_name] = ["type", input_window_type]
    block["defines"]["t_%s_window_struct" % input_name] = [
        "struct",
        [["data", "t_%s_window" % input_name], ["last", "bool"]]
    ]
    input_lb_type = "std::array<t_%s_reduce, %0d>" % (input_name, 1)
    block["defines"]["t_%s_lb" % input_name] = ["type", input_lb_type]
    block["defines"]["t_%s_lb_struct" % input_name] = [
        "struct",
        [["data", "t_%s_lb" % input_name], ["last", "bool"]]
    ]

    if node["pool"] == 1:
        block["defines"]["t_%s_acc" % name]            = ["type", output_type]
    else:
        acc_bits = node["actbits"][0] + math.ceil(math.log2(node["fh"]*node["fw"]))
        acc_type = get_quant_type(True, acc_bits, node["actscale"][0], acc_reg=True)
        block["defines"]["t_%s_acc" % name]        = ["type", acc_type]
    block["defines"]["c_%s_ich" % name]            = ["const", node["ich"]]
    block["defines"]["c_%s_och" % name]            = ["const", node["och"]]
    block["defines"]["c_%s_iw" % name]             = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % name]             = ["const", node["ih"]]
    block["defines"]["c_%s_ow" % name]             = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % name]             = ["const", node["oh"]]
    block["defines"]["c_%s_fw" % name]             = ["const", node["fw"]]
    block["defines"]["c_%s_fh" % name]             = ["const", node["fh"]]
    block["defines"]["c_%s_stride" % name]         = ["const", node["stride"]]
    block["defines"]["c_%s_pad" % name]            = ["const", node["pad"]]
    block["defines"]["c_%s_pool" % name]           = ["const", node["pool"]]
    block["defines"]["c_%s_ow_ops" % name]         = ["const", node["ow_ops"]]
    block["defines"]["c_%s_ow_ops_out" % name]     = ["const", node["ow_ops_out"]]
    block["defines"]["c_%s_ops" % name]            = ["const", node["ops"]]
    block["defines"]["c_%s_in_ops" % name]         = ["const", node["in_ops"]]

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ow_ops"]

    block["declare"].append(declare)

    block["pragma"] = []
    pragma = {}
    pragma["name"] = "aggregate"
    options = [
        ["variable", "s_%s" % output_name],
    ]
    pragma["options"] = options

    # TODO: removed to test without pragma
    # block["pragma"].append(pragma)

    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", "s_%s" % output_name],
        ["type", "cyclic"],
        ["factor", node["ops"]//(8*8//node["bits"][0])],
    ]
    pragma["options"] = options

    # TODO: removed to test without pragma
    # block["pragma"].append(pragma)

    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s" % output_name],
        ["depth", node["och"]//node["ops"]],
        ["type", "fifo"],
    ]
    pragma["options"] = options

    block["pragma"].append(pragma)

    # FIX: Adding pragma to bind storage to SRL
    # if the depth of the fifo is small enough to
    # not justify the use of BRAM
    if (node["och"]//node["ops"] < 64):
        pragma = {}
        pragma["name"] = "bind_storage"
        options = [
            ["variable", "s_%s" % output_name],
            ["impl", "SRL"],
            ["type", "fifo"]
        ]
        pragma["options"] = options

        block["pragma"].append(pragma)

    return block
