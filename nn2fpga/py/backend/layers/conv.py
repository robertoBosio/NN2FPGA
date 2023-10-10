import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import backend.quant
from backend.layers.quant import get_quant_type, get_quant_constant

def info(io_dict, node, node_name, init_info, tensors_info, enable_ws):

    attributes = getattr(node, "attribute" )
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value
    fh       = getattr(attributes[2], 'ints')[0]
    fw       = getattr(attributes[2], 'ints')[1]
    stride   = getattr(attributes[4], 'ints')[0]
    pad      = getattr(attributes[3], 'ints')[0]
    is_1x1   = (fh == 1) and (fw == 1)
    kernel   = fh*fw
    img_ch   = ich*och
    relu     = False
    add      = False
    in_scale_factor = [None]
    in_bits = [None]

    groups = getattr(attributes[1], 'ints')

    if (groups == och):
        depth = 1
    else:
        depth = 0

    if depth == 0:
        total    = 1/(oh*ow*och*ich)
        total_log = 2*oh*ow*och*ich*fh*fw
    else:
        total    = 1/(oh*ow*och)
        total_log = 2*oh*ow*och*fh*fw

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
    io_dict[node_name]["is_1x1"] = is_1x1
    io_dict[node_name]["total"]  = total
    io_dict[node_name]["total_log"]  = total_log
    io_dict[node_name]["kernel"] = kernel
    io_dict[node_name]["img_ch"] = img_ch
    # Reuse is generic
    io_dict[node_name]["enable_ws"] = enable_ws
    io_dict[node_name]["reuse"]  = 1
    # Ws are the operations in parallel
    io_dict[node_name]["ws"]     = 1
    io_dict[node_name]["ws_out"] = 1
    io_dict[node_name]["relu"]   = relu
    io_dict[node_name]["add"]    = add
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["bits"]    = 0
    io_dict[node_name]["in_bits"] = in_bits
    io_dict[node_name]["type"]   = 'conv'
    io_dict[node_name]["wbias"]  = len(node.input) > 2
    io_dict[node_name]["wbits"]  = []
    io_dict[node_name]["actbits"] = []
    io_dict[node_name]["wscale"]  = []
    io_dict[node_name]["actscale"] = []
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["in_ops"] = 1
    io_dict[node_name]["ich_ops"] = 1
    io_dict[node_name]["depth"] = depth

    return io_dict

def parse_comp(name, node):
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    weight_name = node["input"][1]

    # If no batchnorm merge then there is no bias
    has_bias = len(node["input"]) > 2
    if (has_bias):
        bias_name   = node["input"][2]

    if (node["merge_1x1"]):
        weight_1x1_name = node["input"][3]
        bias_1x1_name = node["input"][4]

    if (node["add"]):
        add_name = node["input"][3]
        add_type_name = add_name
        # add_type_name = add_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    if (node["has_forward"]):
        forward_name = node["output"][1]
    if (node["merge_1x1"]):
        output_1x1_name = node["output"][1]
        output_1x1_type_name = output_1x1_name.replace("_skip", "")

    block = {}
    block["func"] = "conv_comp"

    # Template parameters
    block["template"] = []
    if (node["is_1x1"]):
        block["template"].append("t_%s_lb_struct" % input_type_name)
        block["template"].append("t_%s_lb" % input_type_name)
        block["template"].append("t_%s_vector" % input_type_name)
    else:
        block["template"].append("t_%s_window_struct" % input_type_name)
        block["template"].append("t_%s_window" % input_type_name)
        block["template"].append("t_%s_reduce" % input_type_name)
    block["template"].append("t_%s" % weight_name)
    block["template"].append("t_%s_st" % weight_name)
    if (has_bias):
        block["template"].append("t_%s" % bias_name)
    else:
        block["template"].append("std::nullptr_t")

    if (node["add"]):
        block["template"].append("t_%s_struct" % add_type_name)
        # TODO: add case of 1x1 forward
        # block["template"].append("t_%s_vector" % add_type_name)
        block["template"].append("t_%s_vector" % add_type_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    if (node["has_forward"]):
        block["template"].append("t_%s_struct" % forward_name)
    else:
        block["template"].append("std::nullptr_t")

    if (node["in_scale_factor"][0] is not None):
        block["template"].append("t_%s_mod" % input_type_name)
    else:
        block["template"].append("std::nullptr_t")

    if (node["merge_1x1"]):
        block["template"].append("t_%s_1x1" % input_type_name)
        block["template"].append("t_%s" % weight_1x1_name)
        block["template"].append("t_%s_st" % weight_1x1_name)
        block["template"].append("t_%s" % bias_1x1_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    block["template"].append("t_%s_acc_struct" % output_name)
    block["template"].append("t_%s_acc" % output_name)

    if (node["merge_1x1"]):
        block["template"].append("t_%s_acc_struct" % output_1x1_name)
        block["template"].append("t_%s_acc" % output_1x1_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    block["template"].append("t_%s_clip" % output_type_name)
    block["template"].append("t_%s_mask" % output_type_name)
    if (node["merge_1x1"]):
        block["template"].append("t_%s_struct" % output_1x1_name)
        block["template"].append("t_%s" % output_1x1_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_index" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_%s_ich_ops" % name)
    if (node["add"]):
        block["template"].append("c_%s_add_ops" % add_name)
    else:
        block["template"].append("1")
    block["template"].append("c_%s_relu" % name)
    block["template"].append("c_%s_reuse" % name)
    block["template"].append("c_%s_ws" % name)
    block["template"].append("c_%s_ws_out" % name)

    ##############################################################################
    # PACKING: providing info on quantization from template because
    # ap_fixed methods are not available at compile time and the
    # synthesizer gives an error
    simd_bits = 2
    simd = int(np.log2(node["kernel"])/simd_bits) + (0 != (np.log2(node["kernel"]) - int(np.log2(node["kernel"]))))
    mask = (1 << (simd-1)) - 1;
    if (node["in_scale_factor"][0] is not None):
        abits, aibits = get_quant_constant(node["signed"], node["in_bits"][0], node["in_scale_factor"][0])
    else:
        abits, aibits = get_quant_constant(node["signed"], node["actbits"][0], node["actscale"][0])
    block["template"].append("%0d" % abits)
    block["template"].append("%0d" % aibits)
    abits, aibits = get_quant_constant(node["signed"], node["wbits"][0], node["wscale"][0])
    block["template"].append("%0d" % abits)
    block["template"].append("%0d" % aibits)
    block["template"].append("%0d" % simd_bits)
    block["template"].append("%0d" % simd)
    block["template"].append("%0d" % mask)

    #############################################################################
    # PACKING: providing info on quantization from template because
    # ap_fixed methods are not available at compile time and the
    # synthesizer gives an error, for 1x1 conv the dimension of the simd
    # partial results array is fixed at 1

    simd_bits = 2
    simd = 1
    mask = (1 << (simd-1)) - 1;
    if (node["merge_1x1"]):
        if (node["in_scale_factor"][1] is not None):
            abits, aibits = get_quant_constant(node["signed"], node["bits"][1], node["in_scale_factor"][1])
        else:
            abits, aibits = get_quant_constant(node["signed"], node["actbits"][0], node["actscale"][0])
    else:
        abits = 0
        aibits = 0
    block["template"].append("%0d" % abits)
    block["template"].append("%0d" % aibits)

    if (node["merge_1x1"]):
        abits, aibits = get_quant_constant(node["signed"], node["wbits"][1], node["wscale"][1])
    else:
        abits = 0
        aibits = 0
    block["template"].append("%0d" % abits)
    block["template"].append("%0d" % aibits)
    block["template"].append("%0d" % simd_bits)
    block["template"].append("%0d" % simd)
    block["template"].append("%0d" % mask)
    ####################################################################################
    block["template"].append("%0d" % node["depth"])

    acc_type = get_quant_type(True, 32, node["actscale"][0]+node["wscale"][0], acc_reg=True)
    block["defines"] = {}
    block["defines"]["t_%s_acc" % output_name] = ["type", acc_type]
    block["defines"]["t_%s_acc_struct" % output_name] = [
        "struct",
        [["data", "t_%s_acc" % output_name], ["last", "bool"]]
    ]

    output_type = get_quant_type(node["signed"], node["bits"][0], node["scale_factor"][0])
    output_type_clip = get_quant_type(node["clip_signed"][0], node["bits"][0], node["clip_factor"][0])
    output_type_mask = get_quant_type(node["mask_signed"][0], node["bits"][0], node["mask_factor"][0])

    # TODO: check type declaration
    # input window type declaration
    input_reduce_type = "std::array<t_%s, %0d>" % (input_name, node["ich_ops"])
    block["defines"]["t_%s_reduce" % input_name] = ["type", input_reduce_type]
    input_window_type = "std::array<t_%s_reduce, %0d>" % (input_name, node["fh"]*(node["fw"]+(node["ws"]-1)*node["stride"]))
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
    if (node["has_forward"]):
        block["defines"]["t_%s_vector" % forward_name] = ["type", input_reduce_type]
        block["defines"]["t_%s_struct" % forward_name] = [
            "struct",
            [["data", "std::array<t_%s_reduce, 1>" % input_name], ["last", "bool"]]
        ]

    # Output type declaration
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    output_vector_type = "std::array<%s, %0d>" % (output_type, node["ops"])
    block["defines"]["t_%s_vector" % output_name] = ["type", output_vector_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, 1>" % output_name], ["last", "bool"]]
    ]
    block["defines"]["t_%s_clip" % output_name] = ["type", output_type_clip]
    block["defines"]["t_%s_mask" % output_name] = ["type", output_type_mask]

    if (node["merge_1x1"]):
        output_type_1x1 = get_quant_type(True, node["bits"][1], node["scale_factor"][1])
        # TODO: implement array of signed values for multi-output conv
        block["defines"]["t_%s" % output_1x1_name] = ["type", output_type_1x1]
        output_vector_type = "std::array<t_%s, %0d>" % (output_1x1_name, node["ops"])
        block["defines"]["t_%s_vector" % output_1x1_name] = ["type", output_vector_type]
        block["defines"]["t_%s_struct" % output_1x1_name] = [
            "struct",
            [["data", "std::array<t_%s_vector, 1>" % output_1x1_name], ["last", "bool"]]
        ]

    if (node["in_scale_factor"][0] is not None):
        input_type_mod = get_quant_type(True, node["in_bits"][0], node["in_scale_factor"][0])

    else:
        input_type_mod = "std::nullptr_t"

    block["defines"]["t_%s_mod" % input_name] = ["type", input_type_mod]

    if (node["merge_1x1"]):

        if (node["in_scale_factor"][1] is not None):
            input_1x1_type = get_quant_type(True, node["bits"][1], node["in_scale_factor"][1])
        else:
            input_1x1_type = "std::nullptr_t"
        block["defines"]["t_%s_1x1" % input_name] = ["type", input_1x1_type]

        if (node["in_scale_factor"][1] is not None):
            acc_type_1x1 = get_quant_type(True, 32, node["in_scale_factor"][1]+node["wscale"][1]-2)
        else:
            acc_type_1x1 = get_quant_type(True, 32, node["actscale"][0]+node["wscale"][1])

        block["defines"]["t_%s_acc" % output_1x1_name] = ["type", acc_type_1x1]
        block["defines"]["t_%s_acc_struct" % output_1x1_name] = [
            "struct",
            [["data", "t_%s_acc" % output_1x1_name], ["last", "bool"]]
        ]

        block["defines"]["c_%s_add_ops" % output_1x1_name]         = ["const", node["ops"]]

    block["defines"]["c_%s_ich" % name]            = ["const", node["ich"]]
    block["defines"]["c_%s_och" % name]            = ["const", node["och"]]
    block["defines"]["c_%s_iw" % name]             = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % name]             = ["const", node["ih"]]
    block["defines"]["c_%s_fw" % name]             = ["const", node["fw"]]
    block["defines"]["c_%s_fh" % name]             = ["const", node["fh"]]
    block["defines"]["c_%s_ow" % name]             = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % name]             = ["const", node["oh"]]
    block["defines"]["c_%s_relu" % name]           = ["const", int(node["relu"])]
    block["defines"]["c_%s_stride" % name]         = ["const", node["stride"]]
    block["defines"]["c_%s_pad" % name]            = ["const", node["pad"]]
    block["defines"]["c_%s_ops" % name]            = ["const", node["ops"]]
    block["defines"]["c_%s_in_ops" % name]         = ["const", node["in_ops"]]
    block["defines"]["c_%s_ich_ops" % name]        = ["const", node["ich_ops"]]
    block["defines"]["c_%s_index" % name]          = ["const", node["kernel"]]
    block["defines"]["c_%s_reuse" % name]          = ["const", node["reuse"]]
    block["defines"]["c_%s_ws" % name]             = ["const", node["ws"]]
    block["defines"]["c_%s_ws_out" % name]         = ["const", node["ws_out"]]
    if (node["has_forward"]):
        block["defines"]["c_%s_add_ops" % forward_name]         = ["const", node["ich_ops"]]

    block["args"] = []

    if node["pad"] == 0:
        block["args"].append("s_%s_pre_pad" % input_name)
    else:
        block["args"].append("s_%s_compute" % input_name)

    block["args"].append("s_%s" % weight_name)
    if (has_bias):
        block["args"].append("s_%s" % bias_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    if (node["merge_1x1"]):
        block["args"].append("s_%s" % weight_1x1_name)
        block["args"].append("s_%s" % bias_1x1_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    if (node["add"]):
        block["args"].append("s_%s" % add_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    if (node["has_forward"]):
        block["args"].append("s_%s" % forward_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    block["args"].append("s_%s" % output_name)

    if (node["merge_1x1"]):
        block["args"].append("s_%s" % output_1x1_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ws"]
    block["declare"].append(declare)


    if (node["merge_1x1"]):
        declare = {}
        declare["name"] = "s_%s" % output_1x1_name
        declare["type"] = "t_%s_struct" % output_1x1_name
        declare["is_array"] = True
        declare["dim"] = node["ws"]
        block["declare"].append(declare)

    if (node["has_forward"]):
        declare = {}
        declare["name"] = "s_%s" % forward_name
        declare["type"] = "t_%s_struct" % forward_name
        declare["is_array"] = True
        declare["dim"] = node["ws"]
        block["declare"].append(declare)

    block["pragma"] = []

    # depth = int(node["och"]/node["ops"] + 1)

    if (node["has_forward"]):
        # First two lines
        depth = int((node["fh"]-1)*node["iw"]*node["ich"]/node["ich_ops"])
        # first two pixels of the third line
        depth += int((node["fw"]-1)*node["ich"]/node["ich_ops"])
        depth += node["och"]+1
        pragma = {}
        pragma["name"] = "stream"
        pragma_name = "s_%s" % forward_name
        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    depth = int(node["och"]/node["ops"])*node["ws"] + 1

    pragma = {}
    pragma["name"] = "stream"
    pragma_name = "s_%s" % (output_name)
    options = [
        ["variable", pragma_name],
        ["depth", depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    if (node["merge_1x1"]):
        pragma = {}
        pragma["name"] = "stream"
        pragma_name = "s_%s" % (output_1x1_name)

        depth = node["ow"]*int(node["och"]/node["ops"])*(node["fh"]-1)-node["ich"]

        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    return [block]

def parse_split(name, node):

    blocks = []

    blocks = blocks + parse_comp(name, node)
    # blocks = blocks + parse_wout(name, node)

    return blocks

def parse(name, node, wrapper=False):

    return parse_split(name, node)
