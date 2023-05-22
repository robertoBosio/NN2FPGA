import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import backend.quant

def info(io_dict, node, node_name, init_info, tensors_info):

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
    total    = 1/(oh*ow*och*ich*fh*fw)
    kernel   = fh*fw
    img_ch   = ich*och
    reuse    = 1
    relu     = False
    add      = False
    in_scale_factor = [None]

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
    io_dict[node_name]["kernel"] = kernel
    io_dict[node_name]["img_ch"] = img_ch
    io_dict[node_name]["reuse"]  = reuse
    io_dict[node_name]["relu"]   = relu
    io_dict[node_name]["add"]    = add
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["type"]   = 'conv'
    io_dict[node_name]["wbias"]  = len(node.input) > 2
    io_dict[node_name]["wscale"] = []
    io_dict[node_name]["actscale"] = []

    return io_dict

def parse_wout(name, node):
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")
    weight_name = node["input"][1]

    signed = node["signed"]
    # If no batchnorm merge then there is no bias

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    if (node["merge_1x1"]):
        output_1x1_name = node["output"][1]
        output_1x1_type_name = output_1x1_name.replace("_skip", "")

    block = {}
    block["func"] = "StreamOutput"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    if (node["merge_1x1"]):
        block["template"].append("t_%s_struct" % output_1x1_name)
        block["template"].append("t_%s" % output_1x1_name)
    block["template"].append("t_%s_acc_struct" % output_name)
    block["template"].append("t_%s_acc" % output_name)
    if (node["merge_1x1"]):
        block["template"].append("t_%s_acc_struct" % output_1x1_name)
        block["template"].append("t_%s_acc" % output_1x1_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_index" % name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_%s_relu" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_mask" % output_name)
    block["template"].append("c_%s_shift_h" % output_name)
    block["template"].append("c_%s_shift_l" % output_name)
    if (node["merge_1x1"]):
        block["template"].append("c_%s_mask" % output_1x1_name)
        block["template"].append("c_%s_shift_h" % output_1x1_name)
        block["template"].append("c_%s_shift_l" % output_1x1_name)

    block["defines"] = {}

    if (signed):
        output_type = "int8_t"
    else:
        output_type = "uint8_t"

    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "t_%s" % output_name], ["last", "bool"]]
    ]

    if (node["merge_1x1"]):
        # TODO: implement array of signed values for multi-output conv
        block["defines"]["t_%s" % output_1x1_name] = ["type", "int8_t"]
        block["defines"]["t_%s_struct" % output_1x1_name] = [
            "struct",
            [["data", "t_%s" % output_1x1_name], ["last", "bool"]]
        ]

    # Evaluate these values both for the normal output and the pointwise merge
    # If there is a need for in place quantization, the activation scale
    # is the input scale factor
    actscale = node["actscale"][0]
    if node["in_scale_factor"][0] is not None:
        actscale = node["in_scale_factor"][0]

    diff_scale, reduced_clip, reduced_mask = backend.quant.compute_out_quant(
        actscale,
        node["wscale"][0],
        node["scale_factor"][0],
        node["clip_factor"][0],
        node["mask_factor"][0],
        signed=signed
    )

    block["defines"]["c_%s_mask" % output_name] = ["const", reduced_mask]
    block["defines"]["c_%s_shift_h" % output_name] = ["const", reduced_clip]
    block["defines"]["c_%s_shift_l" % output_name] = ["const", diff_scale]

    if (node["merge_1x1"]):
        actscale = node["actscale"][0]
        if node["in_scale_factor"][1] is not None:
          actscale = node["in_scale_factor"][1]

        diff_scale, reduced_clip, reduced_mask = backend.quant.compute_out_quant(
            actscale,
            node["wscale"][1],
            node["scale_factor"][1],
            node["clip_factor"][1],
            node["mask_factor"][1],
            signed=True
        )
        block["defines"]["c_%s_mask" % output_1x1_name] = ["const", reduced_mask]
        block["defines"]["c_%s_shift_h" % output_1x1_name] = ["const", reduced_clip]
        block["defines"]["c_%s_shift_l" % output_1x1_name] = ["const", diff_scale]

    if (node["in_scale_factor"][0] is not None):
        diff_scale, reduced_clip = backend.quant.compute_in_quant(
            node["actscale"][0],
            node["in_scale_factor"][0]
        )

    else:
        reduced_clip = 255
        diff_scale = 0
    block["defines"]["c_%s_in_shift_h" % input_name] = ["const", reduced_clip]
    block["defines"]["c_%s_in_shift_l" % input_name] = ["const", diff_scale]

    if (node["merge_1x1"]):
      if (node["in_scale_factor"][1] is not None):
          if len(node["actscale"]) == 1:
              actscale = node["actscale"][0]
          else:
              actscale = node["actscale"][1]

          diff_scale, reduced_clip = backend.quant.compute_in_quant(
              actscale,
              node["in_scale_factor"][1]
          )

      else:
          reduced_clip = 255
          diff_scale = 0
      block["defines"]["c_%s_in_shift_h_1x1" % input_name] = ["const", reduced_clip]
      block["defines"]["c_%s_in_shift_l_1x1" % input_name] = ["const", diff_scale]

    block["args"] = []
    block["args"].append("s_%s_acc" % output_name)
    if (node["merge_1x1"]):
        block["args"].append("s_%s_acc" % output_1x1_name)

    # If there is a reuse then the output must be rearranged to be channel first
    if (node["reuse"] > 1):
        block["args"].append("s_%s_p_nchw[0]" % output_name)
    else:
        block["args"].append("s_%s[0]" % output_name)

    if (node["merge_1x1"]):
        if (node["reuse"] > 1):
            block["args"].append("s_%s_p_nchw[0]" % output_1x1_name)
        else:
            block["args"].append("s_%s[0]" % output_1x1_name)

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    block["declare"] = []

    declare = {}
    if (node["reuse"] > 1):
        declare["name"] = "s_%s_p_nchw" % output_name
    else:
        declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = 1
    block["declare"].append(declare)


    if (node["merge_1x1"]):
        declare = {}
        if (node["reuse"] > 1):
            declare["name"] = "s_%s_p_nchw" % output_1x1_name
        else:
            declare["name"] = "s_%s" % output_1x1_name
        declare["type"] = "t_%s_struct" % output_1x1_name
        declare["is_array"] = True
        declare["dim"] = 1
        block["declare"].append(declare)

    block["pragma"] = []

    # FIX BUG HALF SPEED, WHEN CHANGING THE OUTPUT CHANNEL THE DEPTH MUST BE
    # PROPORTIONAL TO THE SIZE OF THE CHANGE
      # if (node["och"] > node["ich"]):
    #   depth = node["och"]
    # else:
    #   depth = 2

    # depth = node["och"]*(node["ow"]+node["fw"]-1)
    # depth = node["och"] + node["och"]*int(node["och"]/node["ich"])
    depth = node["och"] + 1

    pragma = {}
    pragma["name"] = "stream"
    if (node["reuse"] > 1):
        pragma_name = "s_%s_p_nchw" % (output_name)
    else:
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
        if (node["reuse"] > 1):
            pragma_name = "s_%s_p_nchw" % (output_1x1_name)
        else:
            pragma_name = "s_%s" % (output_1x1_name)

        if (node["reuse"] > 1):
            depth = 2
        else:
            # depth = node["ow"]*node["och"]*(node["fh"]-1)
            depth = node["ow"]*node["och"]*(node["fh"]-1)-node["ich"]

        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    return [block]

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
        add_type_name = add_name.replace("_skip", "")

    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    if (node["has_forward"]):
        forward_name = node["output"][1]
    if (node["merge_1x1"]):
        output_1x1_name = node["output"][1]
        output_1x1_type_name = output_1x1_name.replace("_skip", "")

    block = {}
    block["func"] = "ConvComp"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s" % weight_name)
    if (has_bias):
        block["template"].append("t_%s" % bias_name)
    if (node["add"]):
        block["template"].append("t_%s_struct" % add_type_name)
    if (node["merge_1x1"]):
        block["template"].append("t_%s" % weight_1x1_name)
        block["template"].append("t_%s" % bias_1x1_name)
    block["template"].append("t_%s_acc_struct" % output_name)
    block["template"].append("t_%s_acc" % output_name)
    if (node["merge_1x1"]):
        block["template"].append("t_%s_acc_struct" % output_1x1_name)
        block["template"].append("t_%s_acc" % output_1x1_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_index" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_%s_reuse" % name)

    if (node["in_scale_factor"][0] is not None) or (node["merge_1x1"]):
        block["template"].append("c_%s_in_shift_h" % input_name)
        block["template"].append("c_%s_in_shift_l" % input_name)

    if (node["merge_1x1"]):
        block["template"].append("c_%s_in_shift_h_1x1" % input_name)
        block["template"].append("c_%s_in_shift_l_1x1" % input_name)

    if (node["add"]):
        block["template"].append("c_%s_add_shift_l" % add_name)

    block["defines"] = {}
    block["defines"]["t_%s_acc" % output_name] = ["type", "int32_t"]
    block["defines"]["t_%s_acc_struct" % output_name] = [
        "struct",
        [["data", "t_%s_acc" % output_name], ["last", "bool"]]
    ]

    if (node["merge_1x1"]):
        block["defines"]["t_%s_acc" % output_1x1_name] = ["type", "int32_t"]
        block["defines"]["t_%s_acc_struct" % output_1x1_name] = [
            "struct",
            [["data", "t_%s_acc" % output_1x1_name], ["last", "bool"]]
        ]

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
    block["defines"]["c_%s_index" % name]          = ["const", node["kernel"]]
    block["defines"]["c_%s_reuse" % name]          = ["const", node["reuse"]]

    if node["add"]:
        diff_scale = node["actscale"][1] - node["wscale"][0] - node["actscale"][0]
        block["defines"]["c_%s_add_shift_l" % add_name] = ["const", diff_scale]

    block["args"] = []

    if (node["reuse"] > 1):
        block["args"].append("s_%s_nchw" % input_name)
    else:
        if node["is_1x1"]:
            block["args"].append("s_%s" % input_name)
        else:
            block["args"].append("s_%s_compute" % input_name)

    block["args"].append("s_%s" % weight_name)
    if (has_bias):
        block["args"].append("s_%s" % bias_name)

    if (node["merge_1x1"]):
        block["args"].append("s_%s" % weight_1x1_name)
        block["args"].append("s_%s" % bias_1x1_name)

    if (node["add"]):
        if (node["reuse"] > 1):
            block["args"].append("s_%s_nchw[0]" % add_name)
        else:
            block["args"].append("s_%s[0]" % add_name)

    if (node["has_forward"]):
        if (node["reuse"] > 1):
            block["args"].append("s_%s_p_nchw[0]" % forward_name)
        else:
            block["args"].append("s_%s[0]" % forward_name)

    block["args"].append("s_%s_acc" % output_name)

    if (node["merge_1x1"]):
        block["args"].append("s_%s_acc" % output_1x1_name)

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s_acc" % output_name
    declare["type"] = "t_%s_acc_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ops"]
    block["declare"].append(declare)

    if (node["merge_1x1"]):
        declare = {}
        declare["name"] = "s_%s_acc" % output_1x1_name
        declare["type"] = "t_%s_acc_struct" % output_1x1_name
        declare["is_array"] = True
        declare["dim"] = node["ops"]
        block["declare"].append(declare)

    if (node["has_forward"]):
        declare = {}
        if (node["reuse"] > 1):
            declare["name"] = "s_%s_p_nchw" % forward_name
        else:
            declare["name"] = "s_%s" % forward_name
        declare["type"] = "t_%s_struct" % input_name
        declare["is_array"] = True
        declare["dim"] = 1
        block["declare"].append(declare)

    block["pragma"] = []

    # depth = int(node["och"]/node["ops"] + 1)
    depth = 3
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s_acc" % (output_name)],
        ["depth", depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    if (node["merge_1x1"]):
        pragma = {}
        pragma["name"] = "stream"
        options = [
            ["variable", "s_%s_acc" % (output_1x1_name)],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    if (node["has_forward"]):
        # First two lines
        depth = (node["fh"]-1)*node["iw"]*node["ich"]
        # first two pixels of the third line
        depth += (node["fw"]-1)*node["ich"]
        depth += node["och"]+1
        pragma = {}
        pragma["name"] = "stream"
        if (node["reuse"] > 1):
            pragma_name = "s_%s_p_nchw" % forward_name
        else:
            pragma_name = "s_%s" % forward_name
        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)


    return [block]

def parse_nchw(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    arrange_name = []
    arrange_name.append(
        [
            input_name, 
            input_type_name, 
            node["fh"]*node["fw"],
            False, # is add
            False # is 1x1
        ]
    )

    if node["add"]:
        add_name = node["input"][3]
        add_type_name = add_name.replace("_skip", "")

        arrange_name.append(
            [
                add_name, 
                add_type_name, 
                1,
                True,
                False
            ]
        )

    blocks = []

    for tensor_name, tensor_type_name, dim, is_add, is_1x1 in arrange_name:
        block = {}
        block["func"] = "ArrangeOp"

        # Template parameters
        block["template"] = []
        block["template"].append("t_%s_struct" % tensor_type_name)
        block["template"].append("c_%s_ich" % name)
        block["template"].append("c_%s_och" % name)
        block["template"].append("c_%s_ow" % name)
        block["template"].append("c_%s_oh" % name)

        if is_add or is_1x1:
            block["template"].append("1")
        else:
            block["template"].append("c_%s_index" % name)
        block["template"].append("c_%s_stride" % name)
        block["template"].append("c_%s_ops" % name)
        block["template"].append("c_%s_reuse" % name)

        block["args"] = []

        if node["is_1x1"] or is_add or is_1x1:
            block["args"].append("s_%s" % tensor_name)
        else:
            block["args"].append("s_%s_compute" % tensor_name)

        block["args"].append("s_%s_nchw" % tensor_name)

        block["declare"] = []
        declare = {}
        declare["name"] = "s_%s_nchw" % tensor_name
        declare["type"] = "t_%s_struct" % tensor_type_name
        declare["is_array"] = True
        declare["dim"] = dim
        block["declare"].append(declare)

        block["pragma"] = []
        pragma = {}
        pragma["name"] = "stream"
        options = [
            ["variable", "s_%s_nchw" % tensor_name],
            ["depth", "2"],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
        blocks.append(block)

    return blocks

def parse_nhwc(name, node):

    blocks = []
    arrange_name = []

    for i, output_name in enumerate(node["output"]):
        output_type_name = output_name.replace("_skip", "")

        # For data forward there is one data coming out for every c_ich 
        # operations
        if "skip" in output_name:
            ops = 1
        else:
            ops = None

        if (i > 0) and (node["merge_1x1"]):
            depth = 2
        else:
            depth = node["ow"]*node["och"]*(node["fh"]-2)

        if depth < 0:
            depth = 2

        arrange_name.append([output_name, output_type_name, ops, depth])

    for tensor_name, tensor_type_name, ops, depth in arrange_name:
        block = {}
        block["func"] = "RearrangeOp"

        # Template parameters
        block["template"] = []
        block["template"].append("t_%s_struct" % tensor_type_name)
        block["template"].append("c_%s_ich" % name)
        block["template"].append("c_%s_och" % name)
        block["template"].append("c_%s_ow" % name)
        block["template"].append("c_%s_oh" % name)
        block["template"].append("c_%s_index" % name)
        block["template"].append("c_%s_stride" % name)
        if ops is not None:
            block["template"].append("%d" % ops)
        else:
            block["template"].append("c_%s_ops" % name)
        block["template"].append("c_%s_reuse" % name)

        block["args"] = []

        block["args"].append("s_%s_p_nchw[0]" % tensor_name)
        block["args"].append("s_%s[0]" % tensor_name)

        block["output"] = []
        block["output"].append("s_%s" % tensor_name)

        block["declare"] = []
        declare = {}
        declare["name"] = "s_%s" % tensor_name
        declare["type"] = "t_%s_struct" % tensor_type_name
        declare["is_array"] = True
        declare["dim"] = 1
        block["declare"].append(declare)

        block["pragma"] = []
        pragma = {}
        pragma["name"] = "stream"
        options = [
            ["variable", "s_%s" % tensor_name],
            ["depth", "%0d" % depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
        blocks.append(block)

    return blocks

def parse_split(name, node):

    blocks = []

    # If the reuse of the weights is greater than 1 then we need to
    # bufferize the input data to reorder from nhwc to nchw
    if (node["reuse"] > 1):
        blocks = blocks + parse_nchw(name, node)

    blocks = blocks + parse_comp(name, node)
    blocks = blocks + parse_wout(name, node)

    # If the reuse of the weights is greater than 1 then we need to
    # restore the output data from nchw to nhwc
    if (node["reuse"] > 1):
        blocks = blocks + parse_nhwc(name, node)

    return blocks

def parse(name, node, wrapper=False):

    return parse_split(name, node)
