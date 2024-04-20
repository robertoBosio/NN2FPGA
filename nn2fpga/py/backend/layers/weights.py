import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
from tabulate import tabulate
import math
import json
import backend.graph as graph
import backend.utils as utils
from backend.layers.uram_download import fill_uram_layer, add_uram_layer
import backend.layers.uram_download as uram_download
from backend.layers.quant import get_quant_type

WIDTH_STREAM = 8

def num_of_words(bits, n_weights, word = 8):
    """ Compute the number of words needed to store 1 parameter """
    # TODO: add support for weights packing in streaming
    if (word >= bits):
        return n_weights
    else:
        return int(n_weights * (bits // word))

def bias_ops_calc(
    ich_ops,
    ops,
    depth
):
    """ Compute the number of parallel biases needed"""
    parallelism = ops
    if depth:
        parallelism = ich_ops

    return parallelism

def mem_shape_calc(node, fh, fw, is_bias=False):
    """ Compute the memory shape needed to store the weights """
    ich = node["ich"]
    och = node["och"]
    ops = node["ops"]
    ich_ops = node["ich_ops"]
    depth = node["depth"]

    if (is_bias):
        parallelism = bias_ops_calc(ich_ops, ops, depth)
        return [och // parallelism, parallelism]

    ich_iter_ops = ich // ich_ops
    och_iter_ops = och // ops
    if depth:
        och_iter_ops = 1

    assert ich_iter_ops > 0
    assert och_iter_ops > 0

    return [fh * fw, ich_iter_ops * och_iter_ops, ich_ops * ops]
    
def compute_bram_layer(weight_bits, weight_number, parallelism, print_info=False):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    bram9 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=9)
    bram18 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=18)
    bram36 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=36)
    bram72 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=72)

    if (print_info):
        print(f"{weight_bits} bits, {weight_number} weights, {parallelism} parallelism, BRAM9: {bram9}, BRAM18: {bram18}, BRAM36: {bram36}, BRAM72: {bram72}")
    
    return min(bram9, bram18, bram36, bram72)

def bram_consumption(weight_bits, weight_number, parallelism, WIDTH=36):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # 36 bits, in which 4 bits are reserved to ECC code
    SIZE_BRAM18 = (18 * 1024)
    
    # Useful space in BRAM36, composed by two BRAM18.
    SIZE_BRAM36 = SIZE_BRAM18 * 2

    WIDTH_BRAM36 = WIDTH

    # Assuming is implemented using LUTRAM
    if (weight_number * weight_bits) <= SIZE_BRAM36:
        return 0
    
    very_long_word = parallelism * weight_bits
    mem_width = very_long_word // WIDTH_BRAM36
    mem_width_rem = very_long_word % WIDTH_BRAM36
    word_depth = weight_number // parallelism
    mem_depth = int(math.ceil(word_depth / (SIZE_BRAM36 // WIDTH_BRAM36)))
    tot_bram = mem_width * mem_depth

    rem_bram = 0
    if (mem_width_rem > 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 72)))
    elif (mem_width_rem > 18 and mem_width_rem <= 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 36)))
    elif (mem_width_rem > 8 and mem_width_rem <= 18):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 18)))
    elif (mem_width_rem > 0 and mem_width_rem <= 8):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 9)))
    
    tot_bram += rem_bram
    
    return tot_bram

def parse_on_chip_weights(
    node_info,
    node_quant,
    pre_values,
    dynamic_init = False,
    is_bias = False
):

    # Transforming a filter of dimension [och][ich][ih][iw] into one of
    # dimension [iw * ih][(ich * och)/(och_ops * ich_ops)][och_ops * ich_ops] where ops is the number 2D convolutions
    # computed in parallel

    dich = node_info["ich"]
    dih  = node_info["fh"]
    diw  = node_info["fw"]
    doch = node_info["och"]
    dops = node_info["ops"]
    dich_ops = node_info["ich_ops"]

    if node_info["depth"]:
        doch = 1

    scale_factor = 2 ** node_quant["scale_factor"]
    signed = node_quant["signed"]
    bits = node_quant["bits"]
    narrow = node_quant["narrow"]

    narrow_h = 0
    if not signed and narrow:
      narrow_h = 1
    limit_h = 2**(bits-signed) - 1 - narrow_h

    narrow_l = 0
    if signed and narrow:
      narrow_l = 1
    limit_l = -1 * signed * 2**(bits-signed) + narrow_l

    doch_ops = int(doch / dops)
    dich_iter_ops = int(dich / dich_ops)

    assert dich_iter_ops > 0
    assert doch_ops > 0

    values = np.zeros(
        mem_shape_calc(node_info, node_info["fh"], node_info["fw"], is_bias)
    )

    print(f"ops: {dops}, ich_ops: {dich_ops}, ich: {dich}, och: {doch}, depth: {node_info['depth']}")
    print(f"shape: {values.shape}")
    # Reordering the weights based on the parallelism needed by the convolution
    for ich in range(dich_iter_ops):
        for och in range(doch_ops):
            off = och * dops
            for ich_ops in range(dich_ops):
                for ih in range(dih - 1, -1, -1):
                    for iw in range(diw - 1, -1, -1):
                        off_ich = ich * dich_ops + ich_ops
                        for ops in range(dops):
                            quant_value = pre_values[off + ops][off_ich][ih][iw]
                            quant_value = np.round(quant_value / scale_factor)
                            
                            if (limit_h < quant_value):
                                quant_value = limit_h

                            if (limit_l > quant_value):
                                quant_value = limit_l
                            
                            if not dynamic_init:
                                quant_value = quant_value * scale_factor

                            index = ih * diw + iw
                            ch = ich * doch_ops + och
                            ops_index = ich_ops * dops + ops
                            values[dih * diw - 1 - index][ch][ops_index] = quant_value
    
    return values

def parse_on_chip_biases(
    node_info,
    node_quant,
    pre_values,
    dynamic_init = False,
    is_bias = False
):
    """ Parse the biases and reorganize the data """

    doch = node_info["och"]
    dops = node_info["ops"]
    dich_ops = node_info["ich_ops"]

    scale_factor = 2**node_quant["scale_factor"]
    bits = node_quant["bits"]
    signed = node_quant["signed"]
    narrow = node_quant["narrow"]

    narrow_h = 0
    if not signed and narrow:
      narrow_h = 1
    limit_h = 2**(bits-signed) - 1 - narrow_h

    narrow_l = 0
    if signed and narrow:
      narrow_l = 1
    limit_l = -1 * signed * 2**(bits-signed) + narrow_l

    parallelism = bias_ops_calc(dich_ops, dops, node_info["depth"])

    values = np.zeros(mem_shape_calc(node_info, node_info["ih"], node_info["iw"], is_bias))
    pre_values = np.squeeze(pre_values)
    
    # Reordering the weights based on the parallelism needed by the convolution
    for s_och in range(doch // parallelism):
        for s_ops in range(parallelism):
            quant_value = pre_values[s_och * parallelism + s_ops]
            quant_value = np.round(quant_value / scale_factor)
            
            if (limit_h < quant_value):
                quant_value = limit_h

            if (limit_l > quant_value):
                quant_value = limit_l
            
            if not dynamic_init:
                quant_value = quant_value * scale_factor
            
            values[s_och][s_ops] = quant_value
    
    return values

def pack_weights(
    values,
    bits=8,
    is_bias=False
):
    """ Pack the weights in a single array """

    if (not is_bias):
        values = np.swapaxes(values, 0, 1)
    values = values.flatten()

    if bits >= 8:
        bytes_num = int(bits / 8)
        new_values = np.zeros([values.shape[0] * bytes_num])

        for i in range(values.shape[0]):
            data = values[i]
            data_bytes = np.zeros([bytes_num])
            for j in range(bytes_num):
                data_byte = int(data) & 0xff
                data_bytes[j] = data_byte
                data = int(data // 256)

            # Changing MSB to LSB order to ease hw reconstruction
            # of the original value
            for j in range(bytes_num - 1, -1, -1):
                new_values[i * bytes_num + j] = data_bytes[bytes_num - 1 - j]
            # for j in range(0, bytes_num):
            #     new_values[i*bytes_num+j] = data_bytes[j]

        values = new_values
    # elif bits < 8:
    #     pack_width = int(8/bits)
    #     new_values = np.zeros([int(values.shape[0]/pack_width)])
    #     mask = 2**bits - 1

    #     for i in range(0, values.shape[0], pack_width):
    #         data_byte = 0
    #         for j in range(pack_width):
    #             data = values[i+j]
    #             data_byte |= (int(data) & mask) << (j*bits)

    #         new_values[int(i/pack_width)] = data_byte

    #     values = new_values
    
    return values

def parse_off_chip(
    node_info,
    pre_values,
    bits=8,
    signed=1,
    narrow=1
):

    dich = node_info["ich"]
    dih  = node_info["ih"]
    diw  = node_info["iw"]
    doch = node_info["och"]
    dops = node_info["ops"]
    scale_factor = 2**node_info["scale_factor"]

    limit_h = 2**(bits-signed)-1
    limit_l = -1*signed*2**(bits-signed)

    values = np.zeros(
        [dih*diw*dich*doch]
    )

    addr = 0
    for ich in range(dich):
        for och in range(int(doch/dops)):
            for ih in range(dih-1, -1, -1):
                for iw in range(diw-1, -1, -1):
                    off = och*dops
                    for ops in range(dops):
                        quant_value  = pre_values[off+ops][ich][ih][iw]
                        quant_value  = np.round(quant_value/scale_factor)
                        if (limit_h < quant_value):
                            quant_value = limit_h

                        if (limit_l > quant_value):
                            quant_value = limit_l

                        values[addr] = quant_value
                        addr         = addr + 1
    
    return values

def extract_info(
    new_node,
    weight_name,
    node_info,
    init_info,
    off_chip_memory,
    dynamic_init,
):
    pre_values = numpy_helper.to_array(init_info[weight_name]) 
    shape = pre_values.shape

    och = node_info["och"]
    ich = node_info["ich"]

    if (len(shape) > 2):
        ih = shape[2]
        iw = shape[3]
    else:
        ih = 1
        iw = 1
        pre_values = np.expand_dims(pre_values, axis=-1)
        pre_values = np.expand_dims(pre_values, axis=-1)

    oh = node_info["oh"]
    ow = node_info["ow"]
    stride = node_info["stride"]
    pad = node_info["pad"]
    ops = node_info["ops"]
    ich_ops = node_info["ich_ops"]

    if len(shape) > 1:
        is_bias = False
    else:
        pre_values = np.expand_dims(pre_values, axis=-1)
        is_bias = True

    if is_bias:
        if node_info["depth"]:
            ops = node_info["ich_ops"]
    else:
        if node_info["depth"]:
            och = 1

    signed = new_node["signed"]
    scale_factor = new_node["scale_factor"]
    if is_bias and new_node["bits"] > 16:
        new_node["bits"] = 16
    bits   = new_node["bits"]
    narrow = new_node["narrow"]
    bw = int(128/bits)

    if is_bias:
        data_type = get_quant_type(signed, bits, scale_factor, narrow=False)
    else:
        data_type = get_quant_type(signed, bits, scale_factor, narrow=True)

    new_node["data_type"] = data_type

    new_node["ich"]    = ich
    new_node["ih"]     = ih
    new_node["iw"]     = iw
    new_node["och"]    = och
    new_node["oh"]     = oh
    new_node["ow"]     = ow
    new_node["stride"] = stride
    new_node["pad"]    = pad
    # FIX: adding this check to avoid problems in merged pipelines
    # with same inputs but different output channels
    # Check if ops are more than och and clip
    if ops > och:
        new_node["ops"] = och
    else:
        new_node["ops"] = ops
    new_node["ich_ops"] = ich_ops
    new_node["is_bias"] = is_bias
    new_node["type"]    = 'const'
    new_node["kernel"]  = ih*iw
    new_node["total"]   = ich*och*ih*iw*oh*ow/stride
    if (not is_bias):
        new_node["n_weights"] = ich*och*ih*iw
    else:
        new_node["n_weights"] = och
    new_node["dynamic_init"] = dynamic_init
    new_node["uram_storage"] = False
    new_node["img_ch"]  = ich*och
    new_node["reuse"] = 1
    new_node["pre_values"] = pre_values
    # added to cope with och and ich exchanged in onnx representation
    new_node["depth"] = node_info["depth"]

    dim = ich*och*ih*iw

    new_node["off_chip_memory"] = off_chip_memory and (dim > 4096)

    return new_node

def weights_info(
    model,
    io_dict,
    init_info,
    off_chip_memory=False,
    dynamic_init=False,
):
    new_nodes = {}
    rem_nodes = []

    io_connect = graph.extract_connections(model, io_dict)

    for node_name, node_info in io_dict.items():
        if 'quant' in node_name.lower():
            all_constant = all(
                [
                    input in init_info.keys()
                    for input in node_info["input"]
                ]
            )
            # Must be checked that all quant layers are related to parameters
            # and that all activations quant layer have been absorbed
            assert all_constant

            conv_name = io_connect[node_info["output"][0]][1][0]

            assert 'conv' in io_dict[conv_name]["type"].lower()

            conv_info = io_dict[conv_name]

            tensor_name = node_info["input"][0]
            new_node_name = "Weight%s" % tensor_name
            new_nodes[new_node_name] = {}
            new_nodes[new_node_name]["input"] = [new_node_name]
            new_nodes[new_node_name]["output"] = [node_info["output"][0]]
            new_nodes[new_node_name]["is_constant"] = True
            new_nodes[new_node_name]["scale_factor"] = node_info["scale_factor"]
            new_nodes[new_node_name]["signed"] = node_info["signed"]
            new_nodes[new_node_name]["bits"] = node_info["bits"]
            new_nodes[new_node_name]["narrow"] = node_info["narrow"]

            new_nodes[new_node_name] = extract_info(
                new_nodes[new_node_name],
                tensor_name,
                conv_info,
                init_info,
                off_chip_memory,
                dynamic_init,
            )

            rem_nodes.append(node_name)
            
    for node_name, node_info in new_nodes.items():
        io_dict[node_name] = node_info

    for node_name in rem_nodes:
        del io_dict[node_name]

    return io_dict

def parse_main(io_dict):
    
    block = {}
    block["func"] = "memory_management"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["output"] = []

    block["declare"] = []
    block["defines"] = {}
    block["pragma"] = []

    # for name, node in io_dict.items():
    #     if 'const' == node["type"]:
    #         output_name = node["output"][0]
    #         if node["off_chip_memory"]:
    #             block["input"].append("%s" % output_name)
    #             block["args"].append("i_data_%s" % output_name)

    #             pragma = {}
    #             pragma["name"] = "interface"
    #             options = [
    #                 ["port", "i_data_%s" % output_name],
    #                 ["mode", "m_axi"],
    #             ]
    #             pragma["options"] = options
    #             block["pragma"].append(pragma)

    output_name = "params"
    block["stream_input"].append({"name" : f"i_data_{output_name}", "type" : "t_params_axi_stream"})
    block["args"].append("i_data_%s" % output_name)

    pragma = {}
    pragma["name"] = "interface"
    options = [
        ["port", "i_data_%s" % output_name],
        ["mode", "axis"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    block["is_const"] = True

    def declare_stream(name, dim):
        """ Declare a stream with all the needed pragmas """
        arg = f"s_{name}"

        tmp = {}
        tmp["name"] = f"s_{name}"
        tmp["type"] = f"t_{name}"
        tmp["dim"]  = dim
        tmp["is_array"] = True
        declare = tmp

        pragma = {}
        pragma["name"] = "stream"
        options = [
            ["variable", f"s_{name}"],
            ["depth", "2"],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        pragma0 = pragma

        pragma = {}
        pragma["name"] = "array_partition"
        options = [
            ["variable", f"s_{name}"],
            ["type", "complete"],
        ]
        pragma["options"] = options
        pragma1 = pragma
        return arg, declare, pragma0, pragma1

    for name, node in io_dict.items():
        if 'conv' == node["type"]:
            weight_name = node["input"][1]
            args, declare, pragma0, pragma1 = declare_stream(weight_name, node["fw"] * node["fh"])
            block["args"].append(args)
            block["declare"].append(declare)
            block["pragma"].append(pragma0)
            block["pragma"].append(pragma1)
            
            if (node["has_bias"]):
                bias_name = node["input"][2]
                args, declare, pragma0, _ = declare_stream(bias_name, 1)
                block["args"].append(args)
                block["declare"].append(declare)
                block["pragma"].append(pragma0)

            if (node["merge_1x1"]):
                weight_name_1x1 = node["input"][3]
                args, declare, pragma0, _ = declare_stream(weight_name_1x1, 1)
                block["args"].append(args)
                block["declare"].append(declare)
                block["pragma"].append(pragma0)
                
                if (node["has_bias"]):
                    bias_name_1x1 = node["input"][4]
                    args, declare, pragma0, _ = declare_stream(bias_name_1x1, 1)
                    block["args"].append(args)
                    block["declare"].append(declare)
                    block["pragma"].append(pragma0)

    return block

def off_chip_ddr(
    name,
    node,
    input_name,
    output_name
):
    blocks = []
    block = {}
    block["func"] = "mem_algo"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["output"] = []
    block["bits"] = node["bits"]
    # TODO: Fix off-chip flow
        # new_node["bw"]    = bw
        # new_node["values"] = parse_off_chip(
        #     new_node,
        #     pre_values,
        #     bits,
        #     signed,
        #     narrow
        # )

    block["template"] = []
    block["template"].append("t_%s_st" % (output_name))
    block["template"].append("t_%s" % (output_name))
    block["template"].append("c_%s_ich" % (name))
    block["template"].append("c_%s_och" % (name))
    block["template"].append("c_%s_ow" % (name))
    block["template"].append("c_%s_oh" % (name))
    block["template"].append("c_%s_iw" % (output_name))
    block["template"].append("c_%s_ih" % (output_name))
    block["template"].append("c_%s_ops" % (output_name))
    block["template"].append("c_%s_rw" % (output_name))
    block["template"].append("c_%s_bw" % (output_name))
    block["template"].append("c_%s_reuse" % (output_name))
    block["template"].append("0")

    block["input"].append("%s" % output_name)
    block["args"].append("s_o_%s" % output_name)
    block["args"].append("i_data_%s" % output_name)

    block["defines"] = {}
    block["defines"]["t_%s_st" % (output_name)]    = ["type", node["data_type"]]
    output_type_name = "hls::vector<%s, %0d>" % (node["data_type"], node["ops"])
    block["defines"]["t_%s" % (output_name)]       = ["type",  output_type_name]
    block["defines"]["c_%s_ich" % (name)]          = ["const", node["ich"]]
    block["defines"]["c_%s_och" % (name)]          = ["const", node["och"]]
    block["defines"]["c_%s_ow" % (name)]           = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % (name)]           = ["const", node["oh"]]
    block["defines"]["c_%s_iw" % (output_name)]    = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % (output_name)]    = ["const", node["ih"]]
    block["defines"]["c_%s_ops" % (output_name)]   = ["const", node["ops"]]
    block["defines"]["c_%s_bw" % (output_name)]    = ["const", node["bw"]]
    block["defines"]["c_%s_reuse" % (output_name)] = ["const", node["reuse"]]
    block["defines"]["c_%s_rw" % (output_name)] = ["const", node["bits"]]

    # Declare only in tb wrapper
    block["tb_declare"] = []
    tmp = {}
    tmp["name"] = "c_%s" % output_name
    tmp["type"] = "t_%s" % output_name
    tmp["is_array"] = True
    tmp["init"] = node["values"]

    block["tb_declare"].append(tmp)


    block["declare"] = []

    declare = {}
    declare["name"] = "s_o_%s" % output_name
    declare["type"] = "t_%s" % output_name
    declare["is_array"] = True
    declare["dim"] = node["bw"]
    block["declare"].append(declare)

    block["pragma"] = []
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_o_%s" % (output_name)],
        ["depth", 2],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    block["size"] = node["iw"]*node["ih"]
    block["is_const"] = True
    blocks.append(block)

    block = {}

    block["func"] = "produce_stream"
    block["args"] = []
    block["input"] = []
    block["output"] = []

    block["template"] = []
    block["template"].append("t_%s_st " % (output_name))
    block["template"].append("t_%s" % (output_name))
    block["template"].append("c_%s_ich" % (name))
    block["template"].append("c_%s_och" % (name))
    block["template"].append("c_%s_ow" % (name))
    block["template"].append("c_%s_oh" % (name))
    block["template"].append("c_%s_iw" % (output_name))
    block["template"].append("c_%s_ih" % (output_name))
    block["template"].append("c_%s_ops" % (output_name))
    block["template"].append("c_%s_bw" % (output_name))
    block["template"].append("c_%s_reuse" % (output_name))
    block["template"].append("c_%s_rw" % (output_name))

    block["output"].append("%s" % output_name)
    block["args"].append("s_o_%s" % output_name)
    block["args"].append("s_%s" % output_name)

    block["defines"] = {}

    block["declare"] = []

    block["pragma"] = []
    block["size"] = node["iw"]*node["ih"]
    block["is_const"] = True
    blocks.append(block)

    return blocks

def generate_axitostandard_stream(tot_params):
    """ Declare the first function that converts from axi to standard stream """
    block = {}
    block["func"] = "axi_to_stream"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["uram_input"] = []
    block["output"] = []
    block["declare"] = []
    block["pragma"] = []
    block["template"] = ["t_params_axi_stream", "t_params_stream",
                            {"name": tot_params, "comment": "tot cycles"}]
    block["args"] = ["i_data_params", "s_axi_to_stream_init_flag", "s_axi_to_stream_out"]
    
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", f"s_axi_to_stream_out"],
        ["depth", "2"],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    
    tmp = {}
    tmp["name"] = f"s_axi_to_stream_init_flag"
    tmp["type"] = "static bool"
    tmp["is_array"] = False
    tmp["is_const"] = False
    tmp["size"] = 1
    tmp["init_value"] = "false"
    block["declare"].append(tmp)
    
    tmp = {}
    tmp["name"] = f"s_axi_to_stream_out"
    tmp["type"] = f"t_params_stream"
    tmp["is_array"] = True
    tmp["dim"] = 1
    block["declare"].append(tmp)
    return block

def on_chip_rom(
    param_node,
    width_stream
):

    blocks = []
    block = {}

    conv_name = param_node["name"]
    conv_node = param_node["conv_node"]
    weight_node = param_node["weight_node"]
    weight_output_name = weight_node["output"][0]
    
    bias_node = param_node["bias_node"]
    if (bias_node is not None):
        bias_output_name = bias_node["output"][0]
    else:
        bias_output_name = None

    weight_node_1x1 = param_node["weight_node_1x1"]
    if (weight_node_1x1 is not None):
        weight_output_name_1x1 = weight_node_1x1["output"][0]
    else:
        weight_output_name_1x1 = None

    bias_node_1x1 = param_node["bias_node_1x1"]
    if (bias_node_1x1 is not None):
        bias_output_name_1x1 = bias_node_1x1["output"][0]
    else:
        bias_output_name_1x1 = None
    
    bias_ops = bias_ops_calc(
        conv_node["ich_ops"],
        conv_node["ops"],
        conv_node["depth"]
    )
    
    # Declare the function to pass from axi stream data type to internal type for params
    if (param_node["first"]):
        block.append(generate_axitostandard_stream(param_node["tot_params"]))

    block["func"] = "produce_shift_stream"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["uram_input"] = []
    block["output"] = []
    # block["bits"] = node["bits"]
    # block["index"] = node["ih"] * node["iw"]
    # block["och"] = node["och"]
    
    # FIX: adding this check to avoid problems in merged pipelines
    # with same inputs but different output channels
    # Check if ops are mode than och and clip
    # if node["ops"] > node["och"]:
    #     block["ops"] = node["och"]
    # else:
    #     block["ops"] = node["ops"]
    
    # block["ich_ops"] = node["ich_ops"]
    # block["dynamic_init"] = dynamic_init
    # block["uram_storage"] = uram_storage

    ### Template declaration of the produce_shift_stream function
    block["template"] = []
    if (bias_node is not None):
        block["template"].append(f"t_{bias_output_name}_mem")
    else:
        block["template"].append("std::nullptr_t")
    
    block["template"].append(f"t_{weight_output_name}_mem")
    
    if (bias_node_1x1 is not None):
        block["template"].append(f"t_{bias_output_name_1x1}_mem")
    else:
        block["template"].append("std::nullptr_t")
    
    if (weight_node_1x1 is not None):
        block["template"].append(f"t_{weight_output_name_1x1}_mem")
    else:
        block["template"].append("std::nullptr_t")
    
    block["template"].append(f"t_params_stream")
    
    if (bias_node is not None):
        block["template"].append(f"t_{bias_output_name}")
    else:
        block["template"].append("std::nullptr_t")
    
    block["template"].append(f"t_{weight_output_name}")
    
    if (bias_node_1x1 is not None):
        block["template"].append(f"t_{bias_output_name_1x1}")
    else:
        block["template"].append("std::nullptr_t")
    
    if (weight_node_1x1 is not None):
        block["template"].append(f"t_{weight_output_name_1x1}")
    else:
        block["template"].append("std::nullptr_t")

    # Not a beatiful solution, we cannot modify och since it is used also for the bias
    # size. Must correct this when introducing groups.
    if (conv_node["depth"]):
        block["template"].append({"name" : f"{weight_node['och']}", "comment" : f"ich divided by groups"})
    else:
        block["template"].append(f"c_{conv_name}_ich")

    block["template"].append(f"c_{conv_name}_och")
    block["template"].append(f"c_{conv_name}_ow")
    block["template"].append(f"c_{conv_name}_oh")
    block["template"].append(f"c_{conv_name}_fw")
    block["template"].append(f"c_{conv_name}_fh")
    block["template"].append(f"c_{conv_name}_ops")
    block["template"].append(f"c_{conv_name}_ich_ops")
    block["template"].append({"name" : bias_ops, "comment" : "bias_ops"})
    block["template"].append(f"c_{conv_name}_reuse")
    block["template"].append({"name" : param_node["shift_cycles"], "comment" : "shift cycles"})
    block["template"].append({"name" : width_stream, "comment" : "width stream"})
    if (bias_node is not None):
        block["template"].append({"name" : int(np.ceil(bias_node["bits"]/width_stream)), "comment" : "bias reads per data"})
    else:
        block["template"].append({"name" : 0, "comment" : "bias reads per data"})
    block["template"].append({"name" : int(np.ceil(weight_node["bits"]/width_stream)), "comment" : "weight reads per data"})
    ### Finish template declaration

    ### Arguments of the produce_shift_stream function
    if (bias_node is not None):
        block["args"].append(f"c_{bias_output_name}")
    else:
        block["args"].append(f"nullptr")

    block["args"].append(f"c_{weight_output_name}")
    
    if (bias_node_1x1 is not None):
        block["args"].append(f"c_{bias_output_name_1x1}")
    else:
        block["args"].append("nullptr")
    
    if (weight_node_1x1 is not None):
        block["args"].append(f"c_{weight_output_name_1x1}")
    else:
        block["args"].append("nullptr")
    
    block["args"].append(f"s_{param_node['connections']['in']}_out")
    block["args"].append(f"s_{conv_name}_init_flag")
    
    if (bias_node is not None):
        block["args"].append(f"s_{bias_output_name}")
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")
    
    block["args"].append(f"s_{weight_output_name}")

    if (bias_node_1x1 is not None):
        block["args"].append(f"s_{bias_output_name_1x1}")
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")
    
    if (weight_node_1x1 is not None):
        block["args"].append(f"s_{weight_output_name_1x1}")
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")
    
    if (not param_node["last"]):
        block["args"].append(f"s_{conv_name}_out")
    else:
        block["args"].append("(hls::stream<t_params_stream>*)(nullptr)")
    ### Finish arguments

    ### Variable declaration
    block["declare"] = []
    tmp = {}
    tmp["name"] = f"static c_{weight_output_name}"
    tmp["type"] = f"t_{weight_output_name}_mem"
    tmp["is_array"] = True
    tmp["is_const"] = False
    # size = weight_node["values"].shape
    tmp["size"] = mem_shape_calc(weight_node, weight_node["ih"], weight_node["iw"], weight_node["is_bias"])
    # tmp["init"] = weight_node["values"]
    tmp["form"] = "float"
    block["declare"].append(tmp)
    
    if (bias_node is not None):
        tmp = {}
        tmp["name"] = f"static c_{bias_output_name}"
        tmp["type"] = f"t_{bias_output_name}_mem"
        tmp["is_array"] = True
        tmp["is_const"] = False
        # size = bias_node["values"].shape
        tmp["size"] = mem_shape_calc(bias_node, bias_node["ih"], bias_node["iw"], bias_node["is_bias"])
        # tmp["init"] = bias_node["values"]
        tmp["form"] = "float"
        block["declare"].append(tmp)
    
    if (weight_node_1x1 is not None):
        tmp = {}
        tmp["name"] = f"static c_{weight_output_name_1x1}"
        tmp["type"] = f"t_{weight_output_name_1x1}_mem"
        tmp["is_array"] = True
        tmp["is_const"] = False
        # size = weight_node_1x1["values"].shape
        tmp["size"] = mem_shape_calc(weight_node_1x1, weight_node_1x1["ih"], weight_node_1x1["iw"], weight_node_1x1["is_bias"])
        # tmp["init"] = weight_node_1x1["values"]
        tmp["form"] = "float"
        block["declare"].append(tmp)
    
    if (bias_node_1x1 is not None):
        tmp = {}
        tmp["name"] = f"static c_{bias_output_name_1x1}"
        tmp["type"] = f"t_{bias_output_name_1x1}_mem"
        tmp["is_array"] = True
        tmp["is_const"] = False
        # size = bias_node_1x1["values"].shape
        tmp["size"] = mem_shape_calc(bias_node_1x1, bias_node_1x1["ih"], bias_node_1x1["iw"], bias_node_1x1["is_bias"])
        # tmp["init"] = bias_node_1x1["values"]
        tmp["form"] = "float"
        block["declare"].append(tmp)

    tmp = {}
    tmp["name"] = f"s_{conv_name}_init_flag"
    tmp["type"] = "static bool"
    tmp["is_array"] = False
    tmp["is_const"] = False
    tmp["size"] = 1
    tmp["init_value"] = "false"
    block["declare"].append(tmp)

    if (not param_node["last"]):
        tmp = {}
        tmp["name"] = f"s_{conv_name}_out"
        tmp["type"] = f"t_params_stream"
        tmp["is_array"] = True
        tmp["dim"] = 1
        block["declare"].append(tmp)
    ### Finish variable declaration

    block["pragma"] = []
    
    # Binding memory to URAM storage
    pragma = {}
    if weight_node["uram_storage"]:
        pragma["name"] = "bind_storage"
        options = [
            ["variable", f"c_{weight_output_name}"],
            ["impl", "uram"],
            ["type", "ram_s2p"]
        ]
        pragma["options"] = options
    else:
        pragma["name"] = "bind_storage"
        options = [
            ["variable", f"c_{weight_output_name}"],
            ["type", "ram_s2p"]
        ]
        pragma["options"] = options

    block["pragma"].append(pragma)
    
    if (weight_node_1x1 is not None):
        pragma = {}
        if (weight_node_1x1["uram_storage"]):
            pragma["name"] = "bind_storage"
            options = [
                ["variable", f"c_{weight_output_name_1x1}"],
                ["impl", "uram"],
                ["type", "ram_s2p"]
            ]
            pragma["options"] = options
        else:
            pragma["name"] = "bind_storage"
            options = [
                ["variable", f"c_{weight_output_name_1x1}"],
                ["type", "ram_s2p"]
            ]
            pragma["options"] = options

        block["pragma"].append(pragma)

    # Completely reshaping weights and bias memory
    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", f"c_{weight_output_name}"],
        ["dim", 3],
        ["type", "complete"]
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    
    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", f"c_{weight_output_name}"],
        ["dim", 1],
        ["type", "complete"]
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    
    if (bias_node is not None):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", f"c_{bias_output_name}"],
            ["dim", 2],
            ["type", "complete"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
    
    if (weight_node_1x1 is not None):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", f"c_{weight_output_name_1x1}"],
            ["dim", 3],
            ["type", "complete"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
    
    if (bias_node_1x1 is not None):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", f"c_{bias_output_name_1x1}"],
            ["dim", 2],
            ["type", "complete"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    # Declaring streams connecting produce_shift_stream
    if (not param_node["last"]):
        pragma = {}
        pragma["name"] = "stream"
        options = [
            ["variable", f"s_{conv_name}_out"],
            ["depth", "2"],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
    
    ### Memory management input and outputs 
    block["output"].append({"name" : f"s_{weight_output_name}", "type" : f"t_{weight_output_name}", "size": conv_node["fh"] * conv_node["fw"]})
    if (bias_node is not None):
        block["output"].append({"name" : f"s_{bias_output_name}", "type": f"t_{bias_output_name}", "size": 1})
    if (weight_node_1x1 is not None):
        block["output"].append({"name" : f"s_{weight_output_name_1x1}", "type": f"t_{weight_output_name_1x1}", "size": 1})
    if (bias_node_1x1 is not None):
        block["output"].append({"name" : f"s_{bias_output_name_1x1}", "type": f"t_{bias_output_name_1x1}", "size": 1})
    block["is_const"] = True
    if (param_node["first"]):
        block["stream_input"].append({"name" : "&i_data_params", "type" : "t_params_axi_stream"})
    blocks.append(block)
    ### Finish memory management outputs

    return blocks

def parse_const(node, name):
    """ Parse the constants of the parameters """

    has_bias = node["has_bias"]
    blocks = []
    block = {}
    conv_node = node
    weight_node = node["weight_quant"]
    weight_output_name = f"{name}_weight"
    bias_node = None
    weight_node_1x1 = None
    bias_node_1x1 = None

    block["bytes_param"] = num_of_words(weight_node["bits"], np.prod(weight_node["values"].shape), WIDTH_STREAM)
    
    if (has_bias):
        bias_node = node["bias_quant"]
        bias_ops = bias_ops_calc(
            conv_node["ich_ops"],
            conv_node["ops"],
            conv_node["depth"]
        )
        block["bytes_param"] += num_of_words(bias_node["bits"], np.prod(bias_node["values"].shape), WIDTH_STREAM)
        bias_output_name = f"{name}_bias"

    if (node["merge_1x1"]):
        weight_node_1x1 = node["merge_node"]["weight_quant"]
        block["bytes_param"] += num_of_words(weight_node_1x1["bits"], np.prod(weight_node_1x1["values"].shape), WIDTH_STREAM)
        weight_output_name_1x1 = f"{name}_merge_weight"
        if (node["merge_node"]["has_bias"]):
            bias_node_1x1 = node["merge_node"]["bias_quant"]
            block["bytes_param"] += num_of_words(bias_node_1x1["bits"], np.prod(bias_node_1x1["values"].shape), WIDTH_STREAM)
            bias_output_name_1x1 = f"{name}_merge_bias"
    

    # Recovering parameters values
    pre_values_weights = weight_node["values"]
    if conv_node["depth"]:
        pre_values_weights = np.swapaxes(pre_values_weights, 0, 1)

    weight_node["values"] = parse_on_chip_weights(
        node,
        weight_node,
        pre_values_weights,
        True,
        False
    )

    if (bias_node is not None):
        pre_values_biases = bias_node["values"]
        bias_node["values"] = parse_on_chip_biases(
            node,
            bias_node,
            pre_values_biases,
            True,
            True
        )
    
    if (weight_node_1x1 is not None):
        pre_values_weights_1x1 = weight_node_1x1["values"]
        if conv_node["depth"]:
            pre_values_weights_1x1 = np.swapaxes(pre_values_weights_1x1, 0, 1)

        weight_node_1x1["values"] = parse_on_chip_weights(
            node["merge_node"],
            weight_node_1x1,
            pre_values_weights_1x1,
            True,
            False
        )

    if (bias_node_1x1 is not None):
        pre_values_biases_1x1 = bias_node_1x1["values"]
        bias_node_1x1["values"] = parse_on_chip_biases(
            node["merge_node"],
            bias_node_1x1,
            pre_values_biases_1x1,
            True,
            True
        )

    ### Type defintions
    block["defines"] = {}
    block["func"] = "fake_func_params"
    block["defines"][f"t_{weight_output_name}_mem"]    = ["type", weight_node["data_type"]]
    output_type_name = f"std::array<std::array<t_{weight_output_name}_mem, {conv_node['ops']}>, {conv_node['ich_ops']}>"
    block["defines"][f"t_{weight_output_name}"]      = ["type",  output_type_name]
    block["params"] = pack_weights(weight_node["values"], weight_node["bits"])
    if (has_bias):
        block["defines"][f"t_{bias_output_name}_mem"]    = ["type", bias_node["data_type"]]
        output_type_name = f"std::array<std::array<t_{bias_output_name}_mem, {bias_ops}>, 1>"
        block["defines"][f"t_{bias_output_name}"]      = ["type",  output_type_name]
        block["params"] = np.concatenate([block["params"], pack_weights(bias_node["values"], bias_node["bits"], True)])
    if (node["merge_1x1"]):
        block["defines"][f"t_{weight_output_name_1x1}_mem"]    = ["type", weight_node_1x1["data_type"]]
        output_type_name = f"std::array<std::array<t_{weight_output_name_1x1}_mem, {conv_node['ops']}>, {conv_node['ich_ops']}>"
        block["defines"][f"t_{weight_output_name_1x1}"]      = ["type",  output_type_name]
        block["params"] = np.concatenate([block["params"], pack_weights(weight_node_1x1["values"], weight_node_1x1["bits"])])
        if (node["merge_node"]["has_bias"]):
            block["defines"][f"t_{bias_output_name_1x1}_mem"]    = ["type", bias_node_1x1["data_type"]]
            output_type_name = f"std::array<std::array<t_{bias_output_name_1x1}_mem, {bias_ops}>, 1>"
            block["defines"][f"t_{bias_output_name_1x1}"]      = ["type",  output_type_name]
            block["params"] = np.concatenate([block["params"], pack_weights(bias_node_1x1["values"], bias_node_1x1["bits"], True)])
    ### Finish type definitions

    blocks.append(block)
    return blocks

def parse_all(io_dict, model, prj_root="/tmp", board="KRIA", generate_report_file="tmp.rpt"):
    """ This function handles the parsing of the weights and biases, the allocation of the functions 
    to stream the parameters and the resource estimation of them. """
    
    io_connect = graph.extract_connections(model, io_dict)
    stream_graph, shift_cycles, tot_cycles, n_weights, fit = handle_streaming_params(io_dict, model, prj_root, board)
    
    # Saving which layer should be implemented in URAM
    for layer in n_weights:
        if not layer["is_bias"]:
            io_dict[layer["name"]]["uram_storage"] = layer["uram_storage"]

    # parsed_write.append(add_uram_layer())
    # Searching for the first and last layer names.
    # PAY ATTENTION: this is a temporary solution, it should be improved
    # as it is assuming sorted layers
    first_layer = [name for name, layer in io_dict.items() if layer["type"] == "conv"][0]
    parsed_write = []
    for name, node in io_dict.items():
        
        if (node["type"] == "conv"):
            weight_node = io_dict[io_connect[node["input"][1]][0][0]]
            bias_node = None
            weight_node_1x1 = None
            bias_node_1x1 = None

            if (node["has_bias"]):
                bias_node = io_dict[io_connect[node["input"][2]][0][0]]
            
            if (node["merge_1x1"]):
                weight_node_1x1 = io_dict[io_connect[node["input"][3]][0][0]]
                
                if (node["has_bias"]):
                    bias_node_1x1 = io_dict[io_connect[node["input"][4]][0][0]]
            
            param_conv_node = {
                "weight_node": weight_node,
                "bias_node": bias_node,
                "weight_node_1x1": weight_node_1x1,
                "bias_node_1x1": bias_node_1x1,
                "conv_node": node,
                "name": name,
                "shift_cycles": shift_cycles[name],
                "connections": stream_graph[name],
                "first": name == first_layer,
                "last": not "out" in stream_graph[name],
                "params" : tot_cycles
            }
            parsed_write += on_chip_rom(param_conv_node, WIDTH_STREAM)

    # parsed_write[0] = fill_uram_layer(parsed_write)

    print_report(n_weights, fit, generate_report_file)
    return parsed_write

def handle_streaming_params(io_dict, model, prj_root, board="KRIA"):
    """ This function handles the generation of the graph of streaming parameters, the computation of the number of weights to shift for each conv. """
    
    def build_graph(io_dict, io_connect, start_node_name):
        """ Build the graph to stream the weights and biases """
        allowed_types = ["conv"]
        stream_graph = {}
        stream_graph[start_node_name] = {}
        stream_graph[start_node_name]["in"] = "axi_to_stream"
        next_node_name = start_node_name

        while (graph.next_layers(io_dict, io_connect, next_node_name) != None):
            next_node_name = graph.next_layers(io_dict, io_connect, next_node_name)[0]
            next_node = io_dict[next_node_name]

            if (next_node["type"] in allowed_types):
                stream_graph[start_node_name]["out"] = next_node_name
                stream_graph[next_node_name] = {}
                stream_graph[next_node_name]["in"] = start_node_name
                start_node_name = next_node_name

        stream_graph[start_node_name]["out"] = "null"
        return stream_graph
     
    board_res = utils.extract_board_info(board, prj_root)
    io_connect = graph.extract_connections(model, io_dict)
    uram_storage = board_res["uram"] > 0
    
    # Creating a new dictionary with all the information about parameters
    n_weights = []
    for name, node in io_dict.items():
        if ('conv' == node["type"]):

            n_weights.append(
                {
                    "name": name,
                    "n_weights": np.prod(node["weight_quant"]["values"].shape),
                    "par": node["ops"] * node["ich_ops"] * node["kernel"],
                    "bits": node['weight_quant']['bits'],
                    "is_bias": False,
                    "ow": node["ow"],
                    "oh": node["oh"],
                    "ich": node["ich"],
                    "och": node["och"],
                    "iw": node["iw"],
                    "ih": node["ih"],
                    "ops": node["ops"],
                    "ich_ops": node["ich_ops"],
                    "quant_node": node["weight_quant"]
                }
            )

            if (node["has_bias"]):
                n_weights.append(
                    {
                        "name": f"{name} (bias)",
                        "n_weights": np.prod(node["bias_quant"]["values"].shape),
                        "par": 1,
                        "bits": node['bias_quant']['bits'],
                        "is_bias": True,
                        "ow": node["ow"],
                        "oh": node["oh"],
                        "ich": node["ich"],
                        "och": node["och"],
                        "iw": node["iw"],
                        "ih": node["ih"],
                        "ops": node["ops"],
                        "ich_ops": node["ich_ops"],
                        "quant_node": node["bias_quant"]
                    }
                )

            if (node["merge_1x1"]):
                n_weights.append(
                    {
                        "name": f"{name} (merge)",
                        "n_weights": np.prod(node["merge_node"]["weight_quant"]["values"].shape),
                        "par": node["ops"] * node["ich_ops"],
                        "bits": node["merge_node"]['weight_quant']['bits'],
                        "is_bias": False,
                        "ow": node["ow"],
                        "oh": node["oh"],
                        "ich": node["ich"],
                        "och": node["och"],
                        "iw": node["iw"],
                        "ih": node["ih"],
                        "ops": node["ops"],
                        "ich_ops": node["ich_ops"],
                        "quant_node": node["merge_node"]["weight_quant"]
                    }
                )

                if (node["merge_node"]["has_bias"]):
                    n_weights.append(
                        {
                            "name": f"{name} (merge, bias)",
                            "n_weights": np.prod(node["merge_node"]["bias_quant"]["values"].shape),
                            "par": 1,
                            "bits": node["merge_node"]['bias_quant']['bits'],
                            "is_bias": True,
                            "ow": node["ow"],
                            "oh": node["oh"],
                            "ich": node["ich"],
                            "och": node["och"],
                            "iw": node["iw"],
                            "ih": node["ih"],
                            "ops": node["ops"],
                            "ich_ops": node["ich_ops"],
                            "quant_node": node["merge_node"]["bias_quant"]
                        }
                    )

    # Compute the number of weights and biases for each layer 
    read_cycles_per_layer = {}
    for name, node in io_dict.items():
        if (node["type"] == "conv"):
            weight_node = node["weight_quant"]
            read_cycles_per_layer[name] = num_of_words(weight_node["bits"], np.prod(weight_node["values"].shape), WIDTH_STREAM)
            
            if (node["has_bias"]):
                bias_node = node["bias_quant"]
                read_cycles_per_layer[name] += num_of_words(bias_node["bits"], np.prod(bias_node["values"].shape), WIDTH_STREAM)
            
            if (node["merge_1x1"]):
                weight_node_1x1 = node["merge_node"]["weight_quant"]
                read_cycles_per_layer[name] += num_of_words(weight_node_1x1["bits"], np.prod(weight_node_1x1["values"].shape), WIDTH_STREAM)
                
                if (node["merge_node"]["has_bias"]):
                    bias_node_1x1 = node["merge_node"]["bias_quant"]
                    read_cycles_per_layer[name] += num_of_words(bias_node_1x1["bits"], np.prod(bias_node_1x1["values"].shape), WIDTH_STREAM)


    # Save the number of weights to shift for each layer 
    shift_cycles = {}
    cycles_to_shift = sum(read_cycles_per_layer.values())
    for name, cycles in read_cycles_per_layer.items():
        cycles_to_shift -= cycles
        shift_cycles[name] = int(cycles_to_shift)

    fit = sorted_bind_storage(n_weights, board_res, uram_storage)

    # Searching for the first layer name.
    for name, node in io_dict.items():
        if (node["type"] == "conv"):
            input_layers = graph.prev_layers(io_dict, io_connect, name)
            if io_dict[input_layers[0]]["type"] == "produce":
                first_layer = name
                break
    
    # Building a graph based only on parameters streaming
    graph_streaming = build_graph(io_dict, io_connect, first_layer)
    
    return graph_streaming, shift_cycles, sum(read_cycles_per_layer.values()), n_weights, fit

def sorted_bind_storage(dict_layers, board_res, uram_storage=False):
    # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # 36 bits, in which 4 bits are reserved to ECC code
    SIZE_BRAM18 = (18 * 1024)
    
    # Useful space in BRAM36, composed by two BRAM18.
    SIZE_BRAM36 = SIZE_BRAM18 * 2
    
    # Useful space in URAM. Each URAM is 288kb with a maximum word width of 72
    # bits.
    SIZE_URAM = (288 * 1024)

    # Bandwidth of the memory interface (both BRAM and URAM) 
    BANDWIDTH = 72

    used_uram = 0 
    used_bram = 0 
    fit = True

    for node in dict_layers:
        node["sort_metric"] = math.ceil(node["par"] * node["bits"] / BANDWIDTH)

    # Sort in descending order
    n_weights = sorted(dict_layers, key=lambda item: item["sort_metric"], reverse=True).copy()
    for node in n_weights:
        # Number of weights needed in parallel for each clock cycle, which is
        # computed as the number of filter needed to achieved the parallelism
        # on input and output channel multiplied by the dimension of the
        # filter
        if not node["is_bias"]:
            tot_bram = 0
            tot_uram = 0
            w_par = node["par"]
            wpp_uram = BANDWIDTH / node["bits"]
            ports_uram = w_par // wpp_uram
            lpm = node["n_weights"] // node["par"]

            tot_uram = math.ceil(lpm / (SIZE_URAM / BANDWIDTH)) * ports_uram
            tot_bram = compute_bram_layer(node["bits"], node["n_weights"], node["par"], True)
            
            # wasted_uram = n_uram * SIZE_URAM - (node['n_weights'] * node["bits"])
            # wasted_bram = n_bram * SIZE_BRAM - (node['n_weights'] * node["bits"])
            fit_uram = (used_uram + tot_uram) <= board_res["uram"] and uram_storage
            fit_bram = (used_bram + tot_bram) <= board_res["bram"]
            if (not fit_bram and not fit_uram):
                fit = False
                node["mems"] = 0
                print(f"XXX produce_stream_{node['ich']}_{node['och']}_{node['ow']}_{node['oh']}_{node['iw']}_{node['ih']} P:{w_par} of {node['bits']}b. {node['n_weights']}. {tot_uram}U {tot_bram}B. XXX")
            elif ((fit_bram and fit_uram and tot_uram < tot_bram) or not fit_bram):
                used_uram += tot_uram
                # print(f"produce_stream_{node['ich']}_{node['och']}_{node['ow']}_{node['oh']}_{node['iw']}_{node['ih']} P:{w_par} of {node['bits']}b. {node['n_weights']}. {tot_uram}U {tot_bram}B.")
                node["mems"] = tot_uram
                node["quant_node"]["uram_storage"] = True
            else:
                used_bram += tot_bram
                # print(f"produce_stream_{node['ich']}_{node['och']}_{node['ow']}_{node['oh']}_{node['iw']}_{node['ih']} P:{w_par} of {node['bits']}b. {node['n_weights']}. {tot_bram}B {tot_uram}U.")
                node["mems"] = tot_bram
                node["quant_node"]["uram_storage"] = False
        else:
            node["quant_node"]["uram_storage"] = False
            node["mems"] = 0

    return fit    

def print_report(weights_layer_dict, fit, generate_report_file="tmp.rpt"):
    with open(generate_report_file, "a+") as f:
        print("="*40, file=f)
        print("== Memory report", file=f)
        print("="*40, file=f)
        if (fit):
            print("The network should fit in the available resources\n", file=f)
        else:
            print("The network does NOT fit in the available resources\n", file=f)
        table_data = []

        #header row
        header = ["Layer name", "Parallelism", "Bits", "Length", "BRAM", "URAM"]
        table_data.append(header)
        tot_bram = 0
        tot_uram = 0
        for layer in weights_layer_dict:
            row = []
            name = f"{layer['name']}"
            row.append(name)
            row.append(layer["par"])
            row.append(layer["bits"])
            row.append(layer["n_weights"])
            if layer["quant_node"]["uram_storage"]:
                row.append("0")
                row.append(layer["mems"])
                tot_uram += layer["mems"]
            else:
                row.append(layer["mems"])
                row.append("0")
                tot_bram += layer["mems"]
            table_data.append(row)
        
        footer = ["Totals", "", "", "", tot_bram, tot_uram]
        table_data.append(footer)

        # Print the tabulated data to the file
        f.write(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print("\n", file=f)

def init(file_name, network_name, parsed_write, uram_layer_include, prj_root="/tmp"):

    libraries = [
        "params.h",
        "ap_int.h",
        "nn2fpga/mem_utils.h",
        "nn2fpga/weights_utils.h",
        "hls_stream.h",
    ]

    with open(prj_root + ("/cc/include/%s.h" % file_name), "w+") as fd:
        
        fd.write(f"#ifndef __NETWORK_{file_name.upper()}_H__\n")
        fd.write(f"#define __NETWORK_{file_name.upper()}_H__\n")
        
        # Write header with network definitions
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")

        # Handle internal or external parameters
        fd.write("void memory_management(\n")

        for layer in parsed_write:
            for name in layer["input"]:
                fd.write("\tconst t_%s_st *i_data_%s,\n" % (name, name))

        # URAM read handled by external DMA
        for layer in parsed_write:
            for dict in layer["stream_input"]:
                name = dict["name"]
                type = dict["type"]
                fd.write(f"\thls::stream<{type}> {name},\n")

        for i, layer in enumerate(parsed_write):
            for j, dict in enumerate(layer["output"]):
                name = dict["name"]
                type = dict["type"]
                size = dict["size"]
                fd.write(f"\thls::stream<{type}> {name}[{size}]")
                if i < (len(parsed_write)-1) or j < (len(layer["output"])-1):
                    fd.write(",")
                fd.write("\n")

        fd.write(") {\n")

        fd.write("\n")

def footer(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s.h" % file_name, "a") as fd:
        fd.write("\n")
        fd.write("#endif")

def write(io_dict, model, network_name, board="KRIA", generate_report_file="tmp.rpt", prj_root="/tmp"):
    
    parsed_write = parse_all(io_dict, model, prj_root, board, generate_report_file)

    uram_layer_include = False
    # for layer in parsed_write:
    #     if layer['func'] == "load_uram":
    #         uram_download.write(layer, network_name, prj_root)
    #         uram_layer_include = True

    memory_management_file_name = f"memory_management_{network_name}"
    file_path = f"{prj_root}/cc/include/memory_management_{network_name}.h"
    init(memory_management_file_name, network_name, parsed_write, uram_layer_include, prj_root=prj_root)
    declare(file_path, parsed_write, ap_ctrl=None, inline=True, prj_root=prj_root)
    body(file_path, parsed_write, prj_root)
    footer(memory_management_file_name, parsed_write, prj_root)

