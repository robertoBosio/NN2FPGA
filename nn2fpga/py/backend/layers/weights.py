import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import math
import json
from backend.graph import extract_connections
from backend.utils import *
from backend.layers.uram_download import fill_uram_layer, add_uram_layer
import backend.layers.uram_download as uram_download
from backend.layers.quant import get_quant_type

def parse_on_chip(
    node_info,
    pre_values,
    bits=8,
    signed=1,
    narrow=1,
    dynamic_init=False
):
    # Transforming a filter of dimension [iw][ih][ich][och] into one of
    # dimension [iw * ih][(ich * och)/ops][ops] where ops is the number 2D convolutions
    # computed in parallel

    dich = node_info["ich"]
    dih  = node_info["ih"]
    diw  = node_info["iw"]
    doch = node_info["och"]
    dops = node_info["ops"]
    dich_ops = node_info["ich_ops"]
    scale_factor = 2**node_info["scale_factor"]

    narrow_h = 0
    if not signed and narrow:
      narrow_h = 1
    limit_h = 2**(bits-signed)-1-narrow_h

    narrow_l = 0
    if signed and narrow:
      narrow_l = 1
    limit_l = -1*signed*2**(bits-signed)+narrow_l

    doch_ops = int(doch/dops)
    dich_iter_ops = int(dich/dich_ops)

    # check that ich_iter_ops and och_ops are greater than 0
    # if (dich_ops > 1):
    #     print("dich_iter_ops: %d" % dich_iter_ops)
    #     print("dich_ops: %d" % dich_ops)
    #     print("depth: %d" % node_info["depth"])
    #     print("dich: %d" % dich)
    #     print("doch: %d" % doch)
    #     print("dops: %d" % dops)

    assert dich_iter_ops > 0
    assert doch_ops > 0

    values = np.zeros(
        [dih*diw, dich_iter_ops*doch_ops, dich_ops*dops]
    )

    for ih in range(dih-1, -1, -1):
        for iw in range(diw-1, -1, -1):
            for ich in range(dich_iter_ops):
                for och in range(doch_ops):
                    off = och*dops
                    for ich_ops in range(dich_ops):
                        off_ich = ich*dich_ops + ich_ops
                        for ops in range(dops):
                            quant_value = pre_values[off+ops][off_ich][ih][iw]
                            quant_value = np.round(quant_value/scale_factor)
                            if (limit_h < quant_value):
                                quant_value = limit_h

                            if (limit_l > quant_value):
                                quant_value = limit_l
                            
                            if not dynamic_init:
                                quant_value = quant_value*scale_factor

                            index = ih*diw+iw
                            ch = ich*doch_ops+och
                            ops_index = ich_ops*dops+ops
                            values[dih*diw-1-index][ch][ops_index] = quant_value
    
    return values

def pack_weights(
    values,
    bits=8,
):

    values = np.swapaxes(values,0,1)
    values = values.flatten()

    if bits >= 8:
        bytes_num = int(bits/8)
        new_values = np.zeros([values.shape[0]*bytes_num])

        for i in range(values.shape[0]):
            data = values[i]
            data_bytes = np.zeros([bytes_num])
            for j in range(bytes_num):
                data_byte = int(data) & 0xff
                data_bytes[j] = data_byte
                data = int(data // 256)

            # Changing MSB to LSB order to ease hw reconstruction
            # of the original value
            for j in range(bytes_num-1,-1,-1):
                new_values[i*bytes_num+j] = data_bytes[bytes_num - 1 - j]
            # for j in range(0, bytes_num):
            #     new_values[i*bytes_num+j] = data_bytes[j]

        values = new_values
    elif bits < 8:
        pack_width = int(8/bits)
        new_values = np.zeros([int(values.shape[0]/pack_width)])
        mask = 2**bits - 1

        for i in range(0, values.shape[0], pack_width):
            data_byte = 0
            for j in range(pack_width):
                data = values[i+j]
                data_byte |= (int(data) & mask) << (j*bits)

            new_values[int(i/pack_width)] = data_byte

        values = new_values
    
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
    uram_storage
):
    pre_values = numpy_helper.to_array(init_info[weight_name]) 
    shape = pre_values.shape

    och = node_info["och"]
    ich = node_info["ich"]
    if node_info["depth"]:
        och = 1

    if len(shape) > 1:
        is_bias = False
    else:
        if (not node_info["depth"]):
            ich = 1
        pre_values = np.expand_dims(pre_values, axis=-1)
        is_bias = True

    if (len(shape) > 2):
        ih = shape[2]
        iw = shape[3]
    else:
        ih = 1
        iw = 1
        pre_values = np.expand_dims(pre_values, axis=-1)
        pre_values = np.expand_dims(pre_values, axis=-1)

    oh     = node_info["oh"]
    ow     = node_info["ow"]
    stride = node_info["stride"]
    pad    = node_info["pad"]

    ops = node_info["ops"]
    if is_bias:
        ich_ops = 1
    else:
        ich_ops = node_info["ich_ops"]

    if ops > och:
        ops = och

    signed = new_node["signed"]
    scale_factor = new_node["scale_factor"]
    bits   = new_node["bits"]
    narrow = new_node["narrow"]
    bw = int(128/bits)

    data_type = get_quant_type(signed, bits, scale_factor)

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
    # Check if ops are mode than och and clip
    if ops > och:
        new_node["ops"] = och
    else:
        new_node["ops"] = ops
    new_node["ich_ops"] = ich_ops
    new_node["is_bias"] = is_bias
    new_node["type"]    = 'const'
    new_node["kernel"]  = ih*iw
    new_node["total"]   = ich*och*ih*iw*oh*ow/stride
    new_node["n_weights"] = ich*och*ih*iw
    new_node["dynamic_init"] = dynamic_init
    # assert that uram_storage allows only if dynamic_init
    assert not(uram_storage and not(dynamic_init))
    new_node["uram_storage"] = uram_storage and not(is_bias)
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
    uram_storage=False
):
    new_nodes = {}
    rem_nodes = []

    io_connect = extract_connections(model, io_dict)

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
                uram_storage
            )

            rem_nodes.append(node_name)
            
    for node_name, node_info in new_nodes.items():
        io_dict[node_name] = node_info

    for node_name in rem_nodes:
        del io_dict[node_name]

    return io_dict

def parse_main(io_dict, dynamic_init):
    
    block = {}
    block["func"] = "memory_management"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["output"] = []

    block["declare"] = []
    block["defines"] = {}
    block["pragma"] = []

    uram_storage = False

    for name, node in io_dict.items():
        if 'const' == node["type"]:
            output_name = node["output"][0]
            if node["off_chip_memory"]:
                block["input"].append("%s" % output_name)
                block["args"].append("i_data_%s" % output_name)

                pragma = {}
                pragma["name"] = "interface"
                options = [
                    ["port", "i_data_%s" % output_name],
                    ["mode", "m_axi"],
                ]
                pragma["options"] = options
                block["pragma"].append(pragma)
            
            if node["uram_storage"]:
                # FIX: removing biases from URAM storage, they are few
                uram_storage = True

    if dynamic_init:
        output_name = "weights"
        block["stream_input"].append("%s" % output_name)
        block["args"].append("i_data_%s" % output_name)

        block["defines"]["t_%s_st" % output_name] = [
            "type", 
            "ap_uint<8>"
        ]

        block["defines"]["t_%s_stream" % output_name] = [
            "type", 
            "ap_axiu<8, 0, 0, 0>"
        ]

        block["defines"]["t_%s_st" % output_name] = [
            "type", 
            "uint8_t"
        ]

        pragma = {}
        pragma["name"] = "interface"
        options = [
            ["port", "i_data_%s" % output_name],
            ["mode", "axis"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

        block["is_const"] = True
 

    for name, node in io_dict.items():
        if 'const' == node["type"]:
            input_name = node["input"][0]
            output_name = node["output"][0]
            block["args"].append("s_%s" % output_name)

            tmp = {}
            tmp["name"] = "s_%s" % output_name
            tmp["type"] = "t_%s" % output_name
            tmp["dim"]  = node["ih"]*node["iw"]
            tmp["is_array"] = True

            block["declare"].append(tmp)

            pragma = {}
            pragma["name"] = "stream"
            options = [
                ["variable", "s_%s" % output_name],
                ["depth", "2"],
                ["type", "fifo"],
            ]
            pragma["options"] = options
            block["pragma"].append(pragma)

            pragma = {}
            pragma["name"] = "array_partition"
            options = [
                ["variable", "s_%s" % output_name],
                ["type", "complete"],
            ]
            pragma["options"] = options
            block["pragma"].append(pragma)

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

def on_chip_rom(
    name,
    node,
    input_name,
    output_name
):

    blocks = []
    block = {}

    uram_storage = node["uram_storage"]
    dynamic_init = node["dynamic_init"]

    block["func"] = "produce_stream"
    block["args"] = []
    block["input"] = []
    block["stream_input"] = []
    block["uram_input"] = []
    block["output"] = []
    block["bits"] = node["bits"]
    block["index"] = node["ih"]*node["iw"]
    block["och"] = node["och"]
    # FIX: adding this check to avoid problems in merged pipelines
    # with same inputs but different output channels
    # Check if ops are mode than och and clip
    if node["ops"] > node["och"]:
        block["ops"] = node["och"]
    else:
        block["ops"] = node["ops"]
    block["ich_ops"] = node["ich_ops"]
    block["dynamic_init"] = dynamic_init
    block["uram_storage"] = uram_storage

    block["template"] = []
    block["template"].append("t_%s_st" % (output_name))
    if dynamic_init:
        block["template"].append("t_%s_init" % (output_name))
    block["template"].append("t_%s" % (output_name))
    block["template"].append("c_%s_ich" % (name))
    block["template"].append("c_%s_och" % (name))
    block["template"].append("c_%s_ow" % (name))
    block["template"].append("c_%s_oh" % (name))
    block["template"].append("c_%s_iw" % (output_name))
    block["template"].append("c_%s_ih" % (output_name))
    block["template"].append("c_%s_ops" % (name))
    block["template"].append("c_%s_ich_ops" % (name))
    block["template"].append("c_%s_reuse" % (output_name))

    block["output"].append("%s" % output_name)
    block["args"].append("c_%s" % output_name)
    if dynamic_init:
        block["args"].append("s_%s_init" % output_name)
        block["args"].append("s_%s_init_flag" % output_name)
    block["args"].append("s_%s" % output_name)

    pre_values = node["pre_values"]
    if node["depth"]:
        pre_values = np.swapaxes(pre_values, 0, 1)

    node["values"] = parse_on_chip(
        node,
        pre_values,
        node["bits"],
        node["signed"],
        node["narrow"],
        node["dynamic_init"],
    )

    if dynamic_init:
        block["uram_input"].append("s_%s_init" % output_name)
        block["uram_total"] = [node["n_weights"]]

        # Declare only in tb wrapper
        block["tb_declare"] = []
        tmp = {}
        tmp["name"] = "c_%s" % output_name
        tmp["type"] = "t_%s" % output_name
        tmp["is_array"] = True
        tmp["is_const"] = not uram_storage
        values = pack_weights(node["values"], node["bits"])
        tmp["size"] = values.shape
        tmp["init"] = values

        block["tb_declare"].append(tmp)

    block["declare"] = []

    tmp = {}
    if dynamic_init:
        tmp["name"] = "static c_%s" % output_name
    else:
        tmp["name"] = "c_%s" % output_name
    tmp["type"] = "t_%s_st" % output_name
    tmp["is_array"] = True
    tmp["is_const"] = not dynamic_init
    size = node["values"].shape
    tmp["size"] = size
    tmp["init"] = node["values"]
    tmp["form"] = "float"

    block["declare"].append(tmp)

    if dynamic_init:
        tmp = {}
        tmp["name"] = "s_%s_init_flag" % output_name
        tmp["type"] = "static bool"
        tmp["is_array"] = False
        tmp["is_const"] = False
        tmp["size"] = 1
        tmp["init"] = None
        block["declare"].append(tmp)

        tmp = {}
        tmp["name"] = "s_%s_init" % output_name
        tmp["type"] = "t_%s_init" % output_name
        tmp["is_array"] = True
        tmp["dim"] = node["ih"]*node["iw"]

        block["declare"].append(tmp)

    block["defines"] = {}
    block["defines"]["t_%s_st" % (output_name)]    = ["type", node["data_type"]]
    output_type_name = "std::array<std::array<%s, %0d>, %0d>" % (node["data_type"], node["ops"], node["ich_ops"])
    if dynamic_init:
        block["defines"]["t_%s_init" % (output_name)]    = ["type", "t_%s_st" % output_name]
    block["defines"]["t_%s" % (output_name)]       = ["type",  output_type_name]
    block["defines"]["c_%s_ich" % (name)]          = ["const", node["ich"]]
    block["defines"]["c_%s_och" % (name)]          = ["const", node["och"]]
    block["defines"]["c_%s_ops" % (name)]          = ["const", node["ops"]]
    block["defines"]["c_%s_ich_ops" % (name)]      = ["const", node["ich_ops"]]
    block["defines"]["c_%s_ow" % (name)]           = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % (name)]           = ["const", node["oh"]]
    block["defines"]["c_%s_iw" % (output_name)]    = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % (output_name)]    = ["const", node["ih"]]
    block["defines"]["c_%s_ops" % (output_name)]   = ["const", node["ops"]]
    block["defines"]["c_%s_reuse" % (output_name)] = ["const", node["reuse"]]

    block["pragma"] = []
    #######################################################################
    # pragma = {}
    # pragma["name"] = "array_partition"
    # options = [
    #     ["variable", "c_%s" % (output_name)],
    #     ["type", "block"],
    #     ["factor", 1],
    #     ["dim", 1],
    # ]
    # pragma["options"] = options

    # block["pragma"].append(pragma)
    #######################################################################
    #######################################################################
    if uram_storage:
        pragma = {}
        pragma["name"] = "bind_storage"
        options = [
            ["variable", "c_%s" % (output_name)],
            ["impl", "URAM"],
            ["type", "RAM_T2P"]
        ]
        pragma["options"] = options

        block["pragma"].append(pragma)

    if uram_storage:
        interface_width = 72
    else:
        interface_width = 64
    
    # FIX: Reducing bias resource usage increasing the produce_stream II
    if node["is_bias"]:
        parallelism = 1
        dim_1_reshape_factor = 1
        dim_2_reshape_factor = 1
        dim_3_reshape_factor = 1
        dim_1_partition_factor = 1
        dim_3_partition_factor = 1
    else: 
        parallelism = node["ops"] * node["ich_ops"]

        # Compute the number of weights fitting a word.
        divider = interface_width // node["bits"]

        if divider == 1:
            dim_1_reshape_factor = 1
        else:
            # If the weights in a filter do not perfectly fit in a word of the
            # memory, do not reshape on the first dimension
            if ((node["ih"] * node["iw"]) % divider) != 0:
                dim_1_reshape_factor = 1
            else:
                dim_1_reshape_factor = np.clip( node["ih"] * node["iw"], 1, divider)
            divider = divider//dim_1_reshape_factor

        if parallelism > divider:
            dim_3_reshape_factor = divider
            divider = 1
        else: 
            dim_3_reshape_factor = parallelism
            divider = divider//parallelism

        if dim_3_reshape_factor >= parallelism:
            dim_3_partition_factor = 1
        else:
            dim_3_partition_factor = math.ceil(parallelism/dim_3_reshape_factor)

        if divider > 1:
            dim_2_reshape_factor = divider
            divider = 1
        else:
            dim_2_reshape_factor = 1

        if dim_1_reshape_factor == 1:
            dim_1_partition_factor = node["ih"]*node["iw"]
        elif dim_1_reshape_factor < (node["ih"]*node["iw"]):
            dim_1_partition_factor = math.ceil(node["ih"]*node["iw"]/dim_1_reshape_factor)
        else:
            dim_1_partition_factor = 1

    #######################################################################

    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", "c_%s" % (output_name)],
        ["dim", 3],
    ]

    if dim_3_reshape_factor < parallelism:
        options.append(["factor", dim_3_reshape_factor])
        options.append(["type", "cyclic"])
    else:
        dim_3_reshape_factor = parallelism
        options.append(["type", "complete"])

    pragma["options"] = options
    if dim_3_reshape_factor > 1:
        block["pragma"].append(pragma)

    #######################################################################
    pragma = {}
    pragma["name"] = "array_partition"
    options = [
        ["variable", "c_%s" % (output_name)],
        ["dim", 3],
    ]

    if dim_3_partition_factor > 1:
        options.append(["factor", dim_3_partition_factor])
        options.append(["type", "cyclic"])

    pragma["options"] = options
    if dim_3_partition_factor > 1:
        block["pragma"].append(pragma)

    #######################################################################

    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", "c_%s" % (output_name)],
        ["dim", 1],
    ]

    if dim_1_reshape_factor > 1:
        options.append(["factor", dim_1_reshape_factor])
        options.append(["type", "cyclic"])

    pragma["options"] = options

    if dim_1_reshape_factor > 1:
        block["pragma"].append(pragma)

    #######################################################################
    if (dim_2_reshape_factor > 1):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", "c_%s" % (output_name)],
            ["type", "cyclic"],
            ["factor", dim_2_reshape_factor],
            ["dim", 2],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    #######################################################################
    pragma = {}
    pragma["name"] = "array_partition"
    options = [
        ["variable", "c_%s" % (output_name)],
        ["dim", 1],
    ]

    if dim_1_partition_factor >= parallelism:
        options.append(["type", "complete"])
    else:
        options.append(["factor", dim_1_partition_factor])
        options.append(["type", "cyclic"])

    pragma["options"] = options
    if dim_1_partition_factor > 1:
        block["pragma"].append(pragma)

    if dynamic_init:
        pragma = {}
        pragma["name"] = "stream"
        options = [
            ["variable", "s_%s_init" % output_name],
            ["depth", "2"],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    block["size"] = node["iw"]*node["ih"]
    block["is_const"] = True
    blocks.append(block)

    return blocks

def parse(name, node):
    
    input_name = node["input"][0]
    output_name = node["output"][0]

    if node["off_chip_memory"]:
        blocks = off_chip_ddr(
            name,
            node,
            input_name,
            output_name
        )

    else:

        blocks = on_chip_rom(
            name,
            node,
            input_name,
            output_name
        )
      
    return blocks

def parse_all(io_dict, prj_root="/tmp", board="KRIA"):
    
    parsed_write = []
    
    # Opening JSON file
    file_path = f"{prj_root}/../nn2fpga/boards/{board}.json"
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block 
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}
    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    
    # Check if there is URAM storage
    dynamic_init = False

    
    # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # 36 bits, in which 4 bits are reserved to ECC code
    SIZE_BRAM18 = ((18 * 1024) / 36) * 32
    
    # Useful space in BRAM36, composed by two BRAM18.
    SIZE_BRAM36 = SIZE_BRAM18 * 2
    
    # Useful space in URAM. Each URAM is 288kb with a maximum word width of 72
    # bits.
    SIZE_URAM = (288 * 1024)

    n_weights = {}
    DSPs = 0
    used_uram = used_bram = 0
    print(board_res)
    for name, node in io_dict.items():
        if ('const' == node["type"]) and node["dynamic_init"]:
            n_weights[name] = node["n_weights"] * node["bits"] / 8
            
            # Number of weights needed in parallel for each clock cycle, which is
            # computed as the number of filter needed to achieved the parallelism
            # on input and output channel multiplied by the dimension of the
            # filter
            if not node["is_bias"]:
                w_par = node["ops"] * node ["ich_ops"] * node["ih"] * node["iw"]
                w_par_bits = w_par * node["bits"]
                DSPs += w_par
                target_pragmas = 0

                n_mem_bram = math.ceil(w_par_bits / 64)
                n_mem_uram = math.ceil(w_par_bits / 72)
                n_uram = math.ceil((node['n_weights'] * node["bits"]) / SIZE_URAM)
                n_bram = math.ceil((node['n_weights'] * node["bits"]) / SIZE_BRAM36)
                if (n_uram < n_mem_uram):
                    n_uram = n_mem_uram
                if (n_bram < n_mem_bram):
                    n_bram = n_mem_bram
                # wasted_uram = n_uram * SIZE_URAM - (node['n_weights'] * node["bits"])
                # wasted_bram = n_bram * SIZE_BRAM - (node['n_weights'] * node["bits"])
                fit_uram = (used_uram + n_uram) <= board_res["uram"]
                fit_bram = (used_bram + n_bram) <= board_res["bram"]
                # fit_uram = True
                # fit_bram = True
                if (not fit_bram and not fit_uram):
                    print("There is not enough space.")
                    exit(-1)
                elif ((fit_bram and fit_uram and n_uram < n_bram) or not fit_bram):
                    used_uram += n_uram
                    target_pragmas = n_uram
                    print(f"{name} P:{w_par} of {node['bits']}b. {node['n_weights']}. {n_uram}U {n_bram}B.")
                    io_dict[name]["uram_storage"] = True
                    bandwidth = 72
                else:
                    used_bram += n_bram
                    target_pragmas = n_bram
                    print(f"{name} P:{w_par} of {node['bits']}b. {node['n_weights']}. {n_bram}B {n_uram}U.")
                    io_dict[name]["uram_storage"] = False
                    bandwidth = 64
                
                # Pragma section
                reshape_factor = bandwidth // node["bits"]
                partition_factor = math.ceil(w_par_bits / bandwidth)
                actual_bandwidth = node["bits"]
                actual_ports = 1
                reshape_pragma = [ 1, 1, 1 ]
                partition_pragma = [ 1, 1, 1 ]
                dim = [ 1, 1, 1 ]
                dim[0] = node["ih"] * node ["iw"]
                dim[2] = node["ops"] * node ["ich_ops"]
                dim[1] = (node["n_weights"] // dim[0]) // dim[2]
                print(f"Starting from {actual_ports}x mem[{dim[0]}][{dim[1]}][{dim[2]}] x {actual_bandwidth}. ") 

                # Try perfectly fitting reshaping only on dimension 1 and 3
                order = [2, 0, 1]
                index = 0
                while (reshape_factor != 1):
                   
                   # Dimension under consideration for reshaping
                    curr_dim = order[index]
                    d = dim[curr_dim]

                    reshape_fit = (reshape_factor % d) == 0
                    reshape_fit |= (d % reshape_factor) == 0
                    reshape_fit &= d != 1
                    if (reshape_fit):
                        reshape_pragma[curr_dim] *= np.clip(reshape_factor, 1, d)
                        reshape_factor = reshape_factor // reshape_pragma[curr_dim]
                        dim[curr_dim] //= reshape_pragma[curr_dim]
                        print(f" - Reshape dim={curr_dim} factor={reshape_pragma[curr_dim]}, ", end="")
                        actual_bandwidth = actual_bandwidth * reshape_pragma[curr_dim]
                        print(f"{actual_ports}x mem[{dim[0]}][{dim[1]}][{dim[2]}] x {actual_bandwidth}. ") 
                    
                    if (dim[curr_dim]) == 1 or not reshape_fit:
                        index += 1
                        if (index == 2):
                            break;

                # Push for not perfectly fitting reshaping, wasting space 
                index = 0
                while (reshape_factor != 1):
                   
                   # Dimension under consideration for reshaping
                    curr_dim = order[index]
                    d = dim[curr_dim]

                    reshape_fit = (d != 1 and d > reshape_factor)
                    if (reshape_fit):
                        curr_reshape = np.clip(reshape_factor, 1, d)
                        reshape_pragma[curr_dim] *= curr_reshape
                        reshape_factor = 1
                        dim[curr_dim] = math.ceil(dim[curr_dim] / curr_reshape)
                        print(f" - Reshape dim={curr_dim} factor={reshape_pragma[curr_dim]}, ", end="")
                        actual_bandwidth = actual_bandwidth * reshape_pragma[curr_dim]
                        print(f"{actual_ports}x mem[{dim[0]}][{dim[1]}][{dim[2]}] x {actual_bandwidth}. ") 
                    
                    if (dim[curr_dim]) == 1 or not reshape_fit:
                        index += 1
                        if (index == 3):
                            break;

                # Partitoning 
                index = 0
                while (partition_factor > 1):
                   
                   # Dimension under consideration for reshaping
                    curr_dim = order[index]
                    d = dim[curr_dim]

                    partition_fit = (partition_factor % d) == 0
                    partition_fit |= (d % partition_factor) == 0
                    partition_fit &= d != 1
                    if (partition_fit):
                        curr_partition = np.clip(partition_factor, 1, d)
                        partition_pragma[curr_dim] *= curr_partition
                        partition_factor = partition_factor // curr_partition
                        dim[curr_dim] = dim[curr_dim] // curr_partition
                        print(f" - Partition dim={curr_dim} factor={partition_pragma[curr_dim]}, ", end="")
                        actual_ports = actual_ports * partition_pragma[curr_dim]
                        print(f"{actual_ports}x mem[{dim[0]}][{dim[1]}][{dim[2]}] x {actual_bandwidth}. ") 
                    
                    if (dim[curr_dim]) == 1 or not partition_fit:
                        index += 1
                        if (index == 2):
                            break;

                if (partition_factor > 1 or reshape_factor > 1):
                    "PARALLELISM NOT ACHIEVED"
                elif (target_pragmas != actual_ports):
                    "ERROR"

    print(f"Totally used {used_bram} BRAMs and {used_uram} URAMs, {DSPs} DSPs.")
    # Sort in descending order
    # n_weights = {k: v for k, v in sorted(n_weights.items(), key=lambda item: item[1], reverse=True)}
    #print(n_weights) 

    # Count the number of URAMs and remove the layers that do not fit
    # uram_count = 0
    # MAX_COUNT = board_res['uram']
    # for name, node in n_weights.items():
    #     uram_count = uram_count + math.ceil(node/36864)
    #     # print(f"{name} before: {io_dict[name]['uram_storage']}, now {uram_count < MAX_COUNT}")
    #     if uram_count > MAX_COUNT:
    #         io_dict[name]["uram_storage"] = False

    # print(uram_count)
    for name, node in io_dict.items():

        if ('const' == node["type"]) and node["dynamic_init"]:
            dynamic_init = True

    if dynamic_init:
        parsed_write.append(add_uram_layer())

    for name, node in io_dict.items():

        # FIX: adding the hybrid bram/uram scheme
        if ('const' == node["type"]):
            parsed_write = parsed_write + parse(name, node)

    if dynamic_init:
        parsed_write[0] = fill_uram_layer(parsed_write)

    return parsed_write

def init(file_name, network_name, parsed_write, uram_layer_include, prj_root="/tmp"):

    libraries = [
        "params.h",
        "ap_int.h",
        "nn2fpga/mem_utils.h",
        "nn2fpga/weights_utils.h",
        "hls_stream.h",
    ]

    if uram_layer_include:
        libraries.append("load_uram_%s.h" % network_name)

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
            for name in layer["stream_input"]:
                fd.write("\thls::stream<t_%s_stream> &i_data_%s,\n" % (name, name))

        for i, layer in enumerate(parsed_write):
            for j, name in enumerate(layer["output"]):
                fd.write(
                    "\thls::stream<t_%s> s_%s[%0d]" % (
                        name,
                        name,
                        layer["size"]
                    )
                )
                if i < (len(parsed_write)-1) or j < (len(layer["output"])-1):
                    fd.write(",")
                fd.write("\n")

        fd.write(") {\n")

        fd.write("\n")

def footer(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s.h" % file_name, "a") as fd:
        fd.write("\n")
        fd.write("#endif")

def write(io_dict, network_name, board="KRIA", prj_root="/tmp"):

    parsed_write = parse_all(io_dict, prj_root, board)

    uram_layer_include = False
    for layer in parsed_write:
        if layer['func'] == "load_uram":
            uram_download.write(layer, network_name, prj_root)
            uram_layer_include = True

    memory_management_file_name = "memory_management_%s" % network_name
    file_path = f"{prj_root}/cc/include/memory_management_{network_name}.h"
    init(memory_management_file_name, network_name, parsed_write, uram_layer_include, prj_root=prj_root)
    declare(file_path, parsed_write, ap_ctrl=None, inline=True, prj_root=prj_root)
    body(file_path, parsed_write, prj_root)
    footer(memory_management_file_name, parsed_write, prj_root)

