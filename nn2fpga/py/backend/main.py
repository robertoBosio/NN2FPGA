import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import backend.layers.conv as conv
import backend.layers.pool as pool
import backend.layers.line_buffer as line_buffer
import backend.layers.pad as pad
import backend.layers.weights as weights
import backend.layers.input_gen as input_gen
import backend.layers.output_gen as output_gen
import backend.layers.detect as detect
import backend.layers.non_max_suppression as non_max_suppression
import backend.layers.bandwidth_adjust as bandwidth_adjust
import backend.graph as graph
from backend.utils import *

def init(file_name, parsed_write, object_detection=False, off_chip_storage=False, prj_root="/tmp"):


    libraries = [
        "params.h",
        "ap_int.h",
        "hls_stream.h",
        "nn2fpga/packed_conv.h",
        "nn2fpga/pool_streams.h",
        "nn2fpga/utils.h",
        "nn2fpga/activations_utils.h",
        "nn2fpga/block_interface.h",
        "nn2fpga/line_buffer_utils.h",
        "nn2fpga/quantisation.h",
        "nn2fpga/weights_utils.h",
    ]

    if (object_detection):
        libraries.append("nn2fpga/detect_utils.h")
        libraries.append("nn2fpga/non_max_suppression.h")
        libraries.append("hls_np_channel.h")
    
    if (off_chip_storage):
        libraries.append("nn2fpga/memory_management.h")

    with open(prj_root + ("/cc/include/%s.h" % file_name), "w+") as fd:

        fd.write(f"#ifndef __{file_name.upper()}__H__\n")
        fd.write(f"#define __{file_name.upper()}__H__\n")
        fd.write("#include \"params.h\"\n\n")


        # Handle internal or external parameters
        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &i_%s,\n" % (name, name))

        for layer in parsed_write:
            if (off_chip_storage):
                if "memory_management" == layer["func"]:
                    for dict in layer["input"]:
                        name = dict["name"]
                        type = dict["type"]
                        fd.write(f"\tconst {type} *{name},\n")

                    for dict in layer["stream_input"]:
                        name = dict["name"]
                        type = dict["type"]
                        fd.write(f"\thls::stream<{type}> &{name},\n")
            else:
                if "axi_to_stream" == layer["func"]:
                    for dict in layer["stream_input"]:
                        name = dict["name"]
                        type = dict["type"]
                        fd.write(f"\thls::stream<{type}> &{name},\n")

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(");\n\n")
        fd.write(f"#endif  /*__{file_name.upper()}__H__ */")
    
    # Writing also the header file. This is mandatory for the top layer as the
    # testbench need to have the definition of the function when using Vitis HLS
    with open(prj_root + ("/cc/src/%s.cc" % file_name), "w+") as fd:

        # Write header with network definitions
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)
        fd.write("\n")
        fd.write("#include \"params.h\"\n\n")
        fd.write("extern \"C++\" {\n\n")
        
        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &i_%s,\n" % (name, name))

        for layer in parsed_write:
            if (off_chip_storage):
                if "memory_management" == layer["func"]:
                    for dict in layer["input"]:
                        name = dict["name"]
                        type = dict["type"]
                        fd.write(f"\tconst {type} *{name},\n")

                    for dict in layer["stream_input"]:
                        name = dict["name"]
                        type = dict["type"]
                        fd.write(f"\thls::stream<{type}> &{name},\n")
            else:
                if "axi_to_stream" == layer["func"]:
                    for dict in layer["stream_input"]:
                        name = dict["name"]
                        type = dict["type"]
                        fd.write(f"\thls::stream<{type}> &{name},\n")

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(") {\n")
        fd.write("\n")

def handle_parameters(io_dict, model, board, off_chip_storage, prj_root, generate_report_file): 
    
    # Writing the separate file for the memory management only in case of off-chip.
    # In the other case the memory management is handled directly by the conv.
    if off_chip_storage: 
        block = weights.parse_main(io_dict)
    else:
        io_connect = graph.extract_connections(model, io_dict)
        graph_streaming, shift_cycles, tot_cycles, n_weights, fit = weights.handle_streaming_params(io_dict, model, prj_root, board)
        for name, node in io_dict.items():
            if 'conv' == node["type"]:
                node["shift_cycles"] = shift_cycles[name]
                node["shift_params_connections"] = graph_streaming[name]
                print(node)
                weights.print_report(n_weights, fit, generate_report_file)
        block = weights.generate_axitostandard_stream(tot_cycles)
        
        for name, node in io_dict.items():
            if (node["type"] == "conv"):
                weight_node = io_connect[node["input"][1]][0][0]
                for elem in n_weights:
                    if elem["name"] == weight_node:
                        node["uram_storage"] = elem["uram_storage"]
                
                if (node["merge_1x1"]):
                    weight_node_1x1 = io_connect[node["input"][3]][0][0]
                    for elem in n_weights:
                        if elem["name"] == weight_node_1x1:
                            node["uram_storage_1x1"] = elem["uram_storage"]

        # Adding declaration of the stream input and pragma interface
        block["stream_input"].append({"name" : "i_data_params", "type" : "t_params_axi_stream"})
        pragma = {}
        pragma["name"] = "interface"
        options = [
            ["port", "i_data_params"],
            ["mode", "axis"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    block["defines"] = {}
    block["defines"]["t_params_stream"] = [
        "type", 
        "ap_uint<8>"
    ]

    block["defines"]["t_params_axi_stream"] = [
        "type", 
        "ap_axiu<8, 0, 0, 0>"
    ]

    block["defines"]["t_params_st"] = [
        "type", 
        "uint8_t"
    ]
    return block

def parse_all_main(io_dict, model, off_chip_storage=False):

    parsed_write = []
    parsed_const = []
    no_output_gen = False
    last_node = None

    for name, node in io_dict.items():

        if 'produce' == node["type"]:
            parsed_write.append(
                input_gen.parse(name, node)
            )

        if 'depth' == node["type"]:
            continue

        if 'conv' == node["type"]:
            if (node["adjust_line_buffer"]):
                adjust_name = conv.get_input_name(node)
                parsed_write = parsed_write + bandwidth_adjust.parse(name, node, adjust_name, "in_ops", "adjust_ops", "ow_ops", "ow_ops", dim="i")

            if (node["adjust_add"]):
                adjust_name = conv.get_add_name(node)
                parsed_write = parsed_write + bandwidth_adjust.parse(name, node, adjust_name, "add_ops", "adjust_add_ops", "ow_ops", "adjust_add_ow_ops", dim="o", skip=True)

            parsed_write = parsed_write + line_buffer.parse(name, node)
            if (node["pad"] != 0) or (node["ow_ops"] > 1):
                parsed_write.append(pad.parse(name, node))

            parsed_write = parsed_write + conv.parse(name, node, off_chip_storage)
            parsed_const = parsed_const + weights.parse_const(io_dict, model, node)
            last_node = node

        if 'pool' == node["type"]:
            if (node["adjust_line_buffer"]):
                adjust_name = conv.get_input_name(node)
                parsed_write = parsed_write + bandwidth_adjust.parse(name, node, adjust_name, "in_ops", "adjust_ops", "ow_ops", "ow_ops", dim="i")
            if (not node["is_adaptive"]):
                parsed_write = parsed_write + line_buffer.parse(name, node)
                parsed_write.append(
                    pad.parse(name, node)
                )
            parsed_write.append(
                pool.parse(name, node)
            )
            last_node = node

        # Just for the sake of constant definition
        # if 'const' == node["type"]:
        #     parsed_const = parsed_const + weights.parse_const(
        #         name,
        #         node
        #     )

        if 'detect' == node["type"]:
            parsed_write.append(
                detect.parse(name, node)
            )

        if 'non_max_suppression' == node["type"]:
            parsed_write.append(
                non_max_suppression.parse(name, node)
            )
            no_output_gen = True


        last_node_name = name

    if not no_output_gen:
        parsed_write.append(
            output_gen.parse(last_node, last_node_name)
        )

    return parsed_write, parsed_const

def defines(parsed_write, prj_root="/tmp"):

    libraries = [
        "ap_int.h",
        "hls_stream.h",
        "hls_vector.h",
        "stdint.h",
        "ap_axi_sdata.h",
    ]

    # Writing parameters of the network. They are written in a separate file
    # such that also the testbench can read them.
    with open(f"{prj_root}/cc/include/params.h", "w+") as fd:
        
        fd.write("#ifndef __NN2FPGA_NETWORK_PARAMS_H__\n")
        fd.write("#define __NN2FPGA_NETWORK_PARAMS_H__\n")
        fd.write("\n")
        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        fd.write("\n")
        for layer in parsed_write:

            if 'defines' in layer.keys():
                write_defines(fd, layer["defines"], layer['func'])

        fd.write("\n")
        fd.write("#endif /*__NN2FPGA_NETWORK_PARAMS_H__ */")

def footer(file_path):

    with open(file_path, "a") as fd:
        fd.write("\n}")

def write(
        io_dict, 
        model, 
        file_name, 
        ap_ctrl_chain, 
        object_detection, 
        dynamic_init, 
        board, 
        off_chip_storage, 
        prj_root="/tmp", 
        generate_report_file="tmp.rpt"
        ):

    if ap_ctrl_chain:
        ap_ctrl = "ap_ctrl_chain"
    else:
        ap_ctrl = "ap_ctrl_none"

    parsed_write_mem = handle_parameters(io_dict, model, board, off_chip_storage, prj_root, generate_report_file)
    parsed_write, parsed_const = parse_all_main(io_dict, model, off_chip_storage)
    parsed_write.insert(0, parsed_write_mem)
    
    file_path = f"{prj_root}/cc/src/{file_name}.cc"
    init(file_name, parsed_write, object_detection, off_chip_storage, prj_root=prj_root)
    declare(file_path, parsed_write, ap_ctrl, prj_root=prj_root)
    body(file_path, parsed_write, prj_root=prj_root)
    footer(file_path)

    parsed_write = parsed_write + parsed_const

    defines(parsed_write, prj_root=prj_root)
