import os
import sys
import onnx
from onnx import numpy_helper
import numpy as np
from quant_dorefa import weight_quantize_fn
import torch
from math import log2
from math import ceil

def write(
    model,
    weights_info,
    skip_connections_info,
    bias_info,
    relu_info,
    split_info,
    flatten_info,
    reordered_layers,
    off_chip_storage
):

    forwarded_streams = {}
    replaced_relu = []
    conv_relu = []
    additional_ports = []

    def write_header(
        fd,
        layers_allocated,
        emit_streams=True,
        write_blocks=True
    ):

        if emit_streams:
            # Write header with network definitions
            fd.write("#include \"Network.hpp\"\n")
            fd.write("#include \"hls_stream.h\"\n")
            fd.write("#include \"ap_int.h\"\n")
            fd.write("#include \"hls_stream.h\"\n")
            fd.write("#include \"PackedConv.hpp\"\n")
            fd.write("#include \"ActivationStreams.hpp\"\n")
            fd.write("#include \"AddStreams.hpp\"\n")
            fd.write("#include \"PoolStreams.hpp\"\n")
            fd.write("#include \"Utils.hpp\"\n")

            fd.write("\n")

            # Handle internal or external parameters
            fd.write("void Network(\n")
            fd.write("\thls::stream<t_i_data> &i_data,\n")
            for name in additional_ports:
                fd.write(
                    "\thls::stream<t_%s> s_%s[c_%s_index],\n" % (
                        name, 
                        name, 
                        name
                    )
                )
            fd.write("\thls::stream<t_o_data> &o_data\n")
            fd.write(") {\n")

            fd.write("\n")

            # fd.write("\t#pragma HLS interface m_axi port=i_data depth=10 offset=slave bundle=gmem0\n")
            # fd.write("\t#pragma HLS interface m_axi port=i_weight depth=10 offset=slave bundle=gmem1 max_read_burst_length=256\n")
            # fd.write("\t#pragma HLS interface m_axi port=o_data depth=10 offset=slave\n")
            fd.write("\t#pragma HLS interface axis port=i_data\n")
            fd.write("\t#pragma HLS interface axis port=o_data\n")
            fd.write("\t#pragma HLS INTERFACE ap_ctrl_none port=return\n")
            fd.write("\t#pragma HLS DATAFLOW\n")

            for name in additional_ports:
                channels = "c_%s_och" % name
                fd.write(
                    "\t#pragma HLS STREAM variable=s_%s depth=%s type=fifo\n" % (
                        name,
                        channels
                    )
                )
                fd.write(
                    "\t#pragma HLS INTERFACE mode=ap_fifo port=s_%s\n" % (
                        name
                    )
                )

            # for i, name in enumerate(additional_ports):
            #     fd.write("\t#pragma HLS interface m_axi port=c_%s_st offset=slave bundle=gmem%0d\n" % (name, i))

            fd.write("\n")

            input_name = model.graph.input[0].name.replace(".", "_")
            input_name = input_name.lower().replace("onnx::", "")
            # input_shape = tensors_info[model.graph.input[0].name].tensor_type.shape

            write_stream(fd, "input", "2*c_%s_ich" % input_name)
            fd.write("\t#define c_last_depth 256\n")
            # write_stream(fd, "last", "256")

        fd.write("\n")

        write_last_flags(fd, layers_allocated)

        if write_blocks:
            fd.write("\tProduceStream<\n")
            fd.write("\t\tt_i_data, \n")
            fd.write("\t\tt_%s,\n" % (input_name))
            fd.write("\t\tc_%s_ich,\n" % (input_name))
            fd.write("\t\tc_%s_iw,\n" % (input_name))
            fd.write("\t\tc_%s_ih,\n" % (input_name))
            fd.write("\t\tc_i_data\n")
            fd.write("\t>(\n")
            fd.write("\t\ti_data,\n")
            fd.write("\t\ts_last_split[0],\n")
            fd.write("\t\ts_%s\n" % (input_name))
            fd.write("\t);\n")

            fd.write("\n")

    def write_stream(fd, name, channels=None):
        
        # Each stream has a channel width equal to the number of channels of the 
        # output feature 
        fd.write("\thls::stream<t_%s> s_%s(\"s_%s\");\n" % (name, name, name))
        if (channels is None):
            fd.write(
                "\t#pragma HLS STREAM variable=s_%s depth=3 type=fifo\n" % (
                    name
                )
            )
        else:
            fd.write(
                "\t#pragma HLS STREAM variable=s_%s depth=%s type=fifo\n" % (
                    name,
                    channels
                )
            )

    def write_array_stream(fd, name, channels=None, c_split=None):
        
        # Each stream has a channel width equal to the number of channels of the 
        # output feature 
        if (c_split is not None):
            fd.write(
                "\thls::stream<t_%s> s_%s[c_%s_split];\n" % (
                    name,
                    name,
                    name
                )
            )
        else:
            fd.write(
                "\thls::stream<t_%s> s_%s[c_%s_ih*c_%s_iw];\n" % (
                    name, 
                    name,
                    name,
                    name
                )
            )

        if (channels is None):
            fd.write(
                "\t#pragma HLS STREAM variable=s_%s depth=3 type=fifo\n" % (
                    name
                )
            )
        else:
            fd.write(
                "\t#pragma HLS STREAM variable=s_%s depth=%s type=fifo\n" % (
                    name,
                    channels
                )
            )

    def write_last_flags(fd, layers_allocated):
        fd.write(
            "\thls::stream<ap_uint<1>> s_last_split[%0d];\n" % (
                layers_allocated + 1
            )
        )
        fd.write(
            "\t#pragma HLS STREAM variable=s_last_split depth=10 type=fifo\n"
        )

        # fd.write("\tSplitStream<\n")
        # fd.write("\t\t%0d\n" % (layers_allocated + 1))
        # fd.write("\t>(\n")
        # fd.write("\t\ts_last,\n")
        # fd.write("\t\ts_last_split\n")
        # fd.write("\t);\n")

        fd.write("\n")

    def write_internal_weight(
        fd,
        name,
        node_name,
        c_stride,
        emit_streams=True,
        write_blocks=True,
        weight_shape=None,
        pack_weights=True,
        off_chip_storage=False
    ):

        if emit_streams:
            if (not off_chip_storage):
                write_array_stream(fd, name, "c_%s_och" % name)

            fd.write("\n")

        if write_blocks:
            if (len(getattr(weight_shape, 'dims')) > 2):
                c_ih     = getattr(weight_shape, 'dims')[2]
                c_iw     = getattr(weight_shape, 'dims')[3]
            else:
                c_ih     = 1
                c_iw     = 1

            if pack_weights:
                if (not off_chip_storage):
                    fd.write("\tProduceStream<\n")
                    fd.write("\t\tt_%s_st,\n" % (name))
                    fd.write("\t\tt_%s,\n" % (name))
                    fd.write("\t\tc_%s_ich,\n" % (name))
                    fd.write("\t\tc_%s_och,\n" % (name))
                    fd.write("\t\tc_%s_ow,\n" % (node_name))
                    # fd.write("\t\tc_%s_oh\n" % (node_name))
                    fd.write("\t\tc_%s_oh,\n" % (node_name))
                    fd.write("\t\tc_%s_fw,\n" % (node_name))
                    fd.write("\t\tc_%s_fh,\n" % (node_name))
                    fd.write("\t\tc_%s_ops\n" % (node_name))
                    fd.write("\t>(\n")
                    fd.write("\t\tc_%s_st,\n" % (name))
                    fd.write("\t\ts_last_%s[0],\n" % (
                            node_name
                        )
                    )
                    fd.write("\t\ts_%s\n" % (name))
                    fd.write("\t);\n")

                    fd.write("\n")
            else:
                for ih in range(c_ih):
                    for iw in range(c_iw):
                        fd.write("\tProduceStream<\n")
                        fd.write("\t\tt_%s_st,\n" % (name))
                        fd.write("\t\tt_%s,\n" % (name))
                        fd.write("\t\tc_%s_ich,\n" % (name))
                        fd.write("\t\tc_%s_och,\n" % (name))
                        fd.write("\t\tc_%s_ow,\n" % (node_name))
                        # fd.write("\t\tc_%s_oh\n" % (node_name))
                        fd.write("\t\tc_%s_oh,\n" % (node_name))
                        fd.write("\t\tc_%s_ops\n" % (node_name))
                        fd.write("\t>(\n")
                        fd.write("\t\tc_%s_st_%0d,\n" % (name, ih*c_iw+iw))
                        fd.write("\t\ts_last_%s[%0d],\n" % (
                                node_name, ih*c_iw+iw
                            )
                        )
                        fd.write("\t\ts_%s[%0d]\n" % (name, ih*c_iw+iw))
                        fd.write("\t);\n")

                        fd.write("\n")

    def write_relu(
        fd,
        node,
        emit_streams=True,
        write_blocks=True,
        last_flag=0
    ):

        node_name = node.name.replace(".", "_").lower()
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        if emit_streams:
            write_stream(fd, output_name, "c_%s_ich" % node_name)
            fd.write("\n")

        if write_blocks:
            fd.write("\tReluStreams<\n")
            fd.write("\t\tt_%s,\n" % (input_name))
            fd.write("\t\tt_%s,\n" % (output_name))
            fd.write("\t\tc_%s_ich,\n" % (node_name))
            fd.write("\t\tc_%s_iw,\n" % (node_name))
            fd.write("\t\tc_%s_ih\n" % (node_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (input_name))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag+1))
            fd.write("\t\ts_%s\n" % (output_name))
            fd.write("\t);\n")

            fd.write("\n")

    def write_add(fd, node, emit_streams=True, write_blocks=True):

        node_name = node.name.replace(".", "_").lower()
        input_name0 = node.input[0].replace(".", "_")
        input_name0 = input_name0.lower().replace("onnx::", "")

        input_name1 = node.input[1].replace(".", "_")
        input_name1 = input_name1.lower().replace("onnx::", "")

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) < 2):
                skip_name = skip_connections_info[node.name][0].replace(".", "_")
                skip_name = skip_name.lower().replace("onnx::", "")

                # Checking to which stream is directed the skip connection
                if (skip_name.replace("_skip", "") == input_name0):
                    input_name0 = skip_name
                if (skip_name.replace("_skip", "") == input_name1):
                    input_name1 = skip_name

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        if emit_streams:
            write_stream(fd, output_name, "c_%s_ich" % node_name)
            fd.write("\n")

        if write_blocks:
            fd.write("\tAddStreams<\n")
            fd.write("\t\tt_%s,\n" % (input_name0))
            fd.write("\t\tt_%s,\n" % (output_name))
            fd.write("\t\tc_%s_ich,\n" % (node_name))
            fd.write("\t\tc_%s_iw,\n" % (node_name))
            fd.write("\t\tc_%s_ih\n" % (node_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (input_name0))
            fd.write("\t\ts_%s,\n" % (input_name1))
            fd.write("\t\ts_%s\n" % (output_name))
            fd.write("\t);\n")

            fd.write("\n")

    def write_pool(
        fd,
        node,
        emit_streams=True,
        write_blocks=True,
        last_flag=0,
        c_pool=0
    ):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        if output_name in flatten_info.keys():
            output_name = flatten_info[output_name][1]
        output_name = output_name.lower().replace("onnx::", "")

        if emit_streams:
            write_stream(fd, output_name, "c_%s_och" % node_name)
            fd.write("\n")

        if write_blocks:
            fd.write("\tPoolStreams<\n")
            fd.write("\t\tt_%s,\n" % (input_name))
            fd.write("\t\tt_%s,\n" % (output_name))
            fd.write("\t\tt_%s_acc,\n" % (node_name))
            fd.write("\t\tc_%s_ich,\n" % (node_name))
            fd.write("\t\tc_%s_och,\n" % (node_name))
            fd.write("\t\tc_%s_iw,\n" % (node_name))
            fd.write("\t\tc_%s_ih,\n" % (node_name))
            fd.write("\t\tc_%s_ow,\n" % (node_name))
            fd.write("\t\tc_%s_oh,\n" % (node_name))
            fd.write("\t\tc_%s_fw,\n" % (node_name))
            fd.write("\t\tc_%s_fh,\n" % (node_name))
            fd.write("\t\tc_%s_stride,\n" % (node_name))
            fd.write("\t\tc_%s_pad,\n" % (node_name))
            fd.write("\t\tc_%s_pool\n" % (node_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (input_name))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag+1))
            fd.write("\t\ts_%s\n" % (output_name))
            fd.write("\t);\n")

            fd.write("\n")

    def write_pad(
        fd,
        node,
        emit_streams=True,
        write_blocks=True,
        last_flag=0
    ):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        if emit_streams:
            write_stream(fd, output_name)
            fd.write("\n")

        if write_blocks:
            fd.write("\tPadStream<\n")
            fd.write("\t\tt_%s,\n" % (input_name))
            fd.write("\t\tt_%s,\n" % (output_name))
            fd.write("\t\tc_%s_ich,\n" % (node_name))
            fd.write("\t\tc_%s_och,\n" % (node_name))
            fd.write("\t\tc_%s_iw,\n" % (node_name))
            fd.write("\t\tc_%s_ih,\n" % (node_name))
            fd.write("\t\tc_%s_ow,\n" % (node_name))
            fd.write("\t\tc_%s_oh,\n" % (node_name))
            fd.write("\t\tc_%s_pad\n" % (node_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (input_name))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag+1))
            fd.write("\t\ts_%s\n" % (output_name))
            fd.write("\t);\n")

            fd.write("\n")

    def write_weights(
        weight_shape,
        weight_name,
        node_name,
        pack_weights=True,
        off_chip_storage=False
    ):

        c_och    = getattr(weight_shape, 'dims')[0]
        c_ich    = getattr(weight_shape, 'dims')[1]
        if (len(getattr(weight_shape, 'dims')) > 2):
            c_ih     = getattr(weight_shape, 'dims')[2]
            c_iw     = getattr(weight_shape, 'dims')[3]
        else:
            c_ih     = 1
            c_iw     = 1

        # fd.write("\ttypedef ap_int<8> t_%s_st;\n" % (weight_name))
        # fd.write("\ttypedef ap_int<8> t_%s;\n" % (weight_name))
        # fd.write("\tconst int c_%s_och = %d;\n" % (weight_name, c_och))
        # fd.write("\tconst int c_%s_ich = %d;\n" % (weight_name, c_ich))
        # fd.write("\tconst int c_%s_ih  = %d;\n" % (weight_name, c_ih))
        # fd.write("\tconst int c_%s_iw  = %d;\n" % (weight_name, c_iw))
        weights = numpy_helper.to_array(
            weight_shape
        )

        wact = weight_quantize_fn(w_bit=8)
        wact.export = False

        # TODO: less dirty
        # weights, max_w = wact(torch.Tensor(weights))
        weights = wact(torch.Tensor(weights))
        weights = np.asarray(weights)

        weights = weights * 128

        # Rescaled to have more precision in hardware, the comma position is
        # taken into account when the output data is extracted from s_acc
        sw = 7 - ceil(log2(np.amax(np.abs(weights))))

        weights = weights * (2**sw)
        weights = np.round(weights)
        fd.write("\tconst int c_%s_scale = %0d;\n" % (node_name, sw))

        # TODO: Specialized for DAC2023 submission, must be automated

        parallel_ops = {}
        parallel_ops['conv_0']  = 1                                                    
        parallel_ops['conv_2']  = 2                                                    
        parallel_ops['conv_4']  = 2                                                    
        parallel_ops['conv_7']  = 2                                                    
        parallel_ops['conv_9']  = 2                                                    
        parallel_ops['conv_12']  = 2                                                   
        parallel_ops['conv_14']  = 2                                                   
        parallel_ops['conv_17']  = 1                                                   
        parallel_ops['conv_19']  = 1                                                   
        parallel_ops['conv_21']  = 1                                                   
        parallel_ops['conv_24']  = 1                                                   
        parallel_ops['conv_26']  = 1                                                   
        parallel_ops['conv_29']  = 1                                                   
        parallel_ops['conv_31']  = 1                                                   
        parallel_ops['conv_34']  = 1                                                   
        parallel_ops['conv_36']  = 1                                                   
        parallel_ops['conv_38']  = 1                                                   
        parallel_ops['conv_41']  = 1                                                   
        parallel_ops['conv_43']  = 1                                                   
        parallel_ops['conv_46']  = 1                                                   
        parallel_ops['conv_48']  = 1                                                   
        parallel_ops['conv_54']  = 1  
        #################################################################

        last_weight = True
        if (pack_weights):
            if (not off_chip_storage):
                fd.write("\tconst int%0d_t c_%s_st[c_%s_fh*c_%s_fw][c_%s_och*c_%s_ich/%0d+1] = {\n" % (
                        8*parallel_ops[node_name],
                        weight_name,
                        node_name,
                        node_name,
                        node_name,
                        node_name,
                        parallel_ops[node_name]
                    )
                )


                for ih in range(c_ih):
                    for iw in range(c_iw):
                        fd.write("{")
                        for ich in range(weights.shape[1]):
                            for och in range(int(weights.shape[0]/parallel_ops[node_name])):
                                weight_value = 0
                                for op in range(parallel_ops[node_name]):
                                    # weight_value = np.random.randint(0, 256)
                                    weight_value |= int(weights[och+op][ich][ih][iw]) << (8*op)
                                fd.write("%0d" % (weight_value))
                                fd.write(", ")
                        fd.write("0")
                        if (ih==(c_ih-1)) and (iw==(c_iw-1)):
                            fd.write("}\n")
                        else:
                            fd.write("},\n")

                fd.write("};\n")
                fd.write("\n")
                fd.write(
                    "\t#pragma HLS ARRAY_PARTITION variable=c_%s_st type=block factor=1 dim=1\n" % (
                        weight_name
                    )
                )
            else:
                weights_export = np.zeros(
                    [
                        c_ih,
                        c_iw,
                        weights.shape[1],
                        int(weights.shape[0]/parallel_ops[node_name])
                    ]
                )
                for ih in range(c_ih):
                    for iw in range(c_iw):
                        for ich in range(weights.shape[1]):
                            for och in range(int(weights.shape[0]/parallel_ops[node_name])):
                                weight_value = 0
                                for op in range(parallel_ops[node_name]):
                                    # weight_value = np.random.randint(0, 256)
                                    weight_value |= int(weights[och+op][ich][ih][iw]) << (8*op)
                                weights_export[ih][iw][ich][och] = weight_value
                np.save("backup/%s" % node_name, weights_export)

                
        else:
            for ih in range(c_ih):
                for iw in range(c_iw):
                    fd.write("\tconst int8_t c_%s_st_%0d[] = {\n" % (weight_name, ih*c_iw+iw))
                    for ich in range(weights.shape[1]):
                        for och in range(weights.shape[0]):
                            # weight_value = np.random.randint(0, 256)
                            weight_value = weights[och][ich][ih][iw]
                            fd.write("%0d" % (weight_value))
                            fd.write(", ")

                    fd.write("0")

                    fd.write("};\n")
                    fd.write("\n")

    def write_conv(
        fd,
        node,
        emit_streams=True,
        write_blocks=True,
        last_flag=0,
        gemm=None
    ):

        node_name = node.name.replace(".", "_").lower()

        # Assuming no skip connection at the start
        no_skip = True
        bias = False
        split = False
        pointwise = False

        # Removing dots from input names
        input_name = node.input[0]

        indexed = False
        if (input_name in split_info.keys()):
            indexed = True
            index = split_info[input_name].index(node.name)

        input_name = input_name.replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) < 2):
                skip_name = skip_connections_info[node.name][0].replace(".", "_")
                skip_name = skip_name.lower().replace("onnx::", "")

                # Checking to which stream is directed the skip connection
                if (skip_name.replace("_skip", "") == input_name):
                    input_name = skip_name

        weight_name = node.input[1].replace(".", "_")
        weight_name = weight_name.lower().replace("onnx::", "")
        weight_shape = weights_info[node.input[1]]

        if (not emit_streams) and (not write_blocks):
            additional_ports.append(weight_name)
            return

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) > 1):
                skip_name = skip_connections_info[node.name][1].replace(".", "_")
                skip_name = skip_name.lower().replace("onnx::", "")
                no_skip = False

                # Declaring copied stream
                if emit_streams:
                    depth = "c_%s_och*(c_%s_fh-1)*(c_%s_iw+c_%s_fw-1)" % (node_name, node_name, node_name, node_name)
                    write_stream(
                        fd,
                        skip_name,
                        "c_%s_ich*c_%s_och" % (node_name, node_name)
                        # depth
                    )
                fd.write("\n")

        # Adding BIAS and merging to add layer
        output_name = node.output[0]
        if output_name in bias_info.keys():
            bias = True
            bias_name = bias_info[output_name][0]
            bias_name = bias_name.replace(".", "_")
            bias_name = bias_name.lower().replace("onnx::", "")

            output_name = bias_info[output_name][1]

        # Merging RELU to conv
        if output_name in relu_info.keys():
            replaced_relu.append(relu_info[output_name][0])
            conv_relu.append(node.name)
            output_name = relu_info[output_name][1]

        # Bypassing flatten
        if output_name in flatten_info.keys():
            output_name = flatten_info[output_name][1]

        if output_name in split_info.keys():
            split = True

        output_name = output_name.replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        if write_blocks:
            if (not off_chip_storage):
                fd.write("\tSplitStream<\n")
                fd.write("\t\tc_%s_split\n" % (node_name))
                fd.write("\t>(\n")
                fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
                fd.write("\t\ts_last_%s\n" % (node_name))
                fd.write("\t);\n")
                fd.write("\n")

        if emit_streams:
            if (split):
                write_array_stream(
                    fd,
                    output_name,
                    "c_%s_ich*c_%s_och" % (node_name, node_name),
                    # 20,
                    2
                )
            else:
                write_stream(
                    fd,
                    output_name,
                    "c_%s_ich*c_%s_och" % (node_name, node_name),
                    # 20,
                )
            fd.write("\n")

        attributes = getattr(node, "attribute" )
        # Specializing by stride, only 1 and 2 are supported right now
        if (gemm is None):
            c_fh     = int(getattr(attributes[2], 'ints')[0])
            c_fw     = int(getattr(attributes[2], 'ints')[1])
            c_stride = getattr(attributes[4], 'ints')[0]
        else:
            c_fh = 1
            c_fw = 1
            c_stride = 1

        if (c_fh*c_fw) == 1:
        	pointwise = True 

        if emit_streams:
            fd.write("\n")

            if (not off_chip_storage):
                fd.write(
                    "\thls::stream<ap_uint<1>> s_last_%s[c_%s_split];\n" % (
                        node_name,
                        node_name
                    )
                )
                fd.write(
                    "\t#pragma HLS STREAM variable=s_last_%s depth=10 type=fifo\n" % (
                        node_name
                    )
                )

                fd.write("\n")

            write_weights(
                weight_shape,
                weight_name,
                node_name,
                pack_weights=True,
                off_chip_storage=off_chip_storage
            )
        # Given stride a different weight stream is selected
        write_internal_weight(
            fd,
            weight_name,
            node_name,
            c_stride,
            emit_streams,
            write_blocks,
            weight_shape,
            off_chip_storage=off_chip_storage
        )

        fd.write("\n")

        if write_blocks:

            if (no_skip):
                fd.write("\tPackedConvBuffAcc<\n")
                fd.write("\t\tt_%s,\n" % (input_name))
                fd.write("\t\tt_%s,\n" % (weight_name))
                fd.write("\t\tt_%s,\n" % (output_name))
                if (bias):
                    fd.write("\t\tt_%s,\n" % (input_name))
                fd.write("\t\tt_%s_acc,\n" % (node_name))
                fd.write("\t\tc_%s_ich,\n" % (node_name))
                fd.write("\t\tc_%s_och,\n" % (node_name))
                fd.write("\t\tc_%s_iw,\n" % (node_name))
                fd.write("\t\tc_%s_ih,\n" % (node_name))
                fd.write("\t\tc_%s_ow,\n" % (node_name))
                fd.write("\t\tc_%s_oh,\n" % (node_name))
                if (not pointwise):
                  fd.write("\t\tc_%s_fw,\n" % (node_name))
                  fd.write("\t\tc_%s_fh,\n" % (node_name))
                fd.write("\t\tc_%s_relu,\n" % (node_name))
                if (split):
                    fd.write("\t\tc_%s_split,\n" % (output_name))
                fd.write("\t\tc_%s_stride,\n" % (node_name))
                fd.write("\t\tc_%s_pad,\n" % (node_name))
                fd.write("\t\tc_%s_scale,\n" % (node_name))
                fd.write("\t\tc_%s_ops\n" % (node_name))
                fd.write("\t> (\n")
                if indexed:
                    fd.write("\t\ts_%s[%d],\n" % (input_name, index))
                else:
                    fd.write("\t\ts_%s,\n" % (input_name))
                fd.write("\t\ts_%s,\n" % (weight_name))
                if (bias):
                    fd.write("\t\ts_%s,\n" % (bias_name))
                if (not off_chip_storage):
                    fd.write("\t\ts_last_%s[1],\n" % (node_name))
                else:
                    fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
                fd.write("\t\ts_last_split[%0d],\n" % (last_flag + 1))
                fd.write("\t\ts_%s\n" % (output_name))
                fd.write("\t);\n")
            else:
                fd.write("\tPackedConvBuffAcc<\n")
                fd.write("\t\tt_%s,\n" % (input_name))
                fd.write("\t\tt_%s,\n" % (weight_name))
                fd.write("\t\tt_%s,\n" % (output_name))
                fd.write("\t\tt_%s_acc,\n" % (node_name))
                fd.write("\t\tc_%s_ich,\n" % (node_name))
                fd.write("\t\tc_%s_och,\n" % (node_name))
                fd.write("\t\tc_%s_iw,\n" % (node_name))
                fd.write("\t\tc_%s_ih,\n" % (node_name))
                fd.write("\t\tc_%s_ow,\n" % (node_name))
                fd.write("\t\tc_%s_oh,\n" % (node_name))
                if (not pointwise):
                  fd.write("\t\tc_%s_fw,\n" % (node_name))
                  fd.write("\t\tc_%s_fh,\n" % (node_name))
                fd.write("\t\tc_%s_relu,\n" % (node_name))
                if (split):
                    fd.write("\t\tc_%s_split,\n" % (output_name))
                fd.write("\t\tc_%s_stride,\n" % (node_name))
                fd.write("\t\tc_%s_pad,\n" % (node_name))
                fd.write("\t\tc_%s_scale,\n" % (node_name))
                fd.write("\t\tc_%s_ops\n" % (node_name))
                fd.write("\t> (\n")
                if indexed:
                    fd.write("\t\ts_%s[%d],\n" % (input_name, index))
                else:
                    fd.write("\t\ts_%s,\n" % (input_name))
                fd.write("\t\ts_%s,\n" % (weight_name))
                if (not off_chip_storage):
                    fd.write("\t\ts_last_%s[1],\n" % (node_name))
                else:
                    fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
                fd.write("\t\ts_last_split[%0d],\n" % (last_flag + 1))
                fd.write("\t\ts_%s,\n" % (output_name))
                fd.write("\t\ts_%s\n" % (skip_name))
                fd.write("\t);\n")

            fd.write("\n")

    def count_allocated(model):

        layers_n = 0

        for node_level in reordered_layers:
            for node in node_level:

                if 'gemm' in node.op_type.lower():
                    layers_n = layers_n + 1
                    continue

                if 'conv' in node.op_type.lower():
                    layers_n = layers_n + 1
                    continue

                if 'add' == node.op_type.lower():
                    # write_add(fd, node)
                    continue

                if 'relu' == node.op_type.lower():
                    # if node.name not in replaced_relu:
                        # layers_n = layers_n + 1
                    continue

                if 'pool' in node.op_type.lower():
                    if 'average' in node.op_type.lower():
                        layers_n = layers_n + 1
                    continue

                if 'pad' in node.op_type.lower():
                    layers_n = layers_n + 1

        return layers_n

    def write_body(fd, model, emit_streams=True, write_blocks=True):

        layers_n = 0

        for node_level in reordered_layers:
            for node in node_level:
                print(node.op_type.lower())

                if 'gemm' in node.op_type.lower():
                    write_conv(fd, node, emit_streams, write_blocks, layers_n, gemm=True)
                    layers_n = layers_n + 1
                    continue

                if 'conv' in node.op_type.lower():
                    write_conv(fd, node, emit_streams, write_blocks, layers_n)
                    layers_n = layers_n + 1
                    continue

                if 'add' == node.op_type.lower():
                    # write_add(fd, node)
                    continue

                # TODO: Write Relu and thinks about folding, if the buffer is small
                # then there is a good chance the overhead is negligible

                if 'relu' == node.op_type.lower():
                    if node.name not in replaced_relu:
                        write_relu(fd, node, emit_streams, write_blocks, layers_n)
                        layers_n = layers_n + 1
                    continue

                if 'pool' in node.op_type.lower():
                    c_pool = 0
                    if 'average' in node.op_type.lower():
                        c_pool = 0

                    if 'max' in node.op_type.lower():
                        c_pool = 1

                    write_pool(fd, node, emit_streams, write_blocks, layers_n)
                    layers_n = layers_n + 1
                    continue

                if 'pad' in node.op_type.lower():
                    write_pad(fd, node, emit_streams, write_blocks, layers_n)
                    layers_n = layers_n + 1

    def write_footer(fd, layers_allocated):

        for output in model.graph.output:
            output_name = output.name.replace(".", "_")
            output_name = output_name.lower().replace("onnx::", "")

            fd.write("\tConsumeStream<\n")
            fd.write("\t\tt_%s,\n" % (output_name))
            fd.write("\t\tt_o_data,\n")
            fd.write("\t\tc_%s_och,\n" % (output_name))
            fd.write("\t\tc_%s_ow,\n" % (output_name))
            fd.write("\t\tc_%s_oh\n" % (output_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (output_name))
            fd.write("\t\ts_last_split[%0d],\n" % layers_allocated)
            fd.write("\t\to_data\n")
            fd.write("\t);\n")

            # End of main file
            fd.write("}\n")

    with open("src/Network.cpp", "w+") as fd:

        layers_allocated = count_allocated(model)

        # parse additional ports for off-chip parameters storage
        if off_chip_storage:
            write_body(fd, model, emit_streams=False, write_blocks=False)

        write_header(fd, layers_allocated)

        write_body(fd, model, emit_streams=True, write_blocks=False)
        write_body(fd, model, emit_streams=False, write_blocks=True)

        write_footer(fd, layers_allocated)

    return conv_relu, additional_ports
