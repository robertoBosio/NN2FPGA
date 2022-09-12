import os
import sys
import onnx
from onnx import numpy_helper

def write(
    model,
    weights_info,
    skip_connections_info,
    bias_info
):

    forwarded_streams = {}

    def write_header(fd):

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
        fd.write("\tt_i_data* i_data,\n")
        fd.write("\tt_weight* i_weight,\n")
        fd.write("\tt_o_data* o_data\n")
        fd.write(") {\n")

        fd.write("\n")

        fd.write("\t#pragma HLS interface m_axi port=i_data depth=10 offset=slave bundle=gmem0\n")
        fd.write("\t#pragma HLS interface m_axi port=i_weight depth=10 offset=slave bundle=gmem1 max_read_burst_length=256\n")
        fd.write("\t#pragma HLS interface m_axi port=o_data depth=10 offset=slave\n")
        fd.write("\t#pragma HLS INTERFACE mode=ap_ctrl_chain port=return\n")
        fd.write("\t#pragma HLS DATAFLOW\n")

        fd.write("\n")

        input_name = model.graph.input[0].name.replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")
        # input_shape = tensors_info[model.graph.input[0].name].tensor_type.shape

        write_stream(fd, "input", "c_%s_ich" % input_name)

        fd.write("\n")

        fd.write("\tProduceStream<\n")
        fd.write("\t\tt_i_data, \n")
        fd.write("\t\tt_%s,\n" % (input_name))
        fd.write("\t\tc_%s_ich,\n" % (input_name))
        fd.write("\t\tc_%s_iw,\n" % (input_name))
        fd.write("\t\tc_%s_ih\n" % (input_name))
        fd.write("\t>(\n")
        fd.write("\t\ti_data,\n")
        fd.write("\t\ts_%s\n" % (input_name))
        fd.write("\t);\n")

        fd.write("\n")

    def write_stream(fd, name, channels=None):
        
        # Each stream has a channel width equal to the number of channels of the 
        # output feature 
        fd.write("\thls::stream<t_%s> s_%s(\"s_%s\");\n" % (name, name, name))
        if (channels is None):
            fd.write(
                "\t#pragma HLS STREAM variable=s_%s depth=2 type=fifo\n" % (
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

    def write_array_stream(fd, name, channels=None, c_stride=1):
        
        # Each stream has a channel width equal to the number of channels of the 
        # output feature 
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
                "\t#pragma HLS STREAM variable=s_%s depth=2 type=fifo\n" % (
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

    def write_internal_weight(fd, name, node_name, c_stride):

        write_array_stream(fd, name, "c_%s_och" % name, c_stride)

        fd.write("\n")

        fd.write("\tProduceStream<\n")
        fd.write("\t\tt_%s_st,\n" % (name))
        fd.write("\t\tt_%s,\n" % (name))
        fd.write("\t\tc_%s_ich,\n" % (name))
        fd.write("\t\tc_%s_och,\n" % (name))
        fd.write("\t\tc_%s_iw,\n" % (name))
        fd.write("\t\tc_%s_ih,\n" % (name))
        fd.write("\t\tc_%s_iw,\n" % (node_name))
        fd.write("\t\tc_%s_ih\n" % (node_name))
        fd.write("\t>(\n")
        fd.write("\t\tc_%s_st,\n" % (name))
        fd.write("\t\ts_%s\n" % (name))
        fd.write("\t);\n")

        fd.write("\n")

    def write_relu(fd, node):

        node_name = node.name.replace(".", "_").lower()
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        write_stream(fd, output_name, "c_%s_ich" % node_name)

        fd.write("\n")

        fd.write("\tReluStreams<\n")
        fd.write("\t\tt_%s,\n" % (input_name))
        fd.write("\t\tt_%s,\n" % (output_name))
        fd.write("\t\tc_%s_ich,\n" % (node_name))
        fd.write("\t\tc_%s_iw,\n" % (node_name))
        fd.write("\t\tc_%s_ih\n" % (node_name))
        fd.write("\t> (\n")
        fd.write("\t\ts_%s,\n" % (input_name))
        fd.write("\t\ts_%s\n" % (output_name))
        fd.write("\t);\n")

        fd.write("\n")

    def write_add(fd, node):

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

        write_stream(fd, output_name, "c_%s_ich" % node_name)

        fd.write("\n")

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

    def write_average_pool(fd, node):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        write_stream(fd, output_name, "c_%s_och" % node_name)

        fd.write("\n")

        fd.write("\tAveragePoolStreams<\n")
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
        fd.write("\t\tc_%s_pad\n" % (node_name))
        fd.write("\t> (\n")
        fd.write("\t\ts_%s,\n" % (input_name))
        fd.write("\t\ts_%s\n" % (output_name))
        fd.write("\t);\n")

        fd.write("\n")

    def write_pad(fd, node):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        write_stream(fd, output_name)

        fd.write("\n")

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
        fd.write("\t\ts_%s\n" % (output_name))
        fd.write("\t);\n")

        fd.write("\n")

    def write_weights(weight_shape, weight_name):

        c_och    = getattr(weight_shape, 'dims')[0]
        c_ich    = getattr(weight_shape, 'dims')[1]
        c_ih     = getattr(weight_shape, 'dims')[2]
        c_iw     = getattr(weight_shape, 'dims')[3]

        # fd.write("\ttypedef ap_uint<8> t_%s_st;\n" % (weight_name))
        # fd.write("\ttypedef ap_uint<8> t_%s;\n" % (weight_name))
        # fd.write("\tconst int c_%s_och = %d;\n" % (weight_name, c_och))
        # fd.write("\tconst int c_%s_ich = %d;\n" % (weight_name, c_ich))
        # fd.write("\tconst int c_%s_ih  = %d;\n" % (weight_name, c_ih))
        # fd.write("\tconst int c_%s_iw  = %d;\n" % (weight_name, c_iw))
        fd.write("\tconst ap_uint<8> c_%s_st[] = {\n" % (weight_name))
        
        weights = numpy_helper.to_array(
            weight_shape
        )

        # TODO: handle weights quantization
        last_weight = True
        for och in range(weights.shape[0]):
            for ich in range(weights.shape[1]):
                for ih in range(weights.shape[2]):
                    for iw in range(weights.shape[3]):
                        fd.write("%0.3f" % (weights[och][ich][ih][iw]))
                        fd.write(", ")

        fd.write("0")

        fd.write("};\n")
        fd.write("\n")

    def write_conv(fd, node):

        node_name = node.name.replace(".", "_").lower()

        # Assuming no skip connection at the start
        no_skip = True
        bias = False

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
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

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) > 1):
                skip_name = skip_connections_info[node.name][1].replace(".", "_")
                skip_name = skip_name.lower().replace("onnx::", "")
                no_skip = False

                # Declaring copied stream
                write_stream(fd, skip_name, "c_%s_ich" % node_name)
                fd.write("\n")

        output_name = node.output[0]
        if output_name in bias_info.keys():
            bias = True
            bias_name = bias_info[output_name][0]
            bias_name = bias_name.replace(".", "_")
            bias_name = bias_name.lower().replace("onnx::", "")

            output_name = bias_info[output_name][1]

        output_name = output_name.replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")

        write_stream(fd, output_name, "c_%s_ich" % node_name)

        fd.write("\n")

        attributes = getattr(node, "attribute" )
        # Specializing by stride, only 1 and 2 are supported right now
        c_stride = getattr(attributes[4], 'ints')[0]

        write_weights(weight_shape, weight_name)
        # Given stride a different weight stream is selected
        write_internal_weight(fd, weight_name, node_name, c_stride)

        fd.write("\n")

        if (no_skip):
            fd.write("\tPackedConvBuffAcc<\n")
            fd.write("\t\tt_%s,\n" % (input_name))
            fd.write("\t\tt_%s,\n" % (weight_name))
            fd.write("\t\tt_%s,\n" % (output_name))
            fd.write("\t\tt_%s,\n" % (input_name))
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
            fd.write("\t\tc_%s_pad\n" % (node_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (input_name))
            fd.write("\t\ts_%s,\n" % (weight_name))
            if (bias):
                fd.write("\t\ts_%s,\n" % (bias_name))
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
            fd.write("\t\tc_%s_fw,\n" % (node_name))
            fd.write("\t\tc_%s_fh,\n" % (node_name))
            fd.write("\t\tc_%s_stride,\n" % (node_name))
            fd.write("\t\tc_%s_pad\n" % (node_name))
            fd.write("\t> (\n")
            fd.write("\t\ts_%s,\n" % (input_name))
            fd.write("\t\ts_%s,\n" % (weight_name))
            fd.write("\t\ts_%s,\n" % (output_name))
            fd.write("\t\ts_%s\n" % (skip_name))
            fd.write("\t);\n")

        fd.write("\n")

    def write_body(fd, model):

        for node in model.graph.node:

            if 'conv' in node.op_type.lower():
                write_conv(fd, node)
                continue

            if 'add' == node.op_type.lower():
                # write_add(fd, node)
                continue

            # TODO: Write Relu and thinks about folding, if the buffer is small
            # then there is a good chance the overhead is negligible

            if 'relu' == node.op_type.lower():
                write_relu(fd, node)
                continue

            if 'pool' in node.op_type.lower():
                if 'average' in node.op_type.lower():
                    write_average_pool(fd, node)
                continue

            if 'pad' in node.op_type.lower():
                write_pad(fd, node)

    def write_footer(fd):

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
            fd.write("\t\to_data\n")
            fd.write("\t);\n")

            # End of main file
            fd.write("}\n")

    with open("src/Network.cpp", "w+") as fd:

        write_header(fd)

        write_body(fd, model)

        write_footer(fd)

