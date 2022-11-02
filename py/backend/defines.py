import os
import sys
import onnx
from onnx import numpy_helper

def write(
    model,
    tensors_info,
    weights_info,
    skip_connections_info,
    bias_info,
    relu_info,
    conv_relu,
    flatten_info,
    split_info
):

    def write_header(fd):

        # Write header with network definitions
        fd.write("#ifndef __NETWORK__\n")
        fd.write("#define __NETWORK__\n")
        fd.write("#include \"ap_axi_sdata.h\"\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("#include \"ap_int.h\"\n")
        fd.write("#include <stdint.h>\n")

        # Handle internal or external parameters
        fd.write("#define c_i_data 64\n")
        fd.write("typedef ap_axiu<c_i_data, 0, 0, 0> t_i_data;\n")
        # fd.write("typedef int8_t t_weight;\n")
        fd.write("#define c_o_data 8\n")
        fd.write("typedef ap_axiu<c_o_data, 0, 0, 0> t_o_data;\n")

        fd.write("typedef ap_uint<1> t_last;\n")

        # Removing dots from input names
        for input in model.graph.input:
            input_name = input.name.replace(".", "_")
            input_name = input_name.lower().replace("onnx::", "")
            input_shape = tensors_info[input.name].tensor_type.shape

            fd.write("typedef uint8_t t_%s;\n" % (input_name))

            fd.write("\n")

            c_ich    = getattr(input_shape, 'dim')[1].dim_value
            c_ih     = getattr(input_shape, 'dim')[2].dim_value
            c_iw     = getattr(input_shape, 'dim')[3].dim_value

            fd.write("const int c_%s_ich    = %d;\n" % (input_name, c_ich))
            fd.write("const int c_%s_ih     = %d;\n" % (input_name, c_ih))
            fd.write("const int c_%s_iw     = %d;\n" % (input_name, c_iw))

            fd.write("\n")

        for output in model.graph.output:
            output_name = output.name.replace(".", "_")
            output_name = output_name.lower().replace("onnx::", "")
            output_shape = tensors_info[output.name].tensor_type.shape

            # TYPEDEF already performed during layer parsing
            # fd.write("typedef uint8_t t_%s;\n" % (output_name))

            fd.write("\n")

            c_och    = getattr(output_shape, 'dim')[1].dim_value
            if len(getattr(output_shape, 'dim')) > 2:
                c_oh     = getattr(output_shape, 'dim')[2].dim_value
                c_ow     = getattr(output_shape, 'dim')[3].dim_value
            else:
                c_oh     = 1
                c_ow     = 1

            fd.write("const int c_%s_och = %d;\n" % (output_name, c_och))
            fd.write("const int c_%s_oh  = %d;\n" % (output_name, c_oh))
            fd.write("const int c_%s_ow  = %d;\n" % (output_name, c_ow))

            fd.write("\n")

    def write_weights(weight_shape, weight_name):

        c_och    = getattr(weight_shape, 'dims')[0]
        c_ich    = getattr(weight_shape, 'dims')[1]
        if (len(getattr(weight_shape, 'dims')) > 2):
            c_ih     = getattr(weight_shape, 'dims')[2]
            c_iw     = getattr(weight_shape, 'dims')[3]
        else:
            c_ih     = 1
            c_iw     = 1

        fd.write("typedef uint8_t t_%s_st;\n" % (weight_name))
        fd.write("typedef ap_uint<8> t_%s;\n" % (weight_name))
        fd.write("const int c_%s_och = %d;\n" % (weight_name, c_och))
        fd.write("const int c_%s_ich = %d;\n" % (weight_name, c_ich))
        fd.write("const int c_%s_ih  = %d;\n" % (weight_name, c_ih))
        fd.write("const int c_%s_iw  = %d;\n" % (weight_name, c_iw))
        # fd.write("const ap_uint<8> c_%s_st[] = {\n" % (weight_name))
        
        # weights = numpy_helper.to_array(
        #     weight_shape
        # )

        # # TODO: handle weights quantization
        # last_weight = True
        # for och in range(weights.shape[0]):
        #     for ich in range(weights.shape[1]):
        #         for ih in range(weights.shape[2]):
        #             for iw in range(weights.shape[3]):
        #                 fd.write("%0.3f" % (weights[och][ich][ih][iw]))
        #                 fd.write(", ")

        # fd.write("0")

        # fd.write("};\n")
        fd.write("\n")

    def write_relu(fd, node):

        node_name = node.name.replace(".", "_").lower()

        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        fd.write("\n")

        fd.write("typedef ap_uint<8> t_%s;\n" % (output_name))

        c_ich = getattr(output_shape, 'dim')[1].dim_value
        c_ih  = getattr(output_shape, 'dim')[2].dim_value
        c_iw  = getattr(output_shape, 'dim')[3].dim_value

        fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
        fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
        fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))

        fd.write("\n")

    def write_add(fd, node):

        node_name = node.name.replace(".", "_").lower()

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        fd.write("\n")

        fd.write("typedef ap_uint<8> t_%s;\n" % (output_name))

        c_ich = getattr(output_shape, 'dim')[1].dim_value
        c_ih  = getattr(output_shape, 'dim')[2].dim_value
        c_iw  = getattr(output_shape, 'dim')[3].dim_value

        fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
        fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
        fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))

        fd.write("\n")

    def write_pool(
        fd,
        node,
        c_pool=0
    ):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")
        input_shape = tensors_info[node.input[0]].tensor_type.shape

        output_name = node.output[0].replace(".", "_")
        if output_name in flatten_info.keys():
            output_name = flatten_info[output_name][1]

        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        fd.write("\n")

        fd.write("typedef ap_uint<8> t_%s;\n" % (output_name))
        fd.write("typedef ap_uint<8> t_%s_acc;\n" % (node_name))

        attributes = getattr(node, "attribute" )

        c_ich    = getattr(input_shape, 'dim')[1].dim_value
        c_ih     = getattr(input_shape, 'dim')[2].dim_value
        c_iw     = getattr(input_shape, 'dim')[3].dim_value
        c_och    = getattr(output_shape, 'dim')[1].dim_value
        c_oh     = getattr(output_shape, 'dim')[2].dim_value
        c_ow     = getattr(output_shape, 'dim')[3].dim_value
        if ('adaptive' in node_name):
            c_fh     = getattr(attributes[1], 'ints')[0]
            c_fw     = getattr(attributes[1], 'ints')[1]
            c_stride = getattr(attributes[3], 'ints')[0]
            c_pad    = getattr(attributes[2], 'ints')[0]
        else:
            c_fh     = c_oh
            c_fw     = c_ow
            c_stride = 1
            c_pad    = 0

        fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
        fd.write("const int c_%s_och    = %d;\n" % (node_name, c_och))
        fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
        fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))
        fd.write("const int c_%s_oh     = %d;\n" % (node_name, c_oh))
        fd.write("const int c_%s_ow     = %d;\n" % (node_name, c_ow))
        fd.write("const int c_%s_fh     = %d;\n" % (node_name, c_fh))
        fd.write("const int c_%s_fw     = %d;\n" % (node_name, c_fw))
        fd.write("const int c_%s_stride = %d;\n" % (node_name, c_stride))
        fd.write("const int c_%s_pad    = %d;\n" % (node_name, c_pad))
        fd.write("const int c_%s_pool   = %d;\n" % (node_name, c_pool))

        fd.write("\n")

    def write_pad(fd, node):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")
        input_shape = tensors_info[node.input[0]].tensor_type.shape

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        fd.write("\n")

        fd.write("typedef ap_uint<8> t_%s;\n" % (output_name))
        fd.write("typedef ap_uint<8> t_%s_acc;\n" % (node_name))

        attributes = getattr(node, "attribute" )

        c_ich    = getattr(input_shape, 'dim')[1].dim_value
        c_ih     = getattr(input_shape, 'dim')[2].dim_value
        c_iw     = getattr(input_shape, 'dim')[3].dim_value
        c_och    = getattr(output_shape, 'dim')[1].dim_value
        c_oh     = getattr(output_shape, 'dim')[2].dim_value
        c_ow     = getattr(output_shape, 'dim')[3].dim_value
        # c_pad    = getattr(attributes[1], 'ints')[0]

        fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
        fd.write("const int c_%s_och    = %d;\n" % (node_name, c_och))
        fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
        fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))
        fd.write("const int c_%s_oh     = %d;\n" % (node_name, c_oh))
        fd.write("const int c_%s_ow     = %d;\n" % (node_name, c_ow))
        fd.write("const int c_%s_pad    = %d;\n" % (node_name, 0))

        fd.write("\n")

    def write_conv(fd, node, gemm=None):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")
        input_shape = tensors_info[node.input[0]].tensor_type.shape

        weight_name = node.input[1].replace(".", "_")
        weight_name = weight_name.lower().replace("onnx::", "")
        weight_shape = weights_info[node.input[1]]

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) > 1):
                skip_name = skip_connections_info[node.name][1].replace(".", "_")
                skip_name = skip_name.lower().replace("onnx::", "")

                # Declaring copied stream
                fd.write("typedef ap_uint<8> t_%s;\n" % (skip_name))
                fd.write("\n")

        output_name = node.output[0]
        if output_name in bias_info.keys():
            bias = True
            bias_name = bias_info[output_name][0]
            bias_name = bias_name.replace(".", "_")
            bias_name = bias_name.lower().replace("onnx::", "")

            output_name = bias_info[output_name][1]

        # Merging RELU to conv
        if output_name in relu_info.keys():
            output_name = relu_info[output_name][1]

        # Bypassing flatten
        if output_name in flatten_info.keys():
            output_name = flatten_info[output_name][1]

        if output_name in split_info.keys():
            c_split = len(split_info[output_name])
        else:
            c_split = 0

        output_name = output_name.replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        fd.write("\n")

        write_weights(weight_shape, weight_name)

        attributes = getattr(node, "attribute" )

        c_ich     = getattr(input_shape, 'dim')[1].dim_value
        # TODO: Generalize the case to not 1, 1 input features w, h
        if gemm is None:
            c_ih      = getattr(input_shape, 'dim')[2].dim_value
            c_iw      = getattr(input_shape, 'dim')[3].dim_value
        else:
            c_ih      = 1
            c_iw      = 1

        c_och     = getattr(output_shape, 'dim')[1].dim_value
        if gemm is None:
            c_oh      = getattr(output_shape, 'dim')[2].dim_value
            c_ow      = getattr(output_shape, 'dim')[3].dim_value
        else:
            c_oh      = 1
            c_ow      = 1

        if gemm is None:
            c_fh      = getattr(attributes[2], 'ints')[0]
            c_fw      = getattr(attributes[2], 'ints')[1]
            c_stride  = getattr(attributes[4], 'ints')[0]
            c_pad     = getattr(attributes[3], 'ints')[0]
        else:
            c_fh      = 1
            c_fw      = 1
            c_stride  = 1
            c_pad     = 0

        c_l_split = c_fh*c_fw+1
        if node.name in conv_relu:
            c_relu = 1
        else:
            c_relu = 0

        fd.write("typedef ap_uint<8> t_%s;\n" % (output_name))
        fd.write("typedef ap_uint<32> t_%s_acc;\n" % (node_name))
        fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
        fd.write("const int c_%s_och    = %d;\n" % (node_name, c_och))
        fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
        fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))
        fd.write("const int c_%s_ow     = %d;\n" % (node_name, c_ow))
        fd.write("const int c_%s_oh     = %d;\n" % (node_name, c_oh))
        fd.write("const int c_%s_fw     = %d;\n" % (node_name, c_fw))
        fd.write("const int c_%s_fh     = %d;\n" % (node_name, c_fh))
        fd.write("const int c_%s_relu   = %d;\n" % (node_name, c_relu))
        fd.write("const int c_%s_split  = %d;\n" % (output_name, c_split))
        fd.write("const int c_%s_stride = %d;\n" % (node_name, c_stride))
        fd.write("const int c_%s_pad    = %d;\n" % (node_name, c_pad))
        fd.write("const int c_%s_split  = %d;\n" % (node_name, c_l_split))

        fd.write("\n")

    def write_body(fd, model):

        for node in model.graph.node:

            if 'gemm' in node.op_type.lower():
                write_conv(fd, node, gemm=True)
                continue

            if 'conv' in node.op_type.lower():
                write_conv(fd, node)
                continue

            if 'add' == node.op_type.lower():
                write_add(fd, node)
                continue

            # TODO: Write Relu and thinks about folding, if the buffer is small
            # then there is a good chance the overhead is negligible

            if 'relu' == node.op_type.lower():
                write_relu(fd, node)
                continue

            if 'pool' in node.op_type.lower():
                c_pool = 0
                if 'average' in node.op_type.lower():
                    c_pool = 0

                if 'max' in node.op_type.lower():
                    c_pool = 1

                write_pool(fd, node, c_pool)
                continue

            if 'pad' in node.op_type.lower():
                write_pad(fd, node)

    def write_footer(fd):

        # Adding prototype declaration
        fd.write("void Network(\n")
        fd.write("\thls::stream<t_i_data> &i_data,\n")
        fd.write("\thls::stream<t_o_data> &o_data\n")
        fd.write(");\n")

        # End of main file
        fd.write("#endif")

    with open("src/Network.hpp", "w+") as fd:

        write_header(fd)

        write_body(fd, model)

        write_footer(fd)

