import os
import sys
import onnx
from onnx import numpy_helper
import numpy

def write(
    model,
    tensors_info,
    quant_info,
    weights_info,
    skip_connections_info,
    bias_info,
    relu_info,
    conv_relu,
    flatten_info,
    split_info,
    off_chip_storage,
    additional_ports,
    parallel_ops,
    read_width,
    reuse, 
    prj_root="/tmp"
):

    def write_header(fd):

        # Write header with network definitions
        fd.write("#ifndef __NETWORK__\n")
        fd.write("#define __NETWORK__\n")
        fd.write("#include \"ap_axi_sdata.h\"\n")
        fd.write("#include \"hls_stream.h\"\n")
        fd.write("#include \"ap_int.h\"\n")
        fd.write("#include \"hls_vector.h\"\n")
        fd.write("#include <stdint.h>\n")
        fd.write("#include \"hls_burst_maxi.h\"\n")

        fd.write("#define READ_WIDTH %0d\n" % read_width)
        # Handle internal or external parameters
        fd.write("#define c_i_data 64\n")
        fd.write("typedef ap_axiu<c_i_data, 0, 0, 0> t_i_data;\n")
        # fd.write("typedef int8_t t_weight;\n")
        fd.write("#define c_o_data 32\n")
        fd.write("typedef ap_axis<c_o_data, 0, 0, 0> t_o_data;\n")

        fd.write("typedef ap_uint<1> t_last;\n")

        # Removing dots from input names
        for input in model.graph.input:
            input_name = input.name.replace(".", "_")
            input_name = input_name.lower().replace("onnx::", "")
            input_shape = tensors_info[input.name].tensor_type.shape

            fd.write("typedef uint8_t t_%s;\n" % (input_name))

            fd.write("typedef struct {\n")
            fd.write("\tt_%s data;\n" % (input_name))
            fd.write("\tbool last;\n")
            fd.write("} t_%s_struct;\n" % (input_name))

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
            # fd.write("typedef int8_t t_%s;\n" % (output_name))

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

    def write_weights(weight_shape, weight_name, node_name, write_file=False):
        
        
        keys = list(weight_shape.keys())
        #print(keys)
        weight_shape_n = numpy_helper.to_array(weight_shape[keys[1]]).shape
        scale_factor = numpy_helper.to_array(weight_shape[keys[2]])
        scale_factor_shift = numpy.log2(scale_factor)
        c_och    = weight_shape_n[0]
        c_ich    = weight_shape_n[1]
        if (len(weight_shape_n) > 2):
            c_ih     = weight_shape_n[2]
            c_iw     = weight_shape_n[3]
        else:
            c_ih     = 1
            c_iw     = 1

        if (write_file):
            
            fd.write("const int c_%s_ops  = %d;\n" % (node_name, parallel_ops[node_name]))
            fd.write("const int c_%s_bw   = 4;\n" % (weight_name))
            fd.write("const int c_%s_reuse = %0d;\n" % (weight_name, reuse[node_name]))

            fd.write("\n")

            n_bits = 8*parallel_ops[node_name]

            # if (off_chip_storage or (n_bits > 64)):
            #     fd.write("typedef ap_uint<%0d> t_%s_st;\n" % (n_bits, weight_name))
            # else:
            #     fd.write("typedef uint%0d_t t_%s_st;\n" % (n_bits, weight_name))
            # fd.write("typedef ap_uint<8*c_%s_ops> t_%s;\n" % (node_name, weight_name))
            fd.write("typedef int8_t t_%s_st;\n" % (weight_name))
            # fd.write("typedef int8_t t_%s;\n" % (weight_name))

            # Using the hls::vector class to handle SIMD operations

            # fd.write(
            #     "typedef hls::vector<int8_t, c_%s_ops> t_%s_st;\n" % (
            #         node_name,
            #         weight_name
            #     )
            # )

            fd.write(
                "typedef hls::vector<int8_t, c_%s_ops> t_%s;\n" % (
                    node_name,
                    weight_name
                )
            )

            fd.write("const int c_%s_och = %d;\n" % (weight_name, c_och))
            fd.write("const int c_%s_ich = %d;\n" % (weight_name, c_ich))
            fd.write("const int c_%s_ih  = %d;\n" % (weight_name, c_ih))
            fd.write("const int c_%s_iw  = %d;\n" % (weight_name, c_iw))
            fd.write("const int c_%s_ops = %0d;\n" % (weight_name, parallel_ops[node_name]))
            fd.write("const int c_%s_index = %0d;\n" % (weight_name, c_ih*c_iw)) 
            fd.write("const int c_%s_iter  = %0d;\n" % (weight_name, c_och*c_ich/parallel_ops[node_name] + 1))
            fd.write("const float c_%s_scale = %f;\n" % (weight_name, scale_factor))
            fd.write("const int c_%s_scale_shift = %d;\n" %  (weight_name, scale_factor_shift))
            fd.write("\n")

    def write_relu(fd, node, write_file=False):

        node_name = node.name.replace(".", "_").lower()

        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        c_ich = getattr(output_shape, 'dim')[1].dim_value
        c_ih  = getattr(output_shape, 'dim')[2].dim_value
        c_iw  = getattr(output_shape, 'dim')[3].dim_value

        if (write_file):
            fd.write("\n")

            # fd.write("typedef uint8_t t_%s;\n" % (output_name))

            fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
            fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
            fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))

            fd.write("\n")

    def write_add(fd, node, write_file=False):

        node_name = node.name.replace(".", "_").lower()

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        c_ich = getattr(output_shape, 'dim')[1].dim_value
        c_ih  = getattr(output_shape, 'dim')[2].dim_value
        c_iw  = getattr(output_shape, 'dim')[3].dim_value

        if (write_file):
            fd.write("\n")

            fd.write("typedef uint8_t t_%s;\n" % (output_name))

            fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
            fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
            fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))

            fd.write("\n")

    def write_pool(
        fd,
        node,
        c_pool=0,
        write_file=False
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

        attributes = getattr(node, "attribute" )

        c_ich    = getattr(input_shape, 'dim')[1].dim_value
        c_ih     = getattr(input_shape, 'dim')[2].dim_value
        c_iw     = getattr(input_shape, 'dim')[3].dim_value
        c_och    = getattr(output_shape, 'dim')[1].dim_value
        c_oh     = getattr(output_shape, 'dim')[2].dim_value
        c_ow     = getattr(output_shape, 'dim')[3].dim_value
        if ('adaptive' in node_name):
            c_fh     = c_oh
            c_fw     = c_ow
            c_stride = 1
            c_pad    = 0
        else:
            c_fh     = getattr(attributes[0], 'ints')[0]
            c_fw     = getattr(attributes[0], 'ints')[1]
            c_stride = getattr(attributes[2], 'ints')[0]
            c_pad    = getattr(attributes[1], 'ints')[0]

        if (write_file):
            fd.write("\n")

            fd.write("typedef uint8_t t_%s;\n" % (output_name))
            fd.write("typedef int32_t t_%s_acc;\n" % (node_name))

            fd.write("typedef struct {\n")
            fd.write("\tt_%s data;\n" % (output_name))
            fd.write("\tbool last;\n")
            fd.write("} t_%s_struct;\n" % (output_name))

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

    def write_pad(fd, node, write_file=False):

        node_name = node.name.replace(".", "_").lower()

        # Removing dots from input names
        input_name = node.input[0].replace(".", "_")
        input_name = input_name.lower().replace("onnx::", "")
        input_shape = tensors_info[node.input[0]].tensor_type.shape

        output_name = node.output[0].replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        attributes = getattr(node, "attribute" )

        c_ich    = getattr(input_shape, 'dim')[1].dim_value
        c_ih     = getattr(input_shape, 'dim')[2].dim_value
        c_iw     = getattr(input_shape, 'dim')[3].dim_value
        c_och    = getattr(output_shape, 'dim')[1].dim_value
        c_oh     = getattr(output_shape, 'dim')[2].dim_value
        c_ow     = getattr(output_shape, 'dim')[3].dim_value
        # c_pad    = getattr(attributes[1], 'ints')[0]

        if (write_file):
            fd.write("\n")

            fd.write("typedef uint8_t t_%s;\n" % (output_name))
            fd.write("typedef int8_t t_%s_acc;\n" % (node_name))

            fd.write("typedef struct {\n")
            fd.write("\tt_%s data;\n" % (output_name))
            fd.write("\tbool last;\n")
            fd.write("} t_%s_struct;\n" % (output_name))

            fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
            fd.write("const int c_%s_och    = %d;\n" % (node_name, c_och))
            fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
            fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))
            fd.write("const int c_%s_oh     = %d;\n" % (node_name, c_oh))
            fd.write("const int c_%s_ow     = %d;\n" % (node_name, c_ow))
            fd.write("const int c_%s_pad    = %d;\n" % (node_name, 0))

            fd.write("\n")

    def write_conv(fd, node, gemm=None, write_file=False):

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

                if (write_file):
                    # Declaring copied stream
                    fd.write("typedef uint8_t t_%s;\n" % (skip_name))
                    fd.write("\n")
                    fd.write("typedef struct {\n")
                    fd.write("\tt_%s data;\n" % (skip_name))
                    fd.write("\tbool last;\n")
                    fd.write("} t_%s_struct;\n" % (skip_name))
                    fd.write("typedef uint8_t t_%s;\n" % (skip_name))
                    fd.write("\n")

        output_name = node.output[0]
        if output_name in bias_info.keys():
            bias = True
            bias_name = bias_info[output_name][0]
            bias_name = bias_name.replace(".", "_")
            bias_name = bias_name.lower().replace("onnx::", "")

            output_name = bias_info[output_name][1]

        if (output_name in weights_info.keys()):
            activation_shape = weights_info[output_name]
            keys = list(activation_shape.keys())
            activation_scale = numpy_helper.to_array(activation_shape[keys[1]])
            activation_scale_shift = numpy.log2(activation_scale)

        # Merging RELU to conv
        if output_name in relu_info.keys():
            output_name = relu_info[output_name][1]

        if output_name in quant_info.keys():
            output_name = quant_info[output_name]

        # Bypassing flatten
        if output_name in flatten_info.keys():
            output_name = flatten_info[output_name][1]

        if output_name in split_info.keys():
            c_split = len(split_info[output_name])
        else:
            c_split = 0

        if output_name == model.graph.output[0].name:
            output_type = "int32_t"
        else:
            output_type = "uint8_t"

        output_name = output_name.replace(".", "_")
        output_name = output_name.lower().replace("onnx::", "")
        output_shape = tensors_info[node.output[0]].tensor_type.shape

        if (write_file):
            fd.write("\n")

        write_weights(weight_shape, weight_name, node_name, write_file)

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

        c_l_split = 2
        if node.name in conv_relu:
            c_relu = 1
        else:
            c_relu = 0

        if (write_file):
            fd.write("typedef %s t_%s;\n" % (output_type, output_name))
            fd.write("typedef struct {\n")
            fd.write("\tt_%s data;\n" % (output_name))
            fd.write("\tbool last;\n")
            fd.write("} t_%s_struct;\n" % (output_name))
            fd.write("typedef ap_int<32> t_%s_acc;\n" % (node_name))
            fd.write("const int c_%s_ich    = %d;\n" % (node_name, c_ich))
            fd.write("const int c_%s_och    = %d;\n" % (node_name, c_och))
            fd.write("const int c_%s_ih     = %d;\n" % (node_name, c_ih))
            fd.write("const int c_%s_iw     = %d;\n" % (node_name, c_iw))
            fd.write("const int c_%s_ow     = %d;\n" % (node_name, c_ow))
            fd.write("const int c_%s_oh     = %d;\n" % (node_name, c_oh))
            fd.write("const int c_%s_fw     = %d;\n" % (node_name, c_fw))
            fd.write("const int c_%s_fh     = %d;\n" % (node_name, c_fh))
            fd.write("const int c_%s_relu   = %d;\n" % (node_name, c_relu))
            fd.write("const int c_%s_a_split  = %d;\n" % (output_name, c_split))
            fd.write("const int c_%s_stride = %d;\n" % (node_name, c_stride))
            fd.write("const int c_%s_pad    = %d;\n" % (node_name, c_pad))
            if 'activation_scale' in locals():
                fd.write("const float c_%s_scale = %f;\n" % (node_name, activation_scale)) 
                fd.write("const int c_%s_scale_shift = %d;\n" % (node_name, activation_scale_shift)) 
                                                
            if (not off_chip_storage):
                fd.write("const int c_%s_split  = %d;\n" % (node_name, c_l_split))

            fd.write("\n")

    def write_body(fd, model, write_file=False):

        for node in model.graph.node:

            if 'gemm' in node.op_type.lower():
                write_conv(fd, node, gemm=True, write_file=write_file)
                continue

            if 'conv' in node.op_type.lower():
                write_conv(fd, node, write_file=write_file)
                continue

            if 'add' == node.op_type.lower():
                write_add(fd, node, write_file=write_file)
                continue

            # TODO: Write Relu and thinks about folding, if the buffer is small
            # then there is a good chance the overhead is negligible

            if 'relu' == node.op_type.lower():
                write_relu(fd, node, write_file=write_file)
                continue

            if 'pool' in node.op_type.lower():
                c_pool = 0
                if 'average' in node.op_type.lower():
                    c_pool = 0

                if 'max' in node.op_type.lower():
                    c_pool = 1

                write_pool(fd, node, c_pool, write_file=write_file)
                continue

            if 'pad' in node.op_type.lower():
                write_pad(fd, node, write_file=write_file)

    def write_footer(fd):

        # Adding prototype declaration
        fd.write("void network(\n")
        fd.write("\thls::stream<t_i_data> &i_data,\n")
        for i, name in enumerate(additional_ports):
            fd.write("\tap_int<READ_WIDTH> *i_data_%s,\n" % name)
        fd.write("\thls::stream<t_o_data> &o_data\n")
        fd.write(");\n")

        # End of main file
        fd.write("#endif")

    with open(prj_root + "/cc/include/network.h", "w+") as fd:

        write_header(fd)

        # print(weights_info.keys())
        write_body(fd, model, write_file=False)

        # TODO: Specialized for DAC2023 submission, must be automated

        write_body(fd, model, write_file=True)

        write_footer(fd)

