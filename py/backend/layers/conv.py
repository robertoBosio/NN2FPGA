import os
import sys
import onnx
from onnx import numpy_helper
import numpy as np

from backend.layers.utils import write_array_stream, write_last_flags
from backend.layers.layer import layer

class conv2d(layer):
    def __init__(
        self,
        node,
        split_info,
        relu_info,
        skip_connections_info,
        weights_info,
        tensors_info,
        gemm
    ):
        super(conv, self).__init__(
        )
        self.node = node
        self.node_name = node.name.replace(".", "_").lower()

        # Attributes for weights generation
        self.attributes = getattr(node, "attribute" )

        # Assuming no skip connection at the start
        self.no_skip = True
        self.bias = False
        self.split = False
        # Removing dots from input names
        self.input_name = node.input[0]

        self.indexed = False
        if (self.input_name in split_info.keys()):
            self.indexed = True
            self.index = split_info[self.input_name].index(node.name)

        self.input_name = self.input_name.replace(".", "_")
        self.input_name = self.input_name.lower().replace("onnx::", "")
        self.input_shape = tensors_info[node.input[0]].tensor_type.shape

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) < 2):
                self.skip_name = skip_connections_info[node.name][0].replace(".", "_")
                self.skip_name = self.skip_name.lower().replace("onnx::", "")

                # Checking to which stream is directed the skip connection
                if (self.skip_name.replace("_skip", "") == self.input_name):
                    self.input_name = self.skip_name

        self.weight_name = node.input[1].replace(".", "_")
        self.weight_name = self.weight_name.lower().replace("onnx::", "")
        self.weight_shape = weights_info[node.input[1]]

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 2 it means is a producer
            if (len(skip_connections_info[node.name]) > 1):
                self.skip_name = skip_connections_info[node.name][1].replace(".", "_")
                self.skip_name = self.skip_name.lower().replace("onnx::", "")
                self.no_skip = False

                # Declaring copied stream

        # Adding BIAS and merging to add layer
        self.output_name = node.output[0]
        if self.output_name in self.bias_info.keys():
            self.bias = True
            self.bias_name = self.bias_info[self.output_name][0]
            self.bias_name = self.bias_name.replace(".", "_")
            self.bias_name = self.bias_name.lower().replace("onnx::", "")

            self.output_name = self.bias_info[self.output_name][1]

        # Merging RELU to conv
        if self.output_name in relu_info.keys():
            self.replaced_relu = relu_info[self.output_name][0]
            self.conv_relu = node.name
            self.output_name = relu_info[self.output_name][1]

        # Bypassing flatten
        if self.output_name in flatten_info.keys():
            self.output_name = flatten_info[self.output_name][1]

        if self.output_name in split_info.keys():
            self.split = True

        self.output_name = self.output_name.replace(".", "_")
        self.output_name = self.output_name.lower().replace("onnx::", "")
        self.output_shape = tensors_info[node.output[0]].tensor_type.shape

        attributes = getattr(node, "attribute" )

        self.c_ich     = getattr(self.input_shape, 'dim')[1].dim_value
        # TODO: Generalize the case to not 1, 1 input features w, h
        if self.gemm is None:
            self.c_ih      = getattr(self.input_shape, 'dim')[2].dim_value
            self.c_iw      = getattr(self.input_shape, 'dim')[3].dim_value
        else:
            self.c_ih      = 1
            self.c_iw      = 1

        self.c_och     = getattr(self.output_shape, 'dim')[1].dim_value
        if self.gemm is None:
            self.c_oh      = getattr(self.output_shape, 'dim')[2].dim_value
            self.c_ow      = getattr(self.output_shape, 'dim')[3].dim_value
        else:
            self.c_oh      = 1
            self.c_ow      = 1

        if self.gemm is None:
            self.c_fh      = getattr(self.attributes[2], 'ints')[0]
            self.c_fw      = getattr(self.attributes[2], 'ints')[1]
            self.c_stride  = getattr(self.attributes[4], 'ints')[0]
            self.c_pad     = getattr(self.attributes[3], 'ints')[0]
        else:
            self.c_fh      = 1
            self.c_fw      = 1
            self.c_stride  = 1
            self.c_pad     = 0

        self.c_l_split = self.c_fh*self.c_fw+1
        if node.name in conv_relu:
            self.c_relu = 1
        else:
            self.c_relu = 0


    def write_streams(self, fd):

        fd.write("\n")

        fd.write(
            "\thls::stream<ap_uint<1>> s_last_%s[c_%s_split];\n" % (
                self.node_name,
                self.node_name
            )
        )
        fd.write(
            "\t#pragma HLS STREAM variable=s_last_%s depth=10 type=fifo\n" % (
                self.node_name
            )
        )

        fd.write("\n")

        write_weights(
            self.weight_shape,
            self.weight_name
        )

        fd.write("\n")

        write_stream(
            fd,
            self.self.skip_name,
            "c_%s_ich*c_%s_och" % (self.node_name, self.node_name)
        )

        if (self.split):
            write_array_stream(
                fd,
                self.output_name,
                "c_%s_ich*c_%s_och" % (self.node_name, self.node_name),
                2
            )
        else:
            write_stream(
                fd,
                self.output_name,
                "c_%s_ich*c_%s_och" % (self.node_name, self.node_name),
            )

        fd.write("\n")

        write_array_stream(fd, self.weight_name, "c_%s_och" % self.weight_name)

        fd.write("\n")


    def write_block(self, fd, last_flag):

        for ih in range(self.c_ih):
            for iw in range(self.c_iw):
                fd.write("\tProduceStream<\n")
                fd.write("\t\tt_%s_st,\n" % (self.weight_name))
                fd.write("\t\tt_%s,\n" % (self.weight_name))
                fd.write("\t\tc_%s_ich,\n" % (self.weight_name))
                fd.write("\t\tc_%s_och,\n" % (self.weight_name))
                fd.write("\t\tc_%s_ow,\n" % (self.node_name))
                fd.write("\t\tc_%s_oh\n" % (self.node_name))
                fd.write("\t>(\n")
                fd.write("\t\tc_%s_st_%0d,\n" % (self.weight_name, ih*self.c_iw+iw))
                fd.write("\t\ts_last_%s[%0d],\n" % (
                        self.node_name, ih*self.c_iw+iw
                    )
                )
                fd.write("\t\ts_%s[%0d]\n" % (self.weight_name, ih*self.c_iw+iw))
                fd.write("\t);\n")

                fd.write("\n")

        fd.write("\tSplitStream<\n")
        fd.write("\t\tc_%s_split\n" % (self.node_name))
        fd.write("\t>(\n")
        fd.write("\t\ts_last_split[%0d],\n" % (last_flag))
        fd.write("\t\ts_last_%s\n" % (self.node_name))
        fd.write("\t);\n")
        fd.write("\n")

        if (self.no_skip):
            fd.write("\tPackedConvBuffAcc<\n")
            fd.write("\t\tt_%s,\n" % (self.input_name))
            fd.write("\t\tt_%s,\n" % (self.weight_name))
            fd.write("\t\tt_%s,\n" % (self.output_name))
            if (self.bias):
                fd.write("\t\tt_%s,\n" % (self.input_name))
            fd.write("\t\tt_%s_acc,\n" % (self.node_name))
            fd.write("\t\tc_%s_ich,\n" % (self.node_name))
            fd.write("\t\tc_%s_och,\n" % (self.node_name))
            fd.write("\t\tc_%s_iw,\n" % (self.node_name))
            fd.write("\t\tc_%s_ih,\n" % (self.node_name))
            fd.write("\t\tc_%s_ow,\n" % (self.node_name))
            fd.write("\t\tc_%s_oh,\n" % (self.node_name))
            fd.write("\t\tc_%s_fw,\n" % (self.node_name))
            fd.write("\t\tc_%s_fh,\n" % (self.node_name))
            fd.write("\t\tc_%s_relu,\n" % (self.node_name))
            if (self.split):
                fd.write("\t\tc_%s_split,\n" % (self.output_name))
            fd.write("\t\tc_%s_stride,\n" % (self.node_name))
            fd.write("\t\tc_%s_pad\n" % (self.node_name))
            fd.write("\t> (\n")
            if indexed:
                fd.write("\t\ts_%s[%d],\n" % (self.input_name, index))
            else:
                fd.write("\t\ts_%s,\n" % (self.input_name))
            fd.write("\t\ts_%s,\n" % (self.weight_name))
            if (self.bias):
                fd.write("\t\ts_%s,\n" % (self.bias_name))
            fd.write("\t\ts_last_%s[c_%s_fh*c_%s_fw],\n" % (self.node_name, self.node_name, self.node_name))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag + 1))
            fd.write("\t\ts_%s\n" % (self.output_name))
            fd.write("\t);\n")
        else:
            fd.write("\tPackedConvBuffAcc<\n")
            fd.write("\t\tt_%s,\n" % (self.input_name))
            fd.write("\t\tt_%s,\n" % (self.weight_name))
            fd.write("\t\tt_%s,\n" % (self.output_name))
            fd.write("\t\tt_%s_acc,\n" % (self.node_name))
            fd.write("\t\tc_%s_ich,\n" % (self.node_name))
            fd.write("\t\tc_%s_och,\n" % (self.node_name))
            fd.write("\t\tc_%s_iw,\n" % (self.node_name))
            fd.write("\t\tc_%s_ih,\n" % (self.node_name))
            fd.write("\t\tc_%s_ow,\n" % (self.node_name))
            fd.write("\t\tc_%s_oh,\n" % (self.node_name))
            fd.write("\t\tc_%s_fw,\n" % (self.node_name))
            fd.write("\t\tc_%s_fh,\n" % (self.node_name))
            fd.write("\t\tc_%s_relu,\n" % (self.node_name))
            if (self.split):
                fd.write("\t\tc_%s_split,\n" % (self.output_name))
            fd.write("\t\tc_%s_stride,\n" % (self.node_name))
            fd.write("\t\tc_%s_pad\n" % (self.node_name))
            fd.write("\t> (\n")
            if indexed:
                fd.write("\t\ts_%s[%d],\n" % (self.input_name, index))
            else:
                fd.write("\t\ts_%s,\n" % (self.input_name))
            fd.write("\t\ts_%s,\n" % (self.weight_name))
            fd.write("\t\ts_last_%s[c_%s_fh*c_%s_fw],\n" % (self.node_name, self.node_name, self.node_name))
            fd.write("\t\ts_last_split[%0d],\n" % (last_flag + 1))
            fd.write("\t\ts_%s,\n" % (self.output_name))
            fd.write("\t\ts_%s\n" % (self.skip_name))
            fd.write("\t);\n")

        fd.write("\n")


    def write_define(self):

        fd.write("typedef ap_uint<8> t_%s;\n" % (self.output_name))
        fd.write("typedef ap_uint<32> t_%s_acc;\n" % (self.node_name))
        fd.write("const int c_%s_ich    = %d;\n" % (self.node_name, c_ich))
        fd.write("const int c_%s_och    = %d;\n" % (self.node_name, c_och))
        fd.write("const int c_%s_ih     = %d;\n" % (self.node_name, c_ih))
        fd.write("const int c_%s_iw     = %d;\n" % (self.node_name, c_iw))
        fd.write("const int c_%s_ow     = %d;\n" % (self.node_name, c_ow))
        fd.write("const int c_%s_oh     = %d;\n" % (self.node_name, c_oh))
        fd.write("const int c_%s_fw     = %d;\n" % (self.node_name, c_fw))
        fd.write("const int c_%s_fh     = %d;\n" % (self.node_name, c_fh))
        fd.write("const int c_%s_relu   = %d;\n" % (self.node_name, c_relu))
        fd.write("const int c_%s_split  = %d;\n" % (self.output_name, c_split))
        fd.write("const int c_%s_stride = %d;\n" % (self.node_name, c_stride))
        fd.write("const int c_%s_pad    = %d;\n" % (self.node_name, c_pad))
        fd.write("const int c_%s_split  = %d;\n" % (self.node_name, c_l_split))

        fd.write("\n")

    def get_layer_ilp_info(self):
        layers_info.append([self.node_name, 1/(self.c_oh*self.c_ow*self.c_och*self.c_ich*self.c_fh*self.c_fw)])

