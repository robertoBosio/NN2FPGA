import os
import sys
import onnx
from onnx import numpy_helper
import numpy as np

class layer():
    def __init__(
        self,
        node,
        split_info,
        relu_info,
        skip_connections_info
    ):
        super(conv, self).__init__()
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
            # If it is less than 2 it means is a producer
            if (len(skip_connections_info[node.name]) < 2):
                self.skip_name = skip_connections_info[node.name][0].replace(".", "_")
                self.skip_name = self.skip_name.lower().replace("onnx::", "")

                # Checking to which stream is directed the skip connection
                if (self.skip_name.replace("_skip", "") == self.input_name):
                    self.input_name = self.skip_name

        if (node.name in skip_connections_info.keys()):
            # If it is greater than 1 it means is a consumer
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

        if node.name in conv_relu:
            self.c_relu = 1
        else:
            self.c_relu = 0


