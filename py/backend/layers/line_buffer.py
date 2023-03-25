import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def parse(name, node):

    dfh = node["fh"]
    dfw = node["fw"]
    dindex = dfh*dfw

    line_buffer_blocks = []

    for fh in range(dfh):
        for fw in range(dfw):
            index = fh*dfw+fw
            input_name  = node["input"][0]
            input_type_name = input_name.replace("_skip", "")

            output_name = input_name
            output_type_name = output_name.replace("_skip", "")

            block = {}
            block["func"] = "ShiftOp"

            # Template parameters
            block["template"] = []
            block["template"].append("t_%s_struct" % input_type_name)
            block["template"].append("c_%s_ich" % name)
            block["template"].append("c_%s_och" % name)
            block["template"].append("c_%s_ih" % name)
            block["template"].append("c_%s_iw" % name)
            block["template"].append("c_%s_oh" % name)
            block["template"].append("c_%s_ow" % name)
            block["template"].append("c_%s_fh" % name)
            block["template"].append("c_%s_fw" % name)
            block["template"].append("c_%s_stride" % name)
            block["template"].append("c_%s_pad" % name)
            block["template"].append("%0d" % (dfh - 1 - fh))
            block["template"].append("%0d" % (dfw - 1 - fw))

            block["args"] = []

            if index == 0:
                block["args"].append("s_%s_padded" % input_name)
            else:
                block["args"].append(
                    "s_%s_data[%0d]" % (input_name, index-1)
                )

            block["args"].append(
                "s_%s_compute[%0d]" % (input_name, index)
            )

            if index < (dindex-1):
                block["args"].append(
                    "s_%s_data[%0d]" % (input_name, index)
                )

            block["declare"] = []

            if (index == 0):
                declare = {}
                declare["name"] = "s_%s_data" % output_name
                declare["type"] = "t_%s_struct" % output_name
                declare["is_array"] = True
                declare["dim"] = dindex-1
                block["declare"].append(declare)

                declare = {}
                declare["name"] = "s_%s_compute" % output_name
                declare["type"] = "t_%s_struct" % output_name
                declare["is_array"] = True
                declare["dim"] = dindex
                block["declare"].append(declare)

            block["pragma"] = []

            if (index == 0):
                pragma = {}
                pragma["name"] = "stream"
                options = [
                    ["variable", "s_%s_compute" % output_name],
                    ["depth", "2"],
                    ["type", "fifo"],
                ]
                pragma["options"] = options
                block["pragma"].append(pragma)

            pragma = {}

            if (index == 0):
                depth = 2
            elif (fw == 0):
                depth = node["iw"]*node["ich"]
            else:
                depth = node["ich"]

            pragma["name"] = "stream"
            options = [
                ["variable", "s_%s_data[%0d]" % (output_name, index)],
                ["depth", depth],
                ["type", "fifo"],
            ]
            pragma["options"] = options

            block["pragma"].append(pragma)

            line_buffer_blocks.append(block)

    return line_buffer_blocks

