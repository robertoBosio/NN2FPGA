import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def return_index(fh, fw, ws, iw, ih):
    base_index = ih*(fw+ws-1)+iw
    if (iw >= (fw-1)) and (iw < (ws)):
        return base_index+fw+ws-1
    elif iw < (fw):
        return base_index+ws
    else:
        return base_index+fw-1

def parse(name, node):

    ws = node["ws"]
    dfh = node["fh"]
    dfw = node["fw"] + (ws-1)
    dindex = dfh*dfw

    line_buffer_blocks = []


    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = input_name
    output_type_name = output_name.replace("_skip", "")

    for fh in range(dfh):
        for fw in range(dfw):
            
            index = fh*dfw+fw
            out_index = return_index(node["fh"], node["fw"], ws, fw, fh)
            print("index: %0d" % index, "out_index: %0d" % out_index, "fh: %0d" % fh, "fw: %0d" % fw)

            block = {}
            block["func"] = "shift_op"

            # Template parameters
            block["template"] = []
            block["template"].append("t_%s_struct" % input_type_name)
            if out_index < (dindex):
                block["template"].append("t_%s_struct" % input_type_name)
            else:
                block["template"].append("std::nullptr_t")
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
            block["template"].append("c_%s_ws" % name)

            block["args"] = []

            if index < ws:
                block["args"].append("s_%s[%0d]" % (input_name, ws-index-1))
            else:
                block["args"].append(
                    "s_%s_data[%0d]" % (input_name, index-ws)
                )

            block["args"].append(
                "s_%s_pre_pad[%0d]" % (input_name, index)
            )

            if out_index < (dindex):
                block["args"].append(
                    "s_%s_data[%0d]" % (input_name, out_index-ws)
                )
            else:
                block["args"].append(
                    "s_%s_null[%0d]" % (input_name, dindex-index-1)
                )

            block["declare"] = []

            if (index == 0):
                declare = {}
                declare["name"] = "s_%s_data" % output_name
                declare["type"] = "t_%s_struct" % output_name
                declare["is_array"] = True
                declare["dim"] = dindex-ws
                block["declare"].append(declare)

                declare = {}
                declare["name"] = "s_%s_null" % output_name
                declare["type"] = "std::nullptr_t"
                declare["is_array"] = True
                declare["dim"] = ws
                block["declare"].append(declare)

                declare = {}
                declare["name"] = "s_%s_pre_pad" % output_name
                declare["type"] = "t_%s_struct" % output_name
                declare["is_array"] = True
                declare["dim"] = dindex
                block["declare"].append(declare)

            block["pragma"] = []

            output_ratio = int((node["ih"]*node["iw"])/(node["oh"]*node["ow"]))
            if output_ratio > 1: 
                depth = node["och"]*output_ratio+1
                # impl = "BRAM"
                impl = "AUTO"
            else:
                depth = node["fh"]*node["fw"]+1
                impl = "AUTO"

            if (index == 0):
                pragma = {}
                pragma["name"] = "stream"
                options = [
                    ["variable", "s_%s_pre_pad" % output_name],
                    ["depth", depth],
                    # ["depth", "2"],
                    ["type", "fifo"],
                ]
                pragma["options"] = options
                block["pragma"].append(pragma)

                pragma = {}
                pragma["name"] = "bind_storage"
                options = [
                    ["variable", "s_%s_pre_pad" % output_name],
                    # ["depth", "2"],
                    ["impl", impl],
                    ["type", "fifo"],
                ]
                pragma["options"] = options
                block["pragma"].append(pragma)

            pragma = {}

            if (fw == (dfw-1)):
                depth = node["iw"]*node["ich"]
            else:
                depth = node["ich"]

            if not((fh == (dfh-1)) and (fw == (dfw-1))):

                pragma["name"] = "stream"
                options = [
                    ["variable", "s_%s_data[%0d]" % (output_name, index)],
                    ["depth", depth],
                    ["type", "fifo"],
                ]
                pragma["options"] = options

                block["pragma"].append(pragma)

                if (fh == 0) and (fw == 0):
                    # TODO: Check correcteness of variable selection
                    pragma = {}
                    pragma["name"] = "bind_storage"
                    options = [
                        ["variable", "s_%s_data[0]" % output_name],
                        # ["depth", "2"],
                        # ["impl", "BRAM"],
                        ["impl", "AUTO"],
                        ["type", "fifo"],
                    ]
                    pragma["options"] = options
                    block["pragma"].append(pragma)


            line_buffer_blocks.append(block)

    return line_buffer_blocks

