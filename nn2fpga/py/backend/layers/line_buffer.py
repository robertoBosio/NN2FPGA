import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def return_index(fh, fw, ws, stride, iw, ih):
    base_index = ih*(fw+(ws-1)*stride)+iw
    if (iw >= (fw-1)) and (iw < (ws)):
        return (ih+1)*(fw+(ws-1)*stride)+iw
    elif iw < (fw):
        return base_index+ws
    else:
        return (ih+1)*(fw+(ws-1)*stride)+iw%ws
    print("ERROR: return_index")
    sys.exit(1)

def parse(name, node, debug=False):

    stride = node["stride"]
    ws = node["ws"]
    dfh = node["fh"]
    dfw = node["fw"] + (ws-1)*(stride)
    dindex = dfh*dfw

    line_buffer_blocks = []


    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = input_name
    output_type_name = output_name.replace("_skip", "")

    pad_value = int(node["pad"]*(node["fw"]-1)/2)
    for fh in range(dfh):
        for fw in range(dfw):
            
            index = fh*dfw+fw
            out_index = return_index(node["fh"], node["fw"], ws, stride, fw, fh)
            if debug:
                print("index: %0d" % index, "out_index: %0d" % out_index, "fh: %0d" % fh, "fw: %0d" % fw)

            block = {}
            block["func"] = "shift_op"

            # Template parameters
            block["template"] = []
            if index < ws:
                block["template"].append("t_%s_struct" % input_type_name)
            else:
                block["template"].append("t_%s_lb_struct" % input_type_name)
            block["template"].append("t_%s_lb_struct" % input_type_name)
            if out_index < (dindex):
                block["template"].append("t_%s_lb_struct" % input_type_name)
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
            if index < ws:
                block["template"].append("%0d" % node["in_ops"])
            else:
                block["template"].append("%0d" % node["ich_ops"])
            block["template"].append("%0d" % node["ich_ops"])

            block["args"] = []

            if index < ws:
                block["args"].append("s_%s[%0d]" % (input_name, (dfw - 1 - fw - pad_value)%ws))
                # block["args"].append("s_%s[%0d]" % (input_name, index))
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
                declare["type"] = "t_%s_lb_struct" % output_name
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
                declare["type"] = "t_%s_lb_struct" % output_name
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
            # depth = 2
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

            # Line buffer long branch must be split in ws parts
            if (fw > (dfw-ws-1)):
                depth = int(node["iw"]/node["ws"])*int(node["ich"]/node["ich_ops"])
            else:
                depth = int(node["ich"]/node["ich_ops"])

            if (index < (dindex-ws)):

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

