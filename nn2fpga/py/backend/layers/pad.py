import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def parse(name, node):

    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = input_name
    output_type_name = output_name.replace("_skip", "")

    block = {}
    block["func"] = "pad_input"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_lb_struct" % input_type_name)
    block["template"].append("t_%s_window_struct" % input_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_pad" % name)
    block["template"].append("c_%s_ws" % name)
    block["template"].append("%0d" % node["ich_ops"])
    block["template"].append("%0d" % node["ich_ops"])

    block["args"] = []

    block["args"].append("s_%s_pre_pad" % input_name)
    block["args"].append("s_%s_compute" % input_name)

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s_compute" % input_name
    declare["type"] = "t_%s_window_struct" % input_name
    declare["is_array"] = True
    # declare["dim"] = node["fh"]*(node["fw"]+(node["ws"]-1)*node["stride"])
    declare["dim"] = 1

    block["declare"].append(declare)

    # depth = node["och"]*int(node["och"]/node["ich"])*4
    depth = 2
    # depth = node["fh"]*node["fw"]
    pragma = {}
    pragma["name"] = "stream"
    options = [
        ["variable", "s_%s_compute" % (input_name)],
        ["depth", depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options

    block["pragma"] = []
    block["pragma"].append(pragma)

    return block
