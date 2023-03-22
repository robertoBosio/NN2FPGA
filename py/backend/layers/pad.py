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
    block["func"] = "PadInput"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_ih" % name)
    block["template"].append("c_%s_iw" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_pad" % name)

    block["args"] = []

    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s_padded" % input_name)

    block["declare"] = []

    declare = {}
    declare["name"] = "s_%s_padded" % input_name
    declare["type"] = "t_%s_struct" % input_name
    declare["is_array"] = False
    declare["dim"] = 1

    block["declare"].append(declare)

    pragma = {}
    pragma["name"] = "s_%s_padded" % input_name
    pragma["depth"] = 2

    block["pragma"] = []
    block["pragma"].append(pragma)

    return block
