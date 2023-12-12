import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def parse(name, node, adjust_name, in_ops, adjust_ops, dim="i"):
    
    node_name = "bandwidth_adjust_%s" % adjust_name
    input_name = adjust_name
    # input_type_name = input_name.replace("_skip", "")
    input_type_name = input_name

    output_name = input_name + "_adj"
    output_type_name = output_name

    old_in_ops = node[in_ops]
    node[in_ops] = node[adjust_ops]

    block = {}
    block["func"] = "bandwidth_adjust"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("c_%s_%sch" % (node_name, dim))
    block["template"].append("c_%s_%sw" % (node_name, dim))
    block["template"].append("c_%s_%sh" % (node_name, dim))
    block["template"].append("c_%s_ow_ops_in" % node_name)
    block["template"].append("c_%s_ow_ops" % node_name)
    block["template"].append("c_%s_old_in_ops" % node_name)
    block["template"].append("c_%s_in_ops" % node_name)

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    # TODO: write the type declaration
    block["defines"] = {}
    output_type = "t_%s" % output_type_name
    output_ops = node[adjust_ops]
    output_vector_type = "std::array<t_%s, %0d>" % (input_name.replace("_skip", ""), output_ops)
    block["defines"]["t_%s_vector" % output_name] = ["type", output_vector_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, 1>" % output_name], ["last", "bool"]]
    ]

    block["defines"]["c_%s_%sch" % (node_name, dim)] = ["const", node["%sch" % dim]]
    block["defines"]["c_%s_%sh" % (node_name, dim)] = ["const", node["%sh" % dim]]
    block["defines"]["c_%s_%sw" % (node_name, dim)] = ["const", node["%sw" % dim]]
    block["defines"]["c_%s_ow_ops_in" % (node_name)] = ["const", node["ow_ops_in"]]
    block["defines"]["c_%s_ow_ops" % (node_name)] = ["const", node["ow_ops"]]
    block["defines"]["c_%s_old_in_ops" % node_name] = ["const", old_in_ops]
    block["defines"]["c_%s_in_ops" % node_name] = ["const", output_ops]

    block["output"] = []
    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ow_ops"]
    block["declare"].append(declare)

    depth = 3
    block["pragma"] = []
    pragma = {}
    pragma["name"] = "stream"
    pragma_name = "s_%s" % (output_name)
    options = [
        ["variable", pragma_name],
        ["depth", depth],
        ["type", "fifo"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    return [block]

