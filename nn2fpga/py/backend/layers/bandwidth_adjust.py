import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import math

# def parse(name, node, ich_ops, och_ops, ow_ops, iw_ops, dim="i", skip=False):
def parse(name, node, dim="i", skip=False):
    input_name = node["input"][0]
    node_name = "bandwidth_adjust_%s" % input_name
    input_type_name = input_name.replace("_skip", "")
    input_type_name = input_name
    
    output_name = node["output"][0]
    output_type_name = output_name.replace("_skip", "")
    output_type_name = output_name
    
    ich_ops = node["ich_ops"]
    och_ops = node["och_ops"]
    iw_ops = node["iw_ops"]
    ow_ops = node["ow_ops"]

    dim = node["dim_adj"]
    adjust_name = input_name
    block = {}
    block["func"] = "bandwidth_adjust"

    # Template parameters
    block["template"] = []
    block["template"].append("t_%s_struct" % input_type_name)
    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("c_%s_%sch" % (node_name, dim))
    block["template"].append("c_%s_%sw" % (node_name, dim))
    block["template"].append("c_%s_%sh" % (node_name, dim))
    block["template"].append("c_%s_iw_ops" % node_name)
    block["template"].append("c_%s_ow_ops" % node_name)
    block["template"].append("c_%s_ich_ops" % node_name)
    block["template"].append("c_%s_och_ops" % node_name)
    if skip:
        block["template"].append({"name": "true", "comment": "skip connection flag"})
    else:
        block["template"].append({"name": "false", "comment": "skip connection flag"})

    block["args"] = []
    block["args"].append("s_%s" % input_name)
    block["args"].append("s_%s" % output_name)

    # TODO: write the type declaration
    block["defines"] = {}
    block["defines"]["t_%s" % output_name] = ["type", "t_%s" % input_name.replace("_skip", "")]
    output_ops = node["och_ops"]
    output_vector_type = "std::array<t_%s, %0d>" % (input_name.replace("_skip", ""), output_ops)
    block["defines"]["t_%s_vector" % output_name] = ["type", output_vector_type]

    # In case of skip connection, the last is not needed
    if not skip:
        block["defines"]["t_%s_struct" % output_name] = [
            "struct",
            [["data", "std::array<t_%s_vector, 1>" % output_name], ["last", "bool"]]
        ]
    else:
        block["defines"]["t_%s_struct" % output_name] = [
            "struct",
            [["data", "std::array<t_%s_vector, 1>" % output_name]]
        ]

    block["defines"]["c_%s_%sch" % (node_name, dim)] = ["const", node["%sch" % dim]]
    block["defines"]["c_%s_%sh" % (node_name, dim)] = ["const", node["%sh" % dim]]
    block["defines"]["c_%s_%sw" % (node_name, dim)] = ["const", node["%sw" % dim]]
    block["defines"]["c_%s_iw_ops" % (node_name)] = ["const", "%s" % iw_ops]
    block["defines"]["c_%s_ow_ops" % (node_name)] = ["const", "%s" % ow_ops]
    block["defines"]["c_%s_ich_ops" % node_name] = ["const", "%s" % ich_ops]
    block["defines"]["c_%s_och_ops" % node_name] = ["const", "%s" % och_ops]

    block["output"] = []
    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ow_ops"]
    block["declare"].append(declare)

    depth = math.ceil(node["%sch" % dim] / output_ops) + 1

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

