import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

# def merge_quant(io_dict, quant_info, inherit_quant=False):

#     # Merging consecutive quantizations
#     change = True
#     new_quant_info = {}
#     while change:

#         change = False

#         remove_node = []

#         for name, node in quant_info.items():

#             keep_elem = []

#             # Creating new node for the quantization info
#             new_node = {}
#             new_node.setdefault("seq", [])
#             new_node.setdefault("seq_scale", [])
#             new_node.setdefault("seq_out", [])
#             new_node.setdefault("others", [])
#             new_node.setdefault("others_scale", [])

#             # Looking for node with in input the previous output
#             for i, output in enumerate(node["seq_out"]):
#                 if output in quant_info.keys():
#                     for j, new_output in enumerate(quant_info[output]["seq_out"]):
#                         new_node_name = quant_info[output][new_output]
#                         new_scale = quant_info[output]["seq_scale"][j]
#                         new_node[new_output] = new_node_name
#                         new_node["seq"].append(new_node_name)
#                         new_node["seq_scale"].append(new_scale)
#                         new_node["seq_out"].append(new_output)

#                     for other in quant_info[output]["others"]:
#                         new_node["others"].append(other)

#                     # If the other node has already a quantization it can be
#                     # inherited
#                     if inherit_quant:
#                         for other in quant_info[output]["others_scale"]:
#                             new_node["others_scale"].append(other)

#                     remove_node.append(output)

#                     change=True

#                 else:
#                     keep_elem.append(i)

#             if change:
#                 for i in keep_elem:
#                     output = node["seq_out"][i]
#                     new_node[output] = quant_info[name][output]
#                     new_node["seq"].append(quant_info[name]["seq"][i])
#                     new_node["seq_scale"].append(quant_info[name]["seq_scale"][i])
#                     new_node["seq_out"].append(quant_info[name]["seq_out"][i])

#                 if not inherit_quant:
#                     for _ in new_node["others"]:
#                         new_node["others_scale"].append(
#                             quant_info[name]["seq_scale"][0]
#                         )

#                 quant_info[name] = new_node

#                 # Every time there is a change the dependencies must be evaluated
#                 # from sratch

#                 if len(remove_node) > 0:
#                     break
        
#         # If there are other nodes attached to this quantization it should not 
#         # removed from database
#         for output in remove_node:
#             del quant_info[output]

#     for name, node in quant_info.items():
#         print(name, node)
#     sys.exit(0)

#     return io_dict

def merge_quant(io_dict, quant_info, inherit_quant=False):

    # Merging consecutive quantizations
    new_quant_info = {}
    while True:

        remove_node = []

        for name, node in quant_info.items():

            keep_elem = []

            # Creating new node for the quantization info
            new_node = {}
            new_node.setdefault("seq", [])
            new_node.setdefault("seq_scale", [])
            new_node.setdefault("seq_out", [])
            new_node.setdefault("changed", False)
            new_node.setdefault("removed", [])

            # Looking for node with in input the previous output
            for i, output in enumerate(node["seq_out"]):
                if output in quant_info.keys():
                    # Avoid splits with unbalanced quantizations, i.e
                    # One branch with quant and one without quant
                    if len(quant_info[output]["others"]) == 0:
                        for j, new_output in enumerate(quant_info[output]["seq_out"]):
                            new_node_name = quant_info[output][new_output]
                            new_scale = quant_info[output]["seq_scale"][j]
                            new_node[new_output] = new_node_name
                            new_node["seq"].append(new_node_name)
                            new_node["seq_scale"].append(new_scale)
                            new_node["seq_out"].append(new_output)
                            new_node["changed"] = True
                        new_node["removed"].append(quant_info[output]["seq"][0])
                        remove_node.append(output)
                else:
                    keep_elem.append(i)

            if len(remove_node) == 0:
                continue

            for i in keep_elem:
                output = node["seq_out"][i]
                new_node[output] = quant_info[name][output]
                new_node["seq"].append(quant_info[name]["seq"][i])
                new_node["seq_scale"].append(quant_info[name]["seq_scale"][i])
                new_node["seq_out"].append(quant_info[name]["seq_out"][i])

            quant_info[name] = new_node

            # Every time there is a change the dependencies must be evaluated
            # from sratch

            if len(remove_node) > 0:
                break
        
        # If there are other nodes attached to this quantization it should not 
        # be removed from database
        if len(remove_node) > 0:
            for output in remove_node:
                del quant_info[output]
        else:
            break

    remove_node = []
    for name, node in io_dict.items():
        if "quant" in name.lower():
            new_output  = []
            new_scale   = []
            # The quant_info database is already selecting the point where 
            # quant layers can be merged, for this reason only the changed layers
            # should be updated
            input_name = node["input"][0]
            if input_name in quant_info.keys():
                if quant_info[input_name]["changed"]:
                    seq = quant_info[input_name]["seq"]
                    seq_out = quant_info[input_name]["seq_out"]
                    seq_scale = quant_info[input_name]["seq_scale"]
                    for i, next_output in enumerate(seq_out):
                        new_output.append(next_output)
                        new_scale.append(seq_scale[i])
                    for rem_name in quant_info[input_name]["removed"]:
                        remove_node.append(rem_name)

            if len(new_output) > 0:
                io_dict[name]["output"] = new_output
                io_dict[name]["scale"] = new_scale

    # print(remove_node)
    for name in remove_node:
        del io_dict[name]

    return io_dict

def extract_quant_info(model, io_dict, init_info):

    quant_info = {}

    last_input = None
    last_output = None
    quant_dict = {}

    for node_name, node in io_dict.items():
        if 'quant' in node_name.lower():

            # Saving connections and scale factors for all the quantizations
            scale_name   = io_dict[node_name]["input"][1]
            scale_info   = init_info[scale_name]
            scale_factor = numpy_helper.to_array(scale_info)
            scale_factor = np.log2(scale_factor)

            quant_info.setdefault(node["input"][0], {})
            quant_info[node["input"][0]].setdefault("seq", [])
            quant_info[node["input"][0]].setdefault("seq_scale", [])
            quant_info[node["input"][0]].setdefault("seq_out", [])
            quant_info[node["input"][0]].setdefault("others", [])
            quant_info[node["input"][0]].setdefault("others_scale", [])
            quant_info[node["input"][0]].setdefault("changed", False)
            quant_info[node["input"][0]].setdefault("removed", [])

            quant_info[node["input"][0]][node["output"][0]] = node_name
            quant_info[node["input"][0]]["seq"].append(node_name)
            quant_info[node["input"][0]]["seq_scale"].append(scale_factor)
            quant_info[node["input"][0]]["seq_out"].append(node["output"][0])

        else:

            # Adding direct connections to other layers from the quant
            # If the quant output is connected with other layers the
            # quantization cannot be completely pruned
            for input in node["input"]:
                if input in quant_info.keys():
                    scale_factor = quant_info[input]["seq_scale"][0]
                    quant_info[input]["others"].append(node_name)

    return quant_info

