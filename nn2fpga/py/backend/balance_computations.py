import os
import sys
import pulp
import math
from backend.ilp_utils import find_divisors
from backend.ilp_utils import find_range

def parallel_ops_number(layers_info, clamp=None, board="ULTRA96v2", prj_root="/tmp"):

    if (board == "ULTRA96v2"):
        NUM_DSP = 400
    elif (board == "KRIA"):
        NUM_DSP = 1300

    MIN_OP = 1
    DELTA = 1

    def find_high_comp(layers_info):
        best_one = layers_info[0]
        best_index = 0

        for i, layer_info in enumerate(layers_info):
            # The value stored is 1/#COMPUTATIONS
            if (layer_info[1] < best_one[1]):
                best_one = layer_info
                best_index = i

        return best_one, best_index

    def happiness(choices, elem, layers_info):
        return choice[0] * layers_info[elem][1]

    num_layers = len(layers_info)

    # Counting merged 1x1 layers
    for i in range(num_layers):
        if layers_info[i][4]:
            NUM_DSP -= 1

    _, best_index = find_high_comp(layers_info)

    prob = pulp.LpProblem("Parallel ops", pulp.LpMaximize)

    choice = pulp.LpVariable.dicts("Layers parallel comp.", range(1), cat="Integer")
    # choices = pulp.LpVariable.dicts("Layers parallel comp.", range(num_layers), cat="Continuous")

    prob += pulp.lpSum(happiness(choice, best_index, layers_info))

    # for i in range(num_layers - 1):
    #     prob += happiness(choices, i, layers_info) == happiness(choices, i+1, layers_info)

    prob += pulp.lpSum([choice[0]*layers_info[i][2]/layers_info[i][5] for i in range(num_layers)]) <= NUM_DSP

    # TODO: Do architectural changes to avoid limiting the parallel ops
    if clamp is not None:
        prob += pulp.lpSum([choice]) <= clamp

    prob.writeLP(prj_root + "/parallel_ops.lp")

    prob.solve()

    print("Status:", pulp.LpStatus[prob.status])

    all_divisors, layers_divisors, layers_offset, layers_name = find_divisors(layers_info, clamp=clamp)

    # Searching for an integer divisor of the number of DSPs that is the closest to the
    # number of operations that should be executed in parallel
    max_data = int(choice[0].value())
    low_range, high_range = find_range(
        all_divisors[layers_offset[best_index]:layers_offset[best_index]+layers_divisors[best_index]],
        max_data
    )

    parallel_op = {}
    max_data = low_range
    for i in range(num_layers):
        # Returning the layers name together with the computed number of 
        # operations that should be executed in parallel
        data = int(max_data/layers_info[i][5])
        low_range, high_range = find_range(
            all_divisors[layers_offset[i]:layers_offset[i]+layers_divisors[i]],
            data
        )
        if data == 0:
          data = 1
        parallel_op[layers_info[i][0]] = data
        print(all_divisors[layers_offset[i]:layers_offset[i]+layers_divisors[i]], data, low_range, high_range)

    return parallel_op

def ilp(io_dict, off_chip_storage, board="ULTRA96v2", double_packing=True, prj_root="/tmp"):

    if off_chip_storage:
        clamp = 8
    else:
        clamp = 64

    layers_info = []

    max_total = 1
    for node_name, node_info in io_dict.items():
        if 'conv' in node_info["type"]:
            if max_total > node_info["total"]:
                max_total = node_info["total"]

    for node_name, node_info in io_dict.items():
        if 'conv' in node_info["type"]:

            value = node_info["total"]/max_total
            # value = 2**value
            print(node_name, node_info["total"], value)

            layers_info.append(
                [
                    node_name,
                    node_info["total"],
                    node_info["kernel"],
                    node_info["img_ch"],
                    node_info["merge_1x1"],
                    value,
                    node_info["och"]
                ]
            )

    parallel_ops = parallel_ops_number(layers_info, clamp, board, prj_root=prj_root)

    # # Manual tuning for DAC
    # parallel_ops['/model_0/conv/Conv'] = 4
    # parallel_ops['/model_1/conv/Conv'] = 16
    # parallel_ops['/model_3/conv/Conv'] = 16
    # parallel_ops['/model_5/conv/Conv'] = 14
    # parallel_ops['/model_7/conv/Conv'] = 13
    # parallel_ops['/model_11/conv/Conv'] = 26
    # parallel_ops['/model_13/m_0/Conv'] = 4

    print(parallel_ops)

    for node_name, ops in parallel_ops.items():
        io_dict[node_name]["ops"] = ops
        io_dict[node_name]["dp"] = False

        # Evaluating neccessary output channels to avoid losing performance
        io_dict[node_name]["out_par"] = int(io_dict[node_name]["oh"] * io_dict[node_name]["ow"] * io_dict[node_name]["och"] * io_dict[node_name]["total"])
        if io_dict[node_name]["out_par"] == 0:
            io_dict[node_name]["out_par"] = 1

    # if double_packing:
    #     for node_name, ops in parallel_ops.items():
    #         if io_dict[node_name]["fh"]*io_dict[node_name]["fw"] > 1:
    #             io_dict[node_name]["reuse"] = 2
    #             io_dict[node_name]["dp"] = True

    # if (board == "ULTRA96v2"):
    #     NUM_BRAM = 216
    # elif (board == "KRIA"):
    #     NUM_BRAM = 144

    # tot_bram = 0
    # for node_name, ops in parallel_ops.items():
    #     node = io_dict[node_name]
    #     och = node["och"]
    #     if (och < ops):
    #         # Add reuse allows performance boost without pipelining ich loop
    #         # And without increasing needed weights bandwitdth
    #         node["ops"] = och
    #         node["reuse"] = int(ops / och)
    #     else:
    #         node["ops"] = ops
    #         node["reuse"] = 1

    #     par_read = node["ops"]*node["fh"]*node["fw"]
    #     node_bram = int(par_read/8)
    #     if (par_read % 8):
    #         node_bram = node_bram + 1

    # if tot_bram > NUM_BRAM:
    #     mult_reuse = int(tot_bram/NUM_BRAM) + 1 
    # else:
    #     mult_reuse = 1

    # for node_name, ops in parallel_ops.items():
    #     io_dict[node_name]["ops"] = int(io_dict[node_name]["ops"] / mult_reuse)
    #     io_dict[node_name]["reuse"] = io_dict[node_name]["reuse"] * mult_reuse

    return io_dict
