import os
import sys
import pulp
import math
from backend.ilp_utils import find_divisors

def reuse_number(layers_info, clamp=16):

    DELTA = 1

    def find_divisors(layers_info):
        all_divisors = []
        layers_divisors = []
        layers_offset = []
        layers_name = []
        offset = 0
        for i, layer_info in enumerate(layers_info):
            divisors = 1
            layers_offset.append(offset)
            all_divisors.append(1)
            for k in range(2, min([layer_info[3], clamp])+1):
                if (layer_info[3] % k) == 0:
                    all_divisors.append(k)
                    divisors = divisors + 1
            layers_divisors.append(divisors)
            offset = offset + divisors
            layers_name.append(layer_info[0])
        return all_divisors, layers_divisors, layers_offset, layers_name

    def find_high_comp(layers_info, layers_name):
        best_one = layers_info[0]
        best_index = 0

        for i, layer_info in enumerate(layers_info):
            # The value stored is 1/#COMPUTATIONS
            if (layer_info[1] < best_one[1]):
                best_one = layer_info
                best_index = layers_name.index(layer_info[0])

        return best_one, best_index

    def happiness(choice, divisor, layer_info):
        return choice * divisor * layer_info[1]

    all_divisors, layers_divisors, layers_offset, layers_name = find_divisors(layers_info)

    num_divisors = len(all_divisors)

    _, best_index = find_high_comp(layers_info, layers_name)

    prob = pulp.LpProblem("Reuse weights", pulp.LpMaximize)

    choices = pulp.LpVariable.dicts(
        "Layers weights resue.",
        range(num_divisors),
        cat="Binary"
    )

    prob += pulp.lpSum(
        [
            happiness(
                choices[i + layers_offset[best_index]],
                all_divisors[i + layers_offset[best_index]],
                layers_info[best_index]
            ) for i in range(layers_divisors[best_index])
        ]
    )

    for j, offset in enumerate(layers_offset):
        prob += pulp.lpSum(
            [
                choices[offset+i] for i in range(layers_divisors[j])
            ]
        ) == 1
        prob += pulp.lpSum(
            [
                happiness(
                    choices[i + layers_offset[j]],
                    all_divisors[i + layers_offset[j]],
                    layers_info[j]
                ) for i in range(layers_divisors[j])
            ]
        ) >= pulp.lpSum(
            [
                happiness(
                    choices[i + layers_offset[best_index]],
                    all_divisors[i + layers_offset[best_index]],
                    layers_info[best_index]
                ) for i in range(layers_divisors[best_index])
            ]
        )

    prob.writeLP("tmp/reuse.lp")

    prob.solve()

    print("Status:", pulp.LpStatus[prob.status])

    if (pulp.LpStatus[prob.status] == "Infeasible"):
        sys.exit(0)

    reuse = {}
    for i, name in enumerate(layers_name):
        # Returning the layers name together with the computed number of 
        # operations that should be executed in parallel
        offset = layers_offset[i]
        for k in range(layers_divisors[i]):
            data = int(choices[offset+k].value())
            if (data == 1):
                reuse[name] = all_divisors[offset+k]
    
    return reuse

def ilp(io_dict):

    clamp = 16
    layers_info = []

    for node_name, node_info in io_dict.items():
        if ('const' == node_info["type"]):
            if node_info["off_chip_memory"]:
                layers_info.append(
                    [
                        node_name,
                        node_info["total"],
                        node_info["kernel"],
                        node_info["img_ch"]
                    ]
                )

    reuse = reuse_number(layers_info, clamp)

    print(reuse)

    for node_name, value in reuse.items():
        io_dict[node_name]["reuse"] = value

    return io_dict
