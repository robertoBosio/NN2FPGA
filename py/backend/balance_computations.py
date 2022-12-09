import os
import sys
import pulp
import math

def parallel_ops_number(layers_info):

    NUM_DSP = 400
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
        return choices[elem] * layers_info[elem][1] * layers_info[elem][2]

    num_layers = len(layers_info)

    _, best_index = find_high_comp(layers_info)

    prob = pulp.LpProblem("Parallel ops", pulp.LpMaximize)

    choices = pulp.LpVariable.dicts("Layers parallel comp.", range(num_layers), cat="Integer")
    # choices = pulp.LpVariable.dicts("Layers parallel comp.", range(num_layers), cat="Continuous")

    prob += pulp.lpSum(happiness(choices, best_index, layers_info))

    # for i in range(num_layers - 1):
    #     prob += happiness(choices, i, layers_info) == happiness(choices, i+1, layers_info)

    for i in range(num_layers):
        prob += happiness(choices, i, layers_info) >= happiness(choices, best_index, layers_info)

    prob += pulp.lpSum([choices[i]*layers_info[i][2] for i in range(num_layers)]) <= NUM_DSP

    for i in range(num_layers):
        prob += pulp.lpSum([choices[i]]) >= MIN_OP

    prob.writeLP("tmp/parallel_ops.lp")

    prob.solve()

    print("Status:", pulp.LpStatus[prob.status])

    parallel_op = {}
    for i in range(num_layers):
        # Returning the layers name together with the computed number of 
        # operations that should be executed in parallel
        exp = int(math.log2(choices[i].value()))
        data = 2**exp
        parallel_op[layers_info[i][0]] = int(data)
    
    return parallel_op

def opt(layers_info):

    parallel_ops = parallel_ops_number(layers_info)

    return parallel_ops
