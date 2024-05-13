import os
import sys
import time
import pulp
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate
import math
import numpy as np
from backend.utils import extract_board_info
from backend.ilp_utils import generate_valid_combinations
from backend.ilp_utils import find_common_mult
from backend.ilp_utils import find_max_commond_div
from backend.graph import extract_connections, prev_layers, next_layers
from backend.layers.weights import compute_bram_layer

def packing_feature(operands_bitwidth, par):
    """ Returns the number of operation that can be packed in a single DSP. 
    
    Arguments:
        operand_bitwidth: Tuple containing information about the bitwidth of the operands.
        par: Tuple containing the parallelization chosen for the layer in the format (och, ich, ow).

    Returns:
        int: The number of operations that can be packed in a single DSP.
        tuple: The packing for each dimension.
    """

    operand_bits = max(operands_bitwidth)
    if (operand_bits == 8):
        if (par[2] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    elif (operand_bits == 4):
        if (par[0] % 2 == 0 and par[2] % 2 == 0):
            return 4, (2, 2)
        elif (par[2] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    return 1, (1, 1)

def layers_extractions(io_dict):
    """ Extracts useful layers information from the io_dict and stores it in a dictionary.""" 
    
    index = 0
    layers_info = []
    par_layers = ["conv", "pool"]
    for node_name, node_info in io_dict.items():
        if node_info["type"] in par_layers:
             
            kernel = node_info["fw"] * node_info["fh"]
            depth = False
            merge_1x1 = False
            weight_bits = []
            act_bits = []
            och_1x1 = node_info["och"]
            
            if node_info["type"] == "conv":
                depth = node_info["depth"]
                merge_1x1 = node_info["merge_1x1"]
                weight_bits.append(node_info["weight_quant"]["bits"])
                act_bits.append(node_info["input_quant"]["bits"])
                if (merge_1x1):
                    och_1x1 = node_info["merge_node"]["och"]
                    weight_bits.append(node_info["merge_node"]["weight_quant"]["bits"])
                    act_bits.append(node_info["merge_node"]["input_quant"]["bits"])
            else:
                weight_bits.append(0)
                act_bits.append(node_info["input_quant"]["bits"])
            
            layers_info.append(
                {
                    "type": node_info["type"],
                    "name": node_name,
                    "total": node_info["total"],
                    "kernel": kernel,
                    "merge_1x1": merge_1x1,
                    "weight_bits": weight_bits,
                    "act_bits": act_bits,
                    "ich" : node_info["ich"],
                    "och" : node_info["och"],
                    "iw" : node_info["iw"],
                    "ow" : node_info["ow"],
                    "ih" : node_info["ih"],
                    "oh" : node_info["oh"],
                    "depth": depth,
                    "index": index,
                    "produce": node_info["start_comp_layer"],
                    "och_1x1": och_1x1
                }
            )
            index += 1

    return layers_info

def generate_architectures(layers_info, NUM_DSP):
    """Given a list of layers, generate all the valid parallelization for each layer. """
    
    valid_par_solutions = []
    for layer in layers_info:
        max_och_par = layer["och"]
        max_ich_par = layer["ich"]
        max_ow_par = layer["ow"]

        # Depthwise convolutions cannot be parallelized on output channels.
        if (layer["depth"] or layer["type"] == "pool"):
            max_och_par = 1

        # In case of merged convolutions, take into account the gcd of the
        # maximum parallelization of och of the two. The transformation is
        # described in the graph optimization section of the paper. 
        if (layer["merge_1x1"]):
            max_och_par = math.gcd(layer["och"], layer["och_1x1"])
        
        if (layer["type"] == "pool"):
            max_ow_par = 1
        
        # Clipping the maximum parallelization to the available DSPs, since it is not
        # possible that one layer uses all the DSPs, considering the packing
        op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), (max_och_par, max_ich_par, max_ow_par))
        op_clip = (NUM_DSP / layer["kernel"]) * op_per_dsp
        
        valid_par_solutions.append(generate_valid_combinations(
            och=max_och_par, ich=max_ich_par, iw=max_ow_par, iw_clip=4, och_clip=20, op_clip=op_clip))

    return valid_par_solutions

def parallelismILP(layers_info, valid_par_solutions, NUM_DSP, NUM_PORTS, prj_root="/tmp"):
    """ Find the parallelization for each layer that maximize the throughput of the network."""

    constraints_counter = 0

    # valid_tot_par_solutions stores the total parallelization for each valid
    # solution and it is useful to use lpDot to compute the parallelization
    # chosen for a layer
    valid_tot_par_solutions = []
    for i, par_sol in enumerate(valid_par_solutions):
        valid_tot_par_solutions.append([])
        for single_par in par_sol:
            valid_tot_par_solutions[i].append(np.prod(single_par))

    # valid_iter_linebuffer stores the line buffer number of iteration for each valid
    # solution and it is useful to linearize the constraint of the line buffer
    valid_iter_linebuffer = []
    valid_iter_produce = []
    for par_sol, layer in zip(valid_par_solutions, layers_info):
        valid_iter_linebuffer.append([])
        layer_iter =  layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in par_sol:
            valid_iter_linebuffer[-1].append(layer_iter // (single_par[1] * single_par[2]))
        
        if (layer["produce"]):
            valid_iter_produce.append([])
            for single_par in par_sol:
                valid_iter_produce[-1].append(layer_iter // single_par[1])
    
    valid_iter_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_iter_solutions.append([])
        layer_iter = layer["total"]
        for single_par in layer_par:
            valid_iter_solutions[-1].append(layer_iter // np.prod(single_par))

    # Creating a second dictionary in which merged convolution are splitted and
    # pool layers are removed to better compute DSPs and PORTs
    layers_info_unmerged = []
    for layer in [x for x in layers_info if x["type"] == "conv"]:
        layers_info_unmerged.append(layer.copy())
        layers_info_unmerged[-1]["weight_bits"] = layer["weight_bits"][0]
        layers_info_unmerged[-1]["act_bits"] = layer["act_bits"][0]
        layers_info_unmerged[-1]["merge_1x1"] = False
        if layer["merge_1x1"]:
            layers_info_unmerged.append(
                {
                    "name": layer["name"],
                    "total": layer["total"],
                    "kernel": 1,
                    "merge_1x1": layer["merge_1x1"],
                    "weight_bits": layer["weight_bits"][1],
                    "act_bits": layer["act_bits"][1],
                    "ich" : layer["ich"],
                    "och" : layer["och_1x1"],
                    "depth": layer["depth"],
                    "index": layer["index"]
                }
            )

    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing 
    valid_dsp_solutions = []
    for layer in layers_info_unmerged:
        valid_dsp_solutions.append([])
        layer_par = valid_par_solutions[layer["index"]]
        
        for single_par in layer_par:
            op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), single_par)
            dsp_used = (np.prod(single_par) * layer["kernel"]) / op_per_dsp
            valid_dsp_solutions[-1].append(dsp_used)
    
    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for layer in layers_info_unmerged:
        valid_bram_solutions.append([])
        layer_par = valid_par_solutions[layer["index"]]
        n_weights = layer["ich"] * layer["och"] * layer["kernel"]

        if (layer["depth"]):
            n_weights = layer["ich"] * layer["kernel"]

        for single_par in layer_par:
            bram_used = compute_bram_layer(layer["weight_bits"], n_weights, np.prod(single_par[:2]) * layer["kernel"])
            valid_bram_solutions[-1].append(bram_used)

    ####### Problem formulated as ILP #######
    # Minimize latencies
    prob = pulp.LpProblem("Parallel_ops", pulp.LpMinimize)
    
    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    var = pulp.LpVariable("slack", lowBound=0, cat="Integer")
    
    # Objective function: minimize the maximum latency of the layers.
    prob += (
        var,
        "Minimization variable"
    )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in [x["index"] for x in layers_info]:
        constraints_counter += 1
        ones = [1] * len(layer_binary_variables[layer_index])
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )

    # Constraint: The total number of DSPs used to achieve the chosen
    # parallelization should be lower than the available ones.
    constraints_counter += 1
    prob += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_DSP,
        f"DSP_constraint"
    )

    # Constraint: The total number of BRAMs used to achieve the chosen
    # parallelization should be lower than the available ones.
    constraints_counter += 1
    prob += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_bram_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_PORTS,
        f"BRAM_constraint"
    )
    
    # Constraints: The latency of each layer should be equal or lower to the minimization variable.
    for layer_index in [x["index"] for x in layers_info]:
        constraints_counter += 1
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= var,
            f"Latency_constraint_layer_{layer_index}"
        )

    # Constraints: The latency of each line buffer should be equal or lower to the minimization variable.
    for layer_index in [x["index"] for x in layers_info]:
        constraints_counter += 1
        prob += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= var,
            f"Linebuffer_constraint_layer_{layer_index}"
        )
    
    # Constraint: the latency of the produce stream should be equal or lower to the minimization variable.
    for layer_index in [x["index"] for x in layers_info]:
        if (layers_info[layer_index]["produce"]):
            constraints_counter += 1
            prob += (
                pulp.lpDot(layer_binary_variables[layer_index].values(),
                        valid_iter_produce[layer_index]) <= var,
                f"Produce_constraint_layer_{layer_index}"
            )
    
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(timeLimit=10, msg=0))
    end_time = time.time()
    if (prob.status == pulp.LpStatusInfeasible):
        print("Throughput problem unfeasible")
        exit(0)

    # Recovering the values of the paralellism for each layer from the binary variables.
    parallel_op = {}
    for i, layer in enumerate(valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]

    return parallel_op, int(pulp.value(prob.objective)), sum([len(s) for s in valid_par_solutions]), constraints_counter, (end_time - start_time)

def resourceILP(layers_info, model_II, valid_par_solutions, parallel_op, NUM_DSP, NUM_PORTS, prj_root="/tmp"):
    """ Given the throughput of the network, find the parallelization for each layer that minimize the resources usage."""
    
    # Creating a second dictionary in which merged convolution are splitted and
    # pool layers are removed to better compute DSPs and PORTs
    layers_info_unmerged = []
    for layer in [x for x in layers_info if x["type"] == "conv"]:
        layers_info_unmerged.append(layer.copy())
        layers_info_unmerged[-1]["weight_bits"] = layer["weight_bits"][0]
        layers_info_unmerged[-1]["act_bits"] = layer["act_bits"][0]
        layers_info_unmerged[-1]["merge_1x1"] = False
        if layer["merge_1x1"]:
            layers_info_unmerged.append(
                {
                    "name": layer["name"],
                    "total": layer["total"],
                    "kernel": 1,
                    "merge_1x1": layer["merge_1x1"],
                    "weight_bits": layer["weight_bits"][1],
                    "act_bits": layer["act_bits"][1],
                    "ich" : layer["ich"],
                    "och" : layer["och_1x1"],
                    "depth": layer["depth"],
                    "index": layer["index"]
                }
            )

    # Retriving only the parallelism combinations for lower throughput to save
    # resources in fast layers. The parallelization over ow is fixed
    layer_binary_variables = []
    clamped_valid_par_solutions = []
    valid_tot_par_solutions = []
    for i, solution_set in enumerate(valid_par_solutions):
        clamped_valid_par_solutions.append([])
        valid_tot_par_solutions.append([])
        chosen_par = np.prod(parallel_op[layers_info[i]['name']][0:2])
        chosen_ow = parallel_op[layers_info[i]['name']][2]
        for combination in solution_set:
            tot_par = np.prod(combination[0:2])
            ow_par = combination[2]
            if (tot_par <= chosen_par and ow_par == chosen_ow):
                clamped_valid_par_solutions[i].append(combination)
                valid_tot_par_solutions[i].append(np.prod(combination))

        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(clamped_valid_par_solutions[i])), cat="Binary"))
    
    # valid_iter_linebuffer stores the line buffer number of iteration for each valid
    # solution and it is useful to linearize the constraint of the line buffer
    valid_iter_solutions = []
    valid_iter_linebuffer = []
    valid_iter_produce = []
    for layer, layer_par in zip(layers_info, clamped_valid_par_solutions):
        valid_iter_solutions.append([])
        valid_iter_linebuffer.append([])
        layer_iter =  layer["total"]
        line_iter =  layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in layer_par:
            valid_iter_solutions[-1].append(layer_iter // np.prod(single_par))
            valid_iter_linebuffer[-1].append(line_iter // (single_par[1] * single_par[2]))
        
        if (layer["produce"]):
            valid_iter_produce.append([])
            for single_par in layer_par:
                valid_iter_produce[-1].append(line_iter // single_par[1])
    
    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing 
    valid_dsp_solutions = []
    for layer in layers_info_unmerged:
        valid_dsp_solutions.append([])
        layer_par = clamped_valid_par_solutions[layer["index"]]
        for single_par in layer_par:
            op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), single_par)
            dsp_used = (np.prod(single_par) * layer["kernel"]) / op_per_dsp
            valid_dsp_solutions[-1].append(dsp_used)
    
    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for layer in layers_info_unmerged:
        valid_bram_solutions.append([])
        layer_par = clamped_valid_par_solutions[layer["index"]]
        n_weights = layer["ich"] * layer["och"] * layer["kernel"]

        if (layer["depth"]):
            n_weights = layer["ich"] * layer["kernel"]

        for single_par in layer_par:
            bram_used = compute_bram_layer(layer["weight_bits"], n_weights, np.prod(single_par[:2]) * layer["kernel"])
            valid_bram_solutions[-1].append(bram_used)
    
    # Minimize resource usage
    prob_min = pulp.LpProblem("Resource_usage", pulp.LpMinimize)
    
    # Objective function: minimize the BRAMs + DSPs required to run the whole network.
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_bram_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) +
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]),
        f"Resource_objective"
    )
    
    # Objective function: minimize the DSPs required to run the whole network.
    # prob_min += (
    #     pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
    #                 valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]),
    #     f"DSP_constraint"
    # )
    
    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in [x["index"] for x in layers_info]:
        ones = [1] * len(layer_binary_variables[layer_index])
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )
    
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_DSP,
        f"DSP_constraint"
    )
    
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_bram_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_PORTS,
        f"BRAM_constraint"
    )
    
    for layer_index in [x["index"] for x in layers_info]:
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= model_II,
            f"Throughtput_constraint_layer_{layer_index}"
        )
    
    for layer_index in [x["index"] for x in layers_info]:
        prob_min += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= model_II,
            f"Linebuffer_constraint_layer_{layer_index}"
        )
    
    for layer_index in [x["index"] for x in layers_info]:
        if (layers_info[layer_index]["produce"]):
            prob_min += (
                pulp.lpDot(layer_binary_variables[layer_index].values(),
                        valid_iter_produce[layer_index]) <= model_II,
                f"Produce_constraint_layer_{layer_index}"
            )
    
    prob_min.solve(PULP_CBC_CMD(msg=0))
    if (prob_min.status == pulp.LpStatusInfeasible):
        print("Resource problem unfeasible")
        exit(0)
    
    parallel_op = {}
    for i, layer in enumerate(clamped_valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]
    
    return parallel_op

def balanceILP(layers_info, model_II, valid_par_solutions, parallel_op, NUM_PORTS, prj_root="/tmp"):
    """ Balance the parallelization for each layer withouth changing the throughput of the network."""
    
    # Creating a second dictionary in which merged convolution are splitted and
    # pool layers are removed to better compute DSPs and PORTs
    layers_info_unmerged = []
    for layer in [x for x in layers_info if x["type"] == "conv"]:
        layers_info_unmerged.append(layer.copy())
        layers_info_unmerged[-1]["weight_bits"] = layer["weight_bits"][0]
        layers_info_unmerged[-1]["act_bits"] = layer["act_bits"][0]
        layers_info_unmerged[-1]["merge_1x1"] = False
        if layer["merge_1x1"]:
            layers_info_unmerged.append(
                {
                    "name": layer["name"],
                    "total": layer["total"],
                    "kernel": 1,
                    "merge_1x1": layer["merge_1x1"],
                    "weight_bits": layer["weight_bits"][1],
                    "act_bits": layer["act_bits"][1],
                    "ich" : layer["ich"],
                    "och" : layer["och_1x1"],
                    "depth": layer["depth"],
                    "index": layer["index"]
                }
            )

    # Retriving only the parallelism combinations with same throughput. 
    # The parallelization over ow is fixed
    layer_binary_variables = []
    clamped_valid_par_solutions = []
    valid_tot_par_solutions = []
    valid_dist_par_solutions = []
    for i, solution_set in enumerate(valid_par_solutions):
        clamped_valid_par_solutions.append([])
        valid_tot_par_solutions.append([])
        valid_dist_par_solutions.append([])
        chosen_par = np.prod(parallel_op[layers_info[i]['name']][0:2])
        chosen_ow = parallel_op[layers_info[i]['name']][2]
        
        # Do not choose combinations which remove the packing feature over och.
        packing_over_och = parallel_op[layers_info[i]
                                       ['name']][0] % 2 == 0 and chosen_ow % 2 != 0
        for combination in solution_set:
            tot_par = np.prod(combination[0:2])
            ow_par = combination[2]

            if (tot_par == chosen_par and ow_par == chosen_ow):
                if (packing_over_och and combination[0] % 2 != 0):
                    continue
                clamped_valid_par_solutions[i].append(combination)
                valid_tot_par_solutions[i].append(np.prod(combination))
                valid_dist_par_solutions[i].append(abs(combination[0] - combination[1]))
            
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(clamped_valid_par_solutions[i])), cat="Binary"))
    
    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for layer in layers_info_unmerged:
        valid_bram_solutions.append([])
        layer_par = clamped_valid_par_solutions[layer["index"]]
        n_weights = layer["ich"] * layer["och"] * layer["kernel"]

        if (layer["depth"]):
            n_weights = layer["ich"] * layer["kernel"]

        for single_par in layer_par:
            bram_used = compute_bram_layer(layer["weight_bits"], n_weights, np.prod(single_par[:2]) * layer["kernel"])
            valid_bram_solutions[-1].append(bram_used)
    
    # valid_iter_linebuffer stores the line buffer number of iteration for each valid
    # solution and it is useful to linearize the constraint of the line buffer
    valid_iter_solutions = []
    valid_iter_linebuffer = []
    valid_iter_produce = []
    for layer, layer_par in zip(layers_info, clamped_valid_par_solutions):
        valid_iter_solutions.append([])
        valid_iter_linebuffer.append([])
        layer_iter =  layer["total"]
        line_iter =  layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in layer_par:
            valid_iter_solutions[-1].append(layer_iter // np.prod(single_par))
            valid_iter_linebuffer[-1].append(line_iter // (single_par[1] * single_par[2]))

        if (layer["produce"]):
            valid_iter_produce.append([])
            for single_par in layer_par:
                valid_iter_produce[-1].append(line_iter // single_par[1])
    
    # Minimize resource usage
    prob_min = pulp.LpProblem("Balance_parallelization", pulp.LpMinimize)
    
    # Objective function: minimize the DSPs required to run the whole network.
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dist_par_solutions[i]) for i, layer in enumerate(layers_info)]),
        f"Distance_constraint"
    )
    
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_bram_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_PORTS,
        f"BRAM_constraint"
    )
    
    for layer_index in [x["index"] for x in layers_info]:
        ones = [1] * len(layer_binary_variables[layer_index])
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )
    
    for layer_index in [x["index"] for x in layers_info]:
        prob_min += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= model_II,
            f"Linebuffer_constraint_layer_{layer_index}"
        )
    
    for layer_index in [x["index"] for x in layers_info]:
        if (layers_info[layer_index]["produce"]):
            prob_min += (
                pulp.lpDot(layer_binary_variables[layer_index].values(),
                        valid_iter_produce[layer_index]) <= model_II,
                f"Produce_constraint_layer_{layer_index}"
            )
    
    prob_min.solve(PULP_CBC_CMD(msg=0))
    if (prob_min.status == pulp.LpStatusInfeasible):
        print("Resource problem unfeasible")
        exit(0)
    
    parallel_op = {}
    for i, layer in enumerate(clamped_valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]
    
    return parallel_op

def opt_steps(layers_info, parallel_op, valid_par_solutions, prj_root="/tmp"):
    """ Balancing the mismatches between the parallelization of consecutive layers."""

    # Retriving only the parallelism combinations with same throughput.
    clamped_valid_par_solutions = []
    for i, solution_set in enumerate(valid_par_solutions):
        clamped_valid_par_solutions.append([])
        chosen_par = np.prod(parallel_op[layers_info[i]['name']][0:2])
        chosen_ow = parallel_op[layers_info[i]['name']][2]
        
        # Do not choose combination which remove the packing feature over och.
        packing_over_och = parallel_op[layers_info[i]
                                       ['name']][0] % 2 == 0 and chosen_ow % 2 != 0
        for combination in solution_set:
            tot_par = np.prod(combination[0:2])
            ow_par = combination[2]

            if (tot_par == chosen_par and ow_par == chosen_ow):
                if (packing_over_och and combination[0] % 2 != 0):
                    continue
                clamped_valid_par_solutions[i].append(combination)
    
    # Computing a value representing the mismatch between the parallelization of
    # consecutive layers. The mismatch is computed as the difference between the
    # parallelization of the output channels of the previous layer and the input
    # channels of the next layer.
    par_prev = parallel_op[layers_info[0]["name"]]
    prev_name = layers_info[0]["name"]
    tot_mismatch_before = 0
    for layer in layers_info[1:]:
        par = parallel_op[layer["name"]]
        name = layer["name"]

        # For depth conv the parallelization in output is the one over ich
        if (layer["depth"]):
            par_prev_out = par_prev[1]
        else:
            par_prev_out = par_prev[0]

        par_in = par[1]
        if (par_prev_out % par_in != 0 and par_in % par_prev_out != 0):
            adjust = find_common_mult(par_prev_out, par_in)
            if adjust > max(par_prev_out, par_in):
                tot_mismatch_before += adjust - max(par_prev_out, par_in)
        par_prev = par
        prev_name = name    

    new_parallel_op = parallel_op.copy()
    # Trying to minimize the mismatch between the parallelization of consecutive
    # layers, choosing between the valid parallelization combinations. Low effort,
    # if after the iteration the mismatches are increased, recover previous result.
    par_prev = new_parallel_op[layers_info[0]["name"]]
    tot_mismatch_after = 0
    for layer in layers_info[1:]:
        par = new_parallel_op[layer["name"]]
        name = layer["name"]
        
        # For depth conv the parallelization in output is the one over ich
        if (layer["depth"]):
            par_prev_out = par_prev[1]
        else:
            par_prev_out = par_prev[0]
        
        par_in = par[1]
        if (par_prev_out % par_in != 0 and par_in % par_prev_out != 0):
            print(f"Error: och_ops i -> {par_prev_out} % ich_ops i+1 -> {par_in} != 0, using {find_common_mult(par_prev_out, par_in)}")
            
            for i, combination in enumerate(clamped_valid_par_solutions[layer['index']]):
                if (par_prev_out % combination[1] == 0):
                    print(f"\t\tAssigning {combination} to {name}")
                    new_parallel_op[name] = combination
                    break
            else:
                tot_mismatch_after += find_common_mult(par_prev_out, par_in) - max(par_prev_out, par_in)
        par_prev = par

    print(f"Before: {tot_mismatch_before}, After: {tot_mismatch_after}") 
    if (tot_mismatch_after > tot_mismatch_before):
        return parallel_op
    else:
        return new_parallel_op

def print_report(layers_info, layer_par, n_variables, n_constraints, model_II, time_spent, generate_report_file, prj_root="/tmp"):
    with open(generate_report_file, "a+") as f:
        print("="*40, file=f)
        print("== Parallelization report", file=f)
        print("="*40, file=f)
        print(f"Number of variables: \t\t\t{n_variables}", file=f)
        print(f"Number of constraints:\t\t\t{n_constraints}", file=f)
        print(f"Time to solve: \t\t\t\t\t{time_spent:.2f}s", file=f)
        print(f"Initiation interval: \t\t\t{model_II}cc", file=f)
        print(f"Theorical throughput @ 200MHz: \t{1000000000.0 / (model_II * 5):.2f}FPS\n", file=f)
        table_data = []

        #header row
        header = ["Layer name", "ICH", "OCH", "OW", "ich_ops", "och_ops", "ow_ops", "DSPs", "PORTs", "Iter"]
        table_data.append(header)

        DSPs = 0
        PORTs = 0
        for layer in layers_info:
            pack = None
            ow_ops = layer_par[layer['name']][2]
            ich_ops = layer_par[layer['name']][1]
            och_ops = layer_par[layer['name']][0]
            bits = layer["weight_bits"][0]
            dsp = layer["kernel"] * och_ops * ich_ops * ow_ops

            op_per_dsp, _ = packing_feature((layer["weight_bits"][0], layer["act_bits"][0]), layer_par[layer['name']])
            pack = str(op_per_dsp)
            dsp = dsp // op_per_dsp

            if layer["type"] == "pool":
                dsp = 0

            iter = int(layer["total"] / (ich_ops * och_ops * ow_ops))
            weights = layer["ich"] * layer["och"] * layer["kernel"]
            if layer["depth"]:
                weights = layer["ich"] * layer["kernel"]

            port = compute_bram_layer(bits, weights,  och_ops * ich_ops * layer["kernel"])
            PORTs += port
            DSPs += dsp

            string_dsp = f"{dsp}"
            if pack:
                string_dsp += f" ({pack})"
    
            name = layer['name']
            if (layer["depth"]):
                name = f"{name} (depth)"
            
            row_data = [
                name,
                layer['ich'],
                layer['och'],
                layer['iw'],
                ich_ops,
                och_ops,
                ow_ops,
                string_dsp,
                port,
                iter
            ]

            table_data.append(row_data)

            if layer["merge_1x1"]:
                bits = layer["weight_bits"][1]
                dsp = och_ops * ich_ops * ow_ops
                iter = int(layer["total"] / (ich_ops * och_ops * ow_ops))
                port = math.ceil(bits * och_ops * ich_ops / 72)

                op_per_dsp, _ = packing_feature((layer["weight_bits"][1], layer["act_bits"][1]), layer_par[layer['name']])
                pack = str(op_per_dsp)
                dsp = dsp // op_per_dsp
                
                weights = layer["ich"] * layer["och"] * layer["kernel"]
                if layer["depth"]:
                    weights = layer["ich"] * layer["kernel"]

                port = compute_bram_layer(bits, weights, och_ops * ich_ops * layer["kernel"])

                PORTs += port
                DSPs += dsp
                
                string_dsp = f"{dsp}"
                if pack:
                    string_dsp += f" ({pack})"

                merge_row_data = [
                    f"{layer['name']} (merge 1x1)",
                    layer['ich'],
                    layer['och'],
                    layer['iw'],
                    ich_ops,
                    och_ops,
                    ow_ops,
                    string_dsp,
                    port,
                    iter
                ]

                table_data.append(merge_row_data)
        
        footer = ["Totals", "", "", "", "", "", "", DSPs, PORTs, ""]
        table_data.append(footer)

        # Print the tabulated data to the file
        f.write(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print("\n", file=f)

def write_parallelism(io_dict, model, parallel_ops):

    io_connect = extract_connections(model, io_dict)
    print(parallel_ops)

    for node_name, ops in parallel_ops.items():
        # The pool layer accept only parallelization over ich
        if (io_dict[node_name]["type"] == "pool"):
            io_dict[node_name]["ops"] = ops[1]
            io_dict[node_name]["ich_ops"] = ops[0]
            io_dict[node_name]["ow_ops"] = ops[2]
            io_dict[node_name]["ow_ops_out"] = ops[2]
            io_dict[node_name]["line_ops"] = ops[0]
        else:
            io_dict[node_name]["ops"] = ops[0]
            io_dict[node_name]["ich_ops"] = ops[1]
            io_dict[node_name]["ow_ops"] = ops[2]
            io_dict[node_name]["reuse"] = ops[2]
            io_dict[node_name]["ow_ops_out"] = ops[2]
            io_dict[node_name]["line_ops"] = ops[1]
            io_dict[node_name]["dp"] = False
            if io_dict[node_name]["merge_1x1"]:
                io_dict[node_name]["merge_node"]["ops"] = ops[0]
                io_dict[node_name]["merge_node"]["ich_ops"] = ops[1]
                io_dict[node_name]["merge_node"]["ow_ops"] = ops[2]
                io_dict[node_name]["merge_node"]["reuse"] = ops[2]


    # Propagating ops and ow_ops to input nodes to assign ops_out and ow_ops_out
    print("Starting backward propagation of ops and ow_ops")
    for name, node in io_dict.items():
        if "ops" in node:

            if (node["type"] == "produce"):
                continue

            input_node_name = prev_layers(io_dict, io_connect, name)[0]

            line_ops = node["line_ops"]
            ow_ops = node["ow_ops"]
            if "depth" in node:
                if node["depth"]:
                    line_ops = node["line_ops"]
            
            if "pool" == node["type"]:
                line_ops = node["ops"]
            
            if "depth" in io_dict[input_node_name]:
                if io_dict[input_node_name]["depth"]:
                    ops_out = io_dict[input_node_name]["ich_ops"]
                else:
                    ops_out = io_dict[input_node_name]["ops"]
            else:
                ops_out = io_dict[input_node_name]["ops"]

            # Assigning as ops_out of the previous layer the mcm between the ops of previous layer and current ich_ops
            io_dict[input_node_name]["ops_out"] = find_common_mult(ops_out, line_ops)
            print(f"\tAssigning to node {input_node_name} ops_out = {io_dict[input_node_name]['ops_out']} as mcm between {ops_out} and {line_ops}")
            # print(f'Could maybe done with {find_max_commond_div(ops_out, line_ops)}') 

    # Propagating ops and ow_ops to output nodes to assign in_ops and ow_ops_in
    print("Starting forward propagation of ops and ow_ops")
    for name, node in io_dict.items():
        if "ops" in node:
            output_name = io_dict[name]["output"][0]
            output_node_name = io_connect[output_name][1][0]
            if io_dict[output_node_name]["type"] != "consume":
                ops_out = node["ops_out"]
                io_dict[output_node_name]["in_ops"] = ops_out
            else:
                io_dict[name]["ops_out"] = node["ops"]

    # Input produce stream ops
    print("Assigning ops to input produce stream")
    print_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in print_layers:
            # check if the input tensor is produced by a produce_stream node
            input_name = node["input"][0]
            input_node_name = io_connect[input_name][0][0]
            if io_dict[input_node_name]["type"] == "produce":
                io_dict[input_node_name]["ops"] = node["line_ops"]
                io_dict[input_node_name]["ow_ops"] = 1
                io_dict[name]["in_ops"] = node["line_ops"]
        
    # Check for necessary bandwidth adjustements for the line buffer
    print("Starting line buffer adjustements over channels")
    line_buffer_layers = ["conv", "pool", "gemm"]
    for name, node in io_dict.items():
        if node["type"] in line_buffer_layers:
            if (node["in_ops"] % node["line_ops"]) != 0:
                node["adjust_line_buffer"] = True
                node["adjust_ops"] = find_common_mult(node["in_ops"],node["line_ops"])
                print(f"\tAdjusting line buffer for {name} over channels: {node['in_ops']} -> {node['adjust_ops']} -> {node['line_ops']} (in_ops -> adjust_ops -> line_ops)")
            else:
                node["adjust_ops"] = node["in_ops"]
                node["adjust_line_buffer"] = False

    # Propagating ops and ow_ops to output nodes to assign in_ops and ow_ops_in
    print("Starting adaptation of tensor width between layers")
    for name, node in io_dict.items():
        if "ow_ops" in node:
            
            output_node_name = next_layers(io_dict, io_connect, name)
            if output_node_name is None:
                continue
            else:
                output_node_name = output_node_name[0]
            
            if io_dict[output_node_name]["type"] != "consume":
                next_node = io_dict[output_node_name]
                print(f"Considering {name} -> {output_node_name}: {node['ow_ops']} -> {next_node['ow_ops']}")

                # Need to decrease ow
                if (node["ow_ops"] > next_node["ow_ops"]):

                    # Setting adjust down for next node line buffer
                    next_node["adjust_line_buffer"] = True
                    if (node['ow_ops'] % next_node['ow_ops'] != 0):

                        # Since the parallelism is not a multiple, need to scale up and then down
                        mcm_ow_ops = find_common_mult(node['ow_ops'], next_node['ow_ops'])
                        if (mcm_ow_ops > node['ow']):
                            print(f"Error: {name} -> {output_node_name} ow_ops_out {node['ow_ops_out']} > ow {node['ow']}")
                            exit(-1)
                        node["ow_ops_out"] = mcm_ow_ops
                        next_node["ow_ops_in"] = mcm_ow_ops
                    else:
                        next_node["ow_ops_in"] = node["ow_ops"]
                        node["ow_ops_out"] = node["ow_ops"]
                    print(f"Adjusting {name} -> {output_node_name}: {node['ow_ops']} -> {node['ow_ops_out']} -> {next_node['ow_ops_in']} -> {next_node['ow_ops']}")
                
                # Need to increase ow only if not multiple of ow_ops
                elif (node['ow_ops'] < next_node['ow_ops']):
                    
                    # Setting adjust down for next node line buffer
                    if (next_node['ow_ops'] % node['ow_ops'] != 0):
                        next_node["adjust_line_buffer"] = True
                        mcm_ow_ops = find_common_mult(node['ow_ops'], next_node['ow_ops'])
                        if (mcm_ow_ops > node['ow']):
                            print(f"Error: {name} -> {output_node_name} ow_ops_out {node['ow_ops_out']} > ow {node['ow']}")
                            exit(-1)
                        node["ow_ops_out"] = mcm_ow_ops
                        next_node["ow_ops_in"] = mcm_ow_ops
                        print(f"Adjusting {name} -> {output_node_name}: {node['ow_ops']} -> {node['ow_ops_out']} -> {next_node['ow_ops_in']} -> {next_node['ow_ops']}")
                    
                    elif (node["type"] == "produce"):
                        # Produce stream are fixed to ow_ops 1
                        next_node["adjust_line_buffer"] = True
                        node["ow_ops_out"] = node["ow_ops"]
                        next_node["ow_ops_in"] = node["ow_ops"]
                        print(f"Adjusting {name} -> {output_node_name}: {node['ow_ops']} -> {node['ow_ops_out']} -> {next_node['ow_ops_in']} -> {next_node['ow_ops']}")
                    else:
                        # Scaling up ow ops since they are multiple
                        next_node["ow_ops_in"] = next_node["ow_ops"]
                        node["ow_ops_out"] = next_node["ow_ops"] 
                
                else:
                    next_node["ow_ops_in"] = next_node["ow_ops"]
                    node["ow_ops_out"] = node["ow_ops"]

    # Adjusting pool layer, since in_ops must be a multiple of ops
    print("Adjusting pool layers")
    for name, node in io_dict.items():
        if node["type"] == "pool":
            output_name = node["output"][0]
            output_node_name = io_connect[output_name][1][0]
            
            # Assigning as ops the ops_out
            node["ops"] = node["ops_out"]

            # Right now node["ops"] is the ich_ops of the follwing layer.
            if (node["in_ops"] < node["ops_out"] or node["in_ops"] % node["ops_out"] != 0):
                node["ops"] = find_max_commond_div(node["in_ops"], node["ops_out"])
                print(f"\tAdjusting pool layer {name} over channels: {node['in_ops']} -> {node['ops']} -> {node['ops_out']} (in_ops -> ops -> ops_out)")

                # A pool cannot scale up over och_ops_out as convolution does, 
                # so we fix the lower-then-expected parallelization with a bandwidth adjust
                # or by scaling down with the line buffer 
                if (io_dict[output_node_name]["in_ops"] > node["ops"]):
                    io_dict[output_node_name]["adjust_line_buffer"] = True
                    io_dict[output_node_name]["adjust_ops"] = io_dict[output_node_name]["in_ops"]
                    io_dict[output_node_name]["line_ops"] = io_dict[output_node_name]["in_ops"]
                    print(f"\tAdjusting line buffer for {output_node_name} over channels: {io_dict[output_node_name]['in_ops']} -> {io_dict[output_node_name]['adjust_ops']} -> {io_dict[output_node_name]['line_ops']} (in_ops -> adjust_ops -> line_ops)")
                else:
                    io_dict[output_node_name]["line_ops"] = io_dict[output_node_name]["in_ops"]
                io_dict[output_node_name]["in_ops"] = node["ops"]

    print("Adjusting line buffer for the last layer")
    for name, node in io_dict.items():
        if node["type"] in line_buffer_layers:
            # check if the input tensor is produced by a produce_stream node
            first_input_node_name = prev_layers(io_dict, io_connect, name)[0]
            if io_dict[first_input_node_name]["type"] != "produce":
                if "adjust_line_buffer" in node.keys():
                    io_dict[first_input_node_name]["adjust_out"] = node["adjust_line_buffer"]
                else:
                    io_dict[first_input_node_name]["adjust_out"] = False

            first_output_node_name = next_layers(io_dict, io_connect, name)[0]
            if first_output_node_name == "consume_stream":
                io_dict[name]["adjust_out"] = False
    
    # Check for necessary bandwidth adjustements for the add stream
    print("Starting add stream adjustements")
    for name, node in io_dict.items():
        if node["type"] == "conv":
            if node["add"]:

                # obtain ops parallelism for the add stream
                skip_producer_name = prev_layers(io_dict, io_connect, name)
                skip_producer_name = skip_producer_name[1]
                if (io_dict[skip_producer_name]["merge_1x1"]):
                    add_ops = io_dict[skip_producer_name]["ops_out"]
                else:
                    # ich_ops because it's a forward
                    add_ops = io_dict[skip_producer_name]["ich_ops"]

                ow_ops = io_dict[skip_producer_name]["ow_ops_out"]
                node["add_ops"] = add_ops

                if (node["add_ops"] % node["ops"] != 0) or (node["ow_ops"] != ow_ops):
                    node["adjust_add"] = True
                    node["adjust_add_ops"] = find_common_mult(
                        node["ops"], node["add_ops"])
                    
                    node["adjust_add_ow_ops_in"] = ow_ops
                    if ow_ops % node["ow_ops"] != 0 and node["ow_ops"] % ow_ops != 0:
                        print(f"Error: {name} ow_ops ({node['ow_ops']}) is not a multiple/divisor of ow_ops_out of {add_node_name} ({ow_ops})")

                    print(f"\tAdjusting add stream for {name} over channels: {node['add_ops']} -> {node['adjust_add_ops']} -> {node['ops']} (add_ops -> adjust_add_ops -> ops)")
                    print(f"\tAdjusting add stream for {name} over width: {ow_ops} -> {node['adjust_add_ow_ops_in']} -> {node['ow_ops']} (ow_ops_out -> adjust_add_ow_ops_in -> ow_ops)")
                else:
                    node["adjust_add"] = False
            else:
                node["adjust_add"] = False
        
    print("Assigning packing")
    for name, node in io_dict.items():
        if node["type"] == "conv":
            _, packing = packing_feature((node["weight_quant"]["bits"], node["input_quant"]["bits"]), [node["ops"], node["ich_ops"], node["ow_ops"]])
            node["och_pack"] = packing[0]
            node["ow_pack"] = packing[1]
            print(f"Assigning packing for {name}: {packing}")

            if node["merge_1x1"]:
                node["merge_node"]["och_pack"] = packing[0]
                node["merge_node"]["ow_pack"] = packing[1]
                print(f"Assigning packing for {name} (merge): {packing}")
    return io_dict

def check_adjustments(io_dict, model):
    """ Check if the adjustments are consistent with the parallelization of the layers."""
    
    io_connect = extract_connections(model, io_dict)
    
    for name, node in io_dict.items():
        if "ops" in node:
            output_name = io_dict[name]["output"][0]
            output_node_name = io_connect[output_name][1][0]

            if io_dict[output_node_name]["type"] != "consume":
                # Checking conv layers
                if node["type"] == "conv" and node["depth"] == False:
                    if (node["ops"] > node["ops_out"]):
                        print(f"Error: {name} ops_out ({node['ops_out']}) is smaller than ops ({node['ops']}).")
                        return False
                    
                    if (node["ops_out"] % node["ops"] != 0):
                        print(f"Error: {name} ops_out ({node['ops_out']}) is not a multiple of ops ({node['ops']}).")
                        return False

                    if (node["ow_ops"] > node["ow_ops_out"]):
                        print(f"Error: {name} ow_ops_out ({node['ow_ops_out']}) is smaller than ow_ops ({node['ow_ops']}).")
                        return False

                    if (node["ow_ops_out"] % node["ow_ops"] != 0):
                        print(f"Error: {name} ow_ops ({node['ow_ops']}) is not a multiple of ow_ops_out ({node['ow_ops_out']}).")
                        return False
                    
                    if node["add"]:
                        if "adjust_add" in node and node["adjust_add"]:
                            if (node["adjust_add_ops"] % node["ops"] != 0):
                                print(f"Error: {name} adjust_add_ops ({node['adjust_add_ops']}) is not a multiple of ops ({node['ops']}).")
                                return False

                            if (node["adjust_add_ow_ops_in"] % node["ow_ops"] != 0 and node["ow_ops"] % node["adjust_add_ow_ops_in"] != 0):
                                print(f"Error: {name} adjust_add_ow_ops ({node['adjust_add_ow_ops_in']}) is not multiple of ow_ops ({node['ow_ops']}) and viceversa.")
                                return False
                        else:
                            if (node["add_ops"] % node["ops"] != 0):
                                print(f"Error: {name} add_ops ({node['add_ops']}) is not a multiple of ops ({node['ops']}).")
                                return False
                
                # Checking depth layers
                if node["type"] == "conv" and node["depth"] == True:
                    if (node["ops_out"] < node["ich_ops"]):
                        print(f"Error: {name} ops_out ({node['ops_out']}) is smaller than ich_ops ({node['ich_ops']}).")
                        return False
                    
                    if (node["ops_out"] % node["ich_ops"] != 0):
                        print(f"Error: {name} ops_out ({node['ops_out']}) is not a multiple of ich_ops ({node['ich_ops']}).")
                        return False

                    if (node["ow_ops"] > node["ow_ops_out"]):
                        print(f"Error: {name} ow_ops_out ({node['ow_ops_out']}) is smaller than ow_ops ({node['ow_ops']}).")
                        return False

                    if (node["ow_ops_out"] % node["ow_ops"] != 0):
                        print(f"Error: {name} ow_ops_out ({node['ow_ops_out']}) is not a multiple of ow_ops ({node['ow_ops']}).")
                        return False

                # Checking pool layers
                if node["type"] == "pool":
                    if (node["in_ops"] % node["ops"] != 0):
                        print(f"Error: {name} in_ops ({node['in_ops']}) is not a multiple of ops ({node['ops']}).")
                        return False

                    if (node["in_ops"] < node["ops"]):
                        print(f"Error: {name} in_ops ({node['in_ops']}) is smaller than ops ({node['ops']}).")
                        return False
                
                # Checking bandwidth adjust
                if "bandwidth_adjust" in node:
                    if (node["in_ops"] % node["adjust_ops"] != 0):
                        print(f"Error: bandwidth adjust before {name} in_ops ({node['in_ops']}) is not a multiple of adjust_ops ({node['adjust_ops']}).")
                        return False

                    if (node["ow_ops"] % node["ow_ops_in"] != 0 and node["ow_ops_in"] % node["ow_ops"] != 0):
                        print(f"Error: bandwidth adjust before {name} ow_ops ({node['ow_ops']}) not multiple of ow_ops_in ({node['ow_ops_in']}) and viceversa. ")
                        return False

                # Checking line buffer
                if "in_ops" in node and "line_ops" in node:
                    if (node["in_ops"] % node["line_ops"] != 0):
                        print(f"Error: line buffer before {name} in_ops ({node['in_ops']}) is not a multiple of line_ops ({node['line_ops']}).")
                        return False

    return True

def ilp(io_dict, off_chip_storage, model, file_name, board="ULTRA96v2", generate_report_file="tmp.rpt", prj_root="/tmp"):

    board_res = extract_board_info(board, prj_root)
    layers_info = layers_extractions(io_dict)

    NUM_PORTS = (board_res["bram"] + board_res["uram"])
    NUM_DSP = board_res["dsp"]
    NUM_DSP = 1800
    NUM_PORTS = int(NUM_PORTS * 0.85)

    valid_par_solutions = generate_architectures(layers_info, NUM_DSP)
    layer_par, model_II, n_variables, n_constraints, time_spent = parallelismILP(layers_info, valid_par_solutions, NUM_DSP, NUM_PORTS, prj_root=prj_root)
    layer_par = resourceILP(layers_info, model_II, valid_par_solutions, layer_par, NUM_DSP, NUM_PORTS, prj_root=prj_root)
    layer_par = balanceILP(layers_info, model_II, valid_par_solutions, layer_par, NUM_PORTS, prj_root=prj_root)
    layer_par = opt_steps(layers_info, layer_par, valid_par_solutions, prj_root=prj_root)
    io_dict = write_parallelism(io_dict, model, layer_par)
    print_report(layers_info, layer_par, n_variables, n_constraints, model_II, time_spent, generate_report_file, prj_root=prj_root)

    if (not check_adjustments(io_dict, model)):
        exit(-1)
    
    return io_dict
