import os
import sys
import pulp
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate
import json
import math
import numpy as np
from backend.ilp_utils import find_divisors
from backend.ilp_utils import find_range
from backend.ilp_utils import generate_valid_combinations
from backend.ilp_utils import generate_valid_parallelism
from backend.ilp_utils import find_higher_mult
from backend.ilp_utils import find_lower_mult
from backend.ilp_utils import find_common_mult
from backend.graph import extract_connections

def extract_board_info(board="ULTRA96v2", prj_root="/tmp"):
    """ Read the board json file and returns a dictionary with the available resources"""
    
    # Opening JSON file with board resources
    file_path = f"{prj_root}/../nn2fpga/boards/{board}.json"
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block 
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}
    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    
    return board_res

def layers_extractions(io_dict):
    """ Extracts the information about the layers from the io_dict and stores it in a dictionary.""" 
    
    # Find the highest number of computations done by a single convolution.

    total_computations = 0
    index = 0
    layers_info = []
    par_layers = ["conv", "pool"]
    for node_name, node_info in io_dict.items():
        if node_info["type"] in par_layers:
            
            kernel = node_info["fw"] * node_info["fh"]
            total_computations += node_info["total_log"]
            depth = False
            merge_1x1 = False
            bits = [0]
            och_1x1 = node_info["och"]

            if node_info["type"] == "conv":
                depth = node_info["depth"]
                merge_1x1 = node_info["merge_1x1"]
                bits = node_info["bits"]
                if (node_info["merge_1x1"]):
                    och_1x1 = node_info["och_1x1"]

            layers_info.append(
                {
                    "type": node_info["type"],
                    "name": node_name,
                    "total": 1 / node_info["total"],
                    "kernel": kernel,
                    "merge_1x1": merge_1x1,
                    "bits": bits,
                    "ich" : node_info["ich"],
                    "och" : node_info["och"],
                    "iw" : node_info["iw"],
                    "ow" : node_info["ow"],
                    "ih" : node_info["ih"],
                    "oh" : node_info["oh"],
                    "depth": depth,
                    "index": index,
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
        max_iw_par = layer["iw"]

        # Depthwise convolutions cannot be parallelized on output channel.
        if (layer["depth"] or layer["type"] == "pool"):
            max_och_par = 1

        if (layer["type"] == "pool"):
            max_iw_par = 1

        # In case of merged convolutions, take into account the gcd of the
        # maximum parallelization of och of the two. The transformation is
        # described in the graph optimization section of the paper. 
        if (layer["merge_1x1"]):
            max_och_par = math.gcd(layer["och"], layer["och_1x1"])

        # Clipping the maximum parallelization to the available DSPs, since it is not
        # possible that one layer uses all the DSPs, considering the packing
        op_clip = (NUM_DSP / layer["kernel"]) * 2 
        
        valid_par_solutions.append(generate_valid_combinations(
            och=max_och_par, ich=max_ich_par, iw=max_iw_par, iw_clip=4, op_clip=op_clip))
        
    return valid_par_solutions

def throughputILP(layers_info, worst_index, NUM_DSP, NUM_PORTS, packing=True, prj_root="/tmp"):

    valid_par_solutions = []
    for layer in layers_info:
        max_och_par = layer["och"]
        max_ich_par = layer["ich"]
        max_iw_par = layer["iw"]
        
        # Depthwise convolutions cannot be parallelized on output channel.
        if (layer["depth"] or layer["type"] == "pool"):
            max_och_par = 1

        if (layer["type"] == "pool"):
            max_iw_par = 1

        # In case of merged convolutions, take into account the gcd of the
        # maximum parallelization of och of the two. The transformation is
        # described in the graph optimization section of the paper. 
        if (layer["merge_1x1"]):
            max_och_par = math.gcd(layer["och"], layer["och_1x1"])

        # Clipping the maximum parallelization to the available DSPs, it is not
        # possible that one layer uses all the DSPs
        op_clip = NUM_DSP / layer["kernel"] 
        
        valid_par_solutions.append(list(generate_valid_parallelism(och=max_och_par, ich=max_ich_par, iw=max_iw_par, op_clip=op_clip)))

    valid_iter_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_iter_solutions.append([])
        layer_iter =  layer["total"]
        for single_par in layer_par:
            valid_iter_solutions[-1].append(layer_iter // single_par)

    # Creating a second dictionary in which merged convolution are splitted and
    # pool layers are removed to better compute DSPs and PORTs
    layers_info_unmerged = []
    for layer in [x for x in layers_info if x["type"] == "conv"]:
        layers_info_unmerged.append(layer.copy())
        layers_info_unmerged[-1]["bits"] = layer["bits"][0]
        layers_info_unmerged[-1]["merge_1x1"] = False
        if layer["merge_1x1"]:
            layers_info_unmerged.append(
                {
                    "name": layer["name"],
                    "total": layer["total"],
                    "kernel": 1,
                    "merge_1x1": layer["merge_1x1"],
                    "value": layer["value"],
                    "bits": layer["bits"][1],
                    "ich" : layer["ich"],
                    "och" : layer["och"],
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
            dsp_used = single_par * layer["kernel"]

            # Packing feature 
            if (packing):
                if (layer["bits"] == 8) and (single_par % 2 == 0):
                    dsp_used = dsp_used / 2
                elif (layer["bits"] == 4) and (single_par % 4 == 0):
                    dsp_used = dsp_used / 4

            valid_dsp_solutions[-1].append(dsp_used)
    
    ####### Problem formulated as ILP #######
    # Maximize throughput of worst layer
    prob = pulp.LpProblem("Parallel_ops", pulp.LpMinimize)
    
    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    slack_variable = pulp.LpVariable("slack", lowBound=0, cat="Integer")

    # Objective function: maximize the parallelization of the heaviest layer.
    # The decision variable "layer_binary_variables[worst_index][i]" select only
    # one combination of ich and och for the worst layer. The multiplication
    # between the two parameters represent the level of parallelization
    prob += (
        slack_variable,
        "Bottleneck_layer_parallelization"
    )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in [x["index"] for x in layers_info]:
        ones = [1] * len(layer_binary_variables[layer_index])
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )

    # Constraint: The total number of DSPs used to achieve the chosen
    # parallelization should be lower than the available ones. The number of
    # DSPs used for each layer is computed as filter_size * och_par * ich_par
    prob += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_DSP,
        f"DSP_constraint"
    )

    # Constraints: The throughtput of each layer should be equal or bigger to
    # the heaviest one. The throughtput of each layer is computed as the parallelism
    # over the total number of iterations:
    #
    #    par worst         par layer  
    #  -------------- <= --------------
    #    iter worst        iter layer   
    for layer_index in [x["index"] for x in layers_info]:
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= slack_variable,
            f"Throughtput_constraint_layer_{layer_index}"
        )
    
    # print(f"Variables: {sum([len(s) for s in valid_par_solutions])}")
    # prob.solve(PULP_CBC_CMD(msg=0))
    prob.solve(PULP_CBC_CMD(timeLimit=100))
    prob.writeLP(prj_root + "/parallel_ops1.lp")
    if (prob.status == pulp.LpStatusInfeasible):
        print("Problem unfeasible")
        exit(0)

    # Recovering the values of the paralellism for each layer from the binary variables.
    parallel_op = {}
    dsp = 0
    for i, layer in enumerate(valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]
                dsp += layers_info[i]["kernel"] * layer[s]

    print(parallel_op)
    print(dsp)
    print(int(layers_info[worst_index]["total"] / (parallel_op[layers_info[worst_index]["name"]])))
    exit(-1)

def parallelismILP(layers_info, valid_par_solutions, NUM_DSP, NUM_PORTS, packing=True, prj_root="/tmp"):
    """ Find the parallelization for each layer that maximize the throughput of the network."""

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
    for par_sol, layer in zip(valid_par_solutions, layers_info):
        valid_iter_linebuffer.append([])
        layer_iter =  layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in par_sol:
            valid_iter_linebuffer[-1].append(layer_iter // (single_par[1] * single_par[2]))
    
    valid_iter_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_iter_solutions.append([])
        layer_iter =  layer["total"]
        for single_par in layer_par:
            valid_iter_solutions[-1].append(layer_iter // np.prod(single_par))

    # Creating a second dictionary in which merged convolution are splitted and
    # pool layers are removed to better compute DSPs and PORTs
    layers_info_unmerged = []
    for layer in [x for x in layers_info if x["type"] == "conv"]:
        layers_info_unmerged.append(layer.copy())
        layers_info_unmerged[-1]["bits"] = layer["bits"][0]
        layers_info_unmerged[-1]["merge_1x1"] = False
        if layer["merge_1x1"]:
            layers_info_unmerged.append(
                {
                    "name": layer["name"],
                    "total": layer["total"],
                    "kernel": 1,
                    "merge_1x1": layer["merge_1x1"],
                    "bits": layer["bits"][1],
                    "ich" : layer["ich"],
                    "och" : layer["och"],
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
            dsp_used = np.prod(single_par) * layer["kernel"]

            # Packing feature 
            if (packing):
                if (layer["bits"] == 8) and (single_par[0] % 2 == 0 or single_par[2] % 2 == 0):
                    dsp_used = np.prod(single_par) * layer["kernel"] / 2
                elif (layer["bits"] == 4) and (single_par[0] % 2 == 0 and single_par[2] % 2 == 0):
                    dsp_used = np.prod(single_par) * layer["kernel"] / 4

            valid_dsp_solutions[-1].append(dsp_used)

    ####### Problem formulated as ILP #######
    # Maximize throughput of worst layer
    # prob = pulp.LpProblem("Parallel_ops", pulp.LpMaximize)
    prob = pulp.LpProblem("Parallel_ops", pulp.LpMinimize)
    
    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    slack_variable = pulp.LpVariable("slack", lowBound=0, cat="Integer")
    
    # Objective function: maximize the parallelization of the heaviest layer.
    # The decision variable "layer_binary_variables[worst_index][i]" select only
    # one combination of ich and och for the worst layer. The multiplication
    # between the two parameters represent the level of parallelization
    prob += (
        slack_variable,
        "Bottleneck_layer_parallelization"
    )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in [x["index"] for x in layers_info]:
        ones = [1] * len(layer_binary_variables[layer_index])
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )

    # Constraint: The total number of DSPs used to achieve the chosen
    # parallelization should be lower than the available ones. The number of
    # DSPs used for each layer is computed as filter_size * och_par * ich_par
    prob += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]) <= NUM_DSP,
        f"DSP_constraint"
    )

    # Constraint: The total number of memory ports used to achieve the chosen
    # parallelization should be lower than the available ones. The number of
    # ports used for each layer is computed as (filter_size * och_par * ich_par
    # * bits) / bandwidth_mem.
    valid_par_solutions_mem = [[x[0] * x[1] for x in layer_sol] for layer_sol in valid_par_solutions]
    prob += (
        pulp.lpSum([layer["kernel"] * layer["bits"] / 72 *
                    pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_par_solutions_mem[layer['index']]) for layer in layers_info_unmerged]) <= NUM_PORTS,
        f"PORTS_constraint"
    )
    
    # Constraints: The throughtput of each layer should be equal or bigger to
    # the heaviest one. The throughtput of each layer is computed as the parallelism
    # over the total number of iterations:
    for layer_index in [x["index"] for x in layers_info]:
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= slack_variable,
            f"Throughtput_constraint_layer_{layer_index}"
        )

    # Constraints: The iteration done by a line_buffer should be always less
    # than the one done by the heaviest layer, to avoid being a bottleneck
    for layer_index in [x["index"] for x in layers_info]:
        prob += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= slack_variable,
            f"Linebuffer_constraint_layer_{layer_index}"
        )
    
    # Constraints: To avoid bottlenecks the write/read bandwidth of consecutive
    # layers should be balanced. The write bandwidth is computed as (och_par *
    # ich_par) / (ich). The read bandwidth is computed as (och_par * ich_par) /
    # (och). For depthwise convolution the write bandwidth is ich_par.
    prob += (
        1.0 - 
        ( pulp.lpDot(layer_binary_variables[0].values(),
            valid_tot_par_solutions[0]) / layers_info[0]["och"] ) >= 0,
        f"ich_constraint_layer_0"
    )
    for layer_index in [x["index"] for x in layers_info[1:]]:
        if layers_info[layer_index - 1]["depth"] or layers_info[layer_index - 1]["type"] == "pool":
            prob += (
                pulp.lpSum(
                    [layer_binary_variables[layer_index - 1][i] * tuple[1] * tuple[2]
                    for i, tuple in enumerate(valid_par_solutions[layer_index - 1])]
                ) -
                ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_tot_par_solutions[layer_index]) / layers_info[layer_index]["och"] ) >= 0,
                f"ich_constraint_layer_{layer_index}"
            )
        else:
            prob += (
                ( pulp.lpDot(layer_binary_variables[layer_index - 1].values(),
                    valid_tot_par_solutions[layer_index - 1]) / layers_info[layer_index - 1]["ich"] ) - 
                ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_tot_par_solutions[layer_index]) / layers_info[layer_index]["och"] ) >= 0,
                f"ich_constraint_layer_{layer_index}"
            )
    

    # print(f"Variables: {sum([len(s) for s in valid_par_solutions])}")
    # prob.solve(PULP_CBC_CMD(msg=0))
    prob.solve(PULP_CBC_CMD(timeLimit=100))
    prob.writeLP(prj_root + "/parallel_ops1.lp")
    if (prob.status == pulp.LpStatusInfeasible):
        print("Problem unfeasible")
        exit(0)

    # Recovering the values of the paralellism for each layer from the binary variables.
    parallel_op = {}
    worst_layer_iter = 0
    for i, layer in enumerate(valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]
                if worst_layer_iter < layers_info[i]["total"] / np.prod(layer[s]):
                    worst_layer_iter = layers_info[i]["total"] / np.prod(layer[s])

    return parallel_op, worst_layer_iter

def resourceILP(layers_info, worst_layer_iter, valid_par_solutions, parallel_op, packing=True, prj_root="/tmp"):
    """ Given the throughput of the network, find the parallelization for each layer that minimize the resources usage."""
    
    # Creating a second dictionary in which merged convolution are splitted and
    # pool layers are removed to better compute DSPs and PORTs
    layers_info_unmerged = []
    for layer in [x for x in layers_info if x["type"] == "conv"]:
        layers_info_unmerged.append(layer.copy())
        layers_info_unmerged[-1]["bits"] = layer["bits"][0]
        layers_info_unmerged[-1]["merge_1x1"] = False
        if layer["merge_1x1"]:
            layers_info_unmerged.append(
                {
                    "name": layer["name"],
                    "total": layer["total"],
                    "kernel": 1,
                    "merge_1x1": layer["merge_1x1"],
                    "bits": layer["bits"][1],
                    "ich" : layer["ich"],
                    "och" : layer["och"],
                    "depth": layer["depth"],
                    "index": layer["index"]
                }
            )
    print(worst_layer_iter)

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
    for layer, layer_par in zip(layers_info, clamped_valid_par_solutions):
        valid_iter_solutions.append([])
        valid_iter_linebuffer.append([])
        layer_iter =  layer["total"]
        line_iter =  layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in layer_par:
            valid_iter_solutions[-1].append(layer_iter // np.prod(single_par))
            valid_iter_linebuffer[-1].append(line_iter // (single_par[1] * single_par[2]))
    
    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing 
    valid_dsp_solutions = []
    for layer in layers_info_unmerged:
        valid_dsp_solutions.append([])
        layer_par = clamped_valid_par_solutions[layer["index"]]
        for single_par in layer_par:
            dsp_used = np.prod(single_par) * layer["kernel"]
            if (packing):
                if (layer["bits"] == 8) and (single_par[0] % 2 == 0 or single_par[2] % 2 == 0):
                    dsp_used = np.prod(single_par) * layer["kernel"] // 2
                elif (layer["bits"] == 4) and (single_par[0] % 2 == 0 and single_par[2] % 2 == 0):
                    dsp_used = np.prod(single_par) * layer["kernel"] // 4
            valid_dsp_solutions[-1].append(dsp_used)
    
    # Minimize resource usage
    prob_min = pulp.LpProblem("Resource_usage", pulp.LpMinimize)
    
    # Objective function: minimize the DSPs required to run the whole network.
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]),
        f"DSP_constraint"
    )
    
    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in [x["index"] for x in layers_info]:
        ones = [1] * len(layer_binary_variables[layer_index])
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )
    
    # Constraints: The throughtput of each layer should be equal or bigger to
    # the heaviest one. The throughtput of each layer is computed as the parallelism
    # over the total number of iterations:
    for layer_index in [x["index"] for x in layers_info]:
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= worst_layer_iter,
            f"Throughtput_constraint_layer_{layer_index}"
        )
    
    # Constraints: The iteration done by a line_buffer should be always less
    # than the one done by the heaviest layer, to avoid being a bottleneck
    for layer_index in [x["index"] for x in layers_info]:
        prob_min += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= worst_layer_iter,
            f"linebuffer_constraint_layer_{layer_index}"
        )
    
    # Constraints: To avoid bottlenecks the write/read bandwidth of consecutive
    # layers should be balanced. The write bandwidth is computed as (och_par *
    # ich_par) / (ich). The read bandwidth is computed as (och_par * ich_par) /
    # (och). For depthwise convolution the write bandwidth is ich_par
    prob_min += (
        1.0 - 
        ( pulp.lpDot(layer_binary_variables[0].values(),
            valid_tot_par_solutions[0]) / layers_info[0]["och"] ) >= 0,
        f"ich_constraint_layer_0"
    )
    for layer_index in [x["index"] for x in layers_info[1:]]:
        if layers_info[layer_index - 1]["depth"] or layers_info[layer_index - 1]["type"] == "pool":
            prob_min += (
                pulp.lpSum(
                    [layer_binary_variables[layer_index - 1][i] * tuple[1] * tuple[2]
                    for i, tuple in enumerate(clamped_valid_par_solutions[layer_index - 1])]
                ) -
                ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_tot_par_solutions[layer_index]) / layers_info[layer_index]["och"] ) >= 0,
                f"ich_constraint_layer_{layer_index}"
            )
        else:
            prob_min += (
                ( pulp.lpDot(layer_binary_variables[layer_index - 1].values(),
                    valid_tot_par_solutions[layer_index - 1]) / layers_info[layer_index - 1]["ich"] ) - 
                ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_tot_par_solutions[layer_index]) / layers_info[layer_index]["och"] ) >= 0,
                f"ich_constraint_layer_{layer_index}"
            )
    
    # prob_min.solve()
    prob_min.solve(PULP_CBC_CMD(msg=0))
    if (prob_min.status == pulp.LpStatusInfeasible):
        print("Problem unfeasible")
        exit(0)
    
    parallel_op = {}
    for i, layer in enumerate(clamped_valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]
    
    return parallel_op

def parallel_ops_number(io_dict, packing=True, board="ULTRA96v2", prj_root="/tmp"):

    board_res = extract_board_info(board, prj_root)
    layers_info = layers_extractions(io_dict)

    NUM_PORTS = (board_res["bram"] + board_res["uram"])
    NUM_DSP = board_res["dsp"]
    # NUM_DSP = int(NUM_DSP * 1.1)
    # NUM_PORTS = 5000

    # throughputILP(layers_info, worst_index, NUM_DSP, NUM_PORTS, packing, prj_root=prj_root)
    valid_par_solutions = generate_architectures(layers_info, NUM_DSP)
    layer_par, worst_iter = parallelismILP(layers_info, valid_par_solutions, NUM_DSP, NUM_PORTS, packing, prj_root=prj_root)
    layer_par = resourceILP(layers_info, worst_iter, valid_par_solutions, layer_par, packing, prj_root=prj_root)

    ###### DEBUG ########
    pipeline_iterations = worst_iter
    for i, layer in enumerate(layers_info):
        ich_ops = layer_par[layer["name"]][1]
        ow_ops = layer_par[layer["name"]][2]
        ich = layer["ich"]
        input_dimension = ich * layer["iw"] * layer["ih"] // (ich_ops * ow_ops)
        print(f"{layer['ich']} lb_iter:{input_dimension}, conv_iter:{pipeline_iterations}, {input_dimension//pipeline_iterations}")


    with open(f"{prj_root}/{board}_par.rpt", "w") as f:
        table_data = []

        #header row
        header = ["Layer name", "ICH", "OCH", "OW", "ich_ops", "och_ops", "ow_ops", "DSPs", "PORTs", "Iter"]
        table_data.append(header)

        DSPs = 0
        PORTs = 0
        for layer in layers_info:
            pack = False
            ow_ops = layer_par[layer['name']][2]
            ich_ops = layer_par[layer['name']][1]
            och_ops = layer_par[layer['name']][0]
            bits = layer["bits"][0]
            dsp = layer["kernel"] * och_ops * ich_ops * ow_ops

            if packing and (och_ops % 2 == 0 or ow_ops % 2 == 0):
                pack = True
                dsp = dsp // 2

            if layer["type"] == "pool":
                dsp = 0

            iter = int(layer["total"] / (ich_ops * och_ops * ow_ops))
            port = math.ceil(layer["kernel"] * bits * och_ops * ich_ops / 72)

            PORTs += port
            DSPs += dsp

            string_dsp = f"{dsp}"
            if pack:
                string_dsp += " (P)"
    
            row_data = [
                layer['name'],
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
                bits = layer["bits"][1]
                dsp = och_ops * ich_ops * ow_ops
                iter = int(layer["total"] / (ich_ops * och_ops * ow_ops))
                port = math.ceil(bits * och_ops * ich_ops / 72)
                
                if packing and (och_ops % 2 == 0 or ow_ops % 2 == 0):
                    pack = True
                    dsp = dsp // 2

                PORTs += port
                DSPs += dsp
                
                string_dsp = f"{dsp}"
                if pack:
                    string_dsp += " (P)"

                merge_row_data = [
                    f"{layer['name']}_merge",
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
        
        # for layer in layers_info:
        #     if layer["depth"] or layer["type"] == "pool":
        #         Bw = layer_par[layer['name']][1] * layer_par[layer['name']][2]
        #         Br = np.prod(layer_par[layer['name']]) / layer['och']
        #         print(f"{layer['name']} Br = {Br:.3f} Bw = {Bw:.3f}", file=f)
        #     else:
        #         Bw = np.prod(layer_par[layer['name']]) / layer['ich']
        #         Br = np.prod(layer_par[layer['name']]) / layer['och']
        #         print(f"{layer['name']} Br = {Br:.3f} Bw = {Bw:.3f}", file=f)
    ###### END DEBUG ########

    return layer_par

def ilp(io_dict, off_chip_storage, model, board="ULTRA96v2", packing=True, prj_root="/tmp"):

    parallel_ops = parallel_ops_number(io_dict, packing=True, board=board, prj_root=prj_root)
    io_connect = extract_connections(model, io_dict)

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
            io_dict[node_name]["ow_ops_out"] = ops[2]
            io_dict[node_name]["ops_1x1"] = ops[0]
            io_dict[node_name]["line_ops"] = ops[1]
            io_dict[node_name]["dp"] = False

    # Avoid the line buffer to become a bottleneck when there is a mismatch
    # between its iterations and the number of operation of the served
    # convolution. The padding reduce the bandwidth. Often for a depthwise.
    line_buffer_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in line_buffer_layers:
            ich_ops = node["ich_ops"]
            och_ops = node["ops"]
            ich = node["ich"]
            och = node["och"]
            if node["type"] == "conv":
                if node["depth"] == 1:
                    och = 1
            input_dimension = (node["ich"] * node["iw"] * node["ih"] // (ich_ops * node["ow_ops"]))
            pipeline_iterations = (och / och_ops) * (ich / ich_ops) * (node["ow"] / node["ow_ops"]) * node["oh"]
            # TODO: check that after initializing line_ops with ich_ops everything works
            # line_ops = int(input_dimension // pipeline_iterations)
            # if line_ops == 0:
            #     line_ops = 1
            # print(name, input_dimension, pipeline_iterations, node["ich"], node["ich_ops"])
            # if node["ich_ops"] < line_ops:
            #     io_dict[name]["line_ops"] = line_ops
            #     print(f"#### Changing line_ops for {name} to {line_ops} to avoid bottleneck")
            # else:
            #     io_dict[name]["line_ops"] = ich_ops
    
    #TODO: Avoiding cycling twice because of pool layers
    # for node_name, ops in parallel_ops.items():
    #     output_name = io_dict[node_name]["output"][0]
    #     output_node_name = io_connect[output_name][1][0]
    #     if output_node_name != "consume_stream":
    #         print(f"1 - I'm writing for {output_node_name} in_ops {ops}")
    #         io_dict[output_node_name]["in_ops"] = ops

    print("#### Propagating ops and ow_ops to input nodes")
    for name, node in io_dict.items():
        if "ops" in node:
            input_name = io_dict[name]["input"][0]
            print(f"Propagating ops and ow_ops to {input_name}")

            # Check if the input tensor is an input of the model
            # If it is, skip it
            is_model_input = False
            for input_graph_name in model.graph.input:
                input_graph_name = input_graph_name.name.replace(".", "_")
                if input_graph_name == input_name:
                    is_model_input = True
            
            if is_model_input:
                continue

            input_node_name = io_connect[input_name][0][0]
            ops = node["ops"]
            ow_ops = node["ow_ops"]
            if "depth" in node:
                if node["depth"]:
                    ops = node["ich_ops"]
            io_dict[input_node_name]["out_ops"] = ops
            if ow_ops > io_dict[input_node_name]["ow_ops_out"]:
                print(f"Updating for {input_node_name} ow_ops {ow_ops}")
                io_dict[input_node_name]["ow_ops_out"] = ow_ops
            print(f"Node {input_node_name} ow_ops_out {io_dict[input_node_name]['ow_ops_out']} ow_ops {ow_ops}")

    for name, node in io_dict.items():
        if "ops" in node:
            output_name = io_dict[name]["output"][0]
            output_node_name = io_connect[output_name][1][0]
            ops = node["ops"]
            ow_ops = node["ow_ops_out"]
            if "depth" in node:
                if node["depth"]:
                    ops = node["ich_ops"]
            if output_node_name != "consume_stream":
                io_dict[output_node_name]["in_ops"] = ops
                io_dict[output_node_name]["ow_ops_in"] = ow_ops

    print("##################################################")
    # for name, node in io_dict.items():
    #     if "in_ops" not in node:
    #         node["in_ops"] = 1

    #     if "conv" in node["type"]:
    #         # FIX: adding this check to avoid problems in merged pipelines
    #         # with same inputs but different output channels
    #         if node["merge_1x1"]:
    #             if node["och_1x1"] < node["ops"]:
    #                 node["ops_1x1"] = node["och_1x1"]
    #                 # Propagate to merged weights and bias parameters
    #                 start_index = 2
    #                 if "has_bias" in node.keys():
    #                     if node["has_bias"]:
    #                         start_index = 3

    #                 for i in range(start_index, len(node["input"])):
    #                     input_name = node["input"][i]
    #                     input_node_name = io_connect[input_name][0][0]
    #                     print(f'{io_dict[input_node_name]["ops"]} = {node["ops_1x1"]}')
    #                     print(f'{io_dict[input_node_name]["och"]} = {node["och_1x1"]}')
    #                     io_dict[input_node_name]["ops"] = node["ops_1x1"]
    #                     io_dict[input_node_name]["och"] = node["och_1x1"]
    #             else:
    #                 node["ops_1x1"] = node["ops"]
    #                 print(f'2.5 - {name} ops_1x1 merge = before {node["ops_1x1"]} now {node["ops"]}')
    #         else:
    #             node["ops_1x1"] = node["ops"]
    #             print(f'2.5 - {name} ops_1x1 merge = before {node["ops_1x1"]} now {node["ops"]}')
    
    # Avoiding line buffer to be the bottleneck in case of strides
    # for name, node in io_dict.items():
    #     # print ops and ich_ops for conv and pool layers
    #     if node["type"] == "conv":
    #         output_name = node["output"][0]
    #         output_node_name = io_connect[output_name][1][0]
    #         if output_node_name != "consume_stream":
    #             if node["depth"]:
    #                 io_dict[output_node_name]["in_ops"] = node["ich_ops"]
    #             else:
    #                 print(f"3 - I'm writing for {output_node_name} in_ops = before {io_dict[output_node_name]['in_ops']} now {node['ops']}")
    #                 io_dict[output_node_name]["in_ops"] = node["ops"]

    # Input produce stream ops
    print_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in print_layers:
            # check if the input tensor is produced by a produce_stream node
            print(f'{name}: {node["in_ops"]} {node["ow_ops_in"]} {node["ich_ops"]} {node["ow_ops"]} {node["ops"]}')
            input_name = node["input"][0]
            input_node_name = io_connect[input_name][0][0]
            if io_dict[input_node_name]["type"] == "produce":
                io_dict[input_node_name]["ops"] = node["line_ops"]
                io_dict[name]["in_ops"] = node["line_ops"]
                # print(f"3 - I'm writing for {input_node_name} ops {node['line_ops']}")
        
    # Check for necessary bandwidth adjustements for the line buffer
    # line_buffer_layers = ["conv", "pool"]
    # for name, node in io_dict.items():
    #     if node["type"] in line_buffer_layers:
    #         print(name, node["in_ops"], node["line_ops"])
    #         if (node["in_ops"] % node["line_ops"]) != 0:
    #             node["adjust_line_buffer"] = True
    #             node["adjust_ops"] = find_common_mult(node["in_ops"],node["line_ops"])
    #             print("#### Found line buffer read/write rate for", name, "read", node["in_ops"], "write", node["line_ops"], "to avoid bottleneck")
    #             print("#### Balancing line buffer for", name, "from", node["in_ops"], "to", node["adjust_ops"], "to avoid bottleneck")
    #         else:
    #             node["adjust_line_buffer"] = False
    line_buffer_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in line_buffer_layers:
            print(name, node["in_ops"], node["line_ops"])
            if (node["in_ops"] % node["line_ops"]) != 0:
                node["adjust_line_buffer"] = True
                node["adjust_ops"] = find_common_mult(node["in_ops"],node["line_ops"])
                print("#### Found line buffer read/write rate for", name, "read", node["in_ops"], "write", node["line_ops"], "to avoid bottleneck")
                print("#### Balancing line buffer for", name, "from", node["in_ops"], "to", node["adjust_ops"], "to avoid bottleneck")
            else:
                node["adjust_ops"] = node["in_ops"]
                node["adjust_line_buffer"] = False

            if (node['ow_ops'] < node['ow_ops_in']):
                node["adjust_line_buffer"] = True
                print(f"Insert bandwidth_adjust from {node['ow_ops_in']} to {node['ow_ops']}")
    
    # Check for necessary bandwidth adjustements for the add stream
    for name, node in io_dict.items():
        if node["type"] == "conv":
            if node["add"]:

                # obtain ops parallelism for the add stream
                if (node["add"]):
                    if (node["has_bias"]):
                        add_name = node["input"][3]
                    else:
                        add_name = node["input"][2]

                add_node_name = io_connect[add_name][0][0]
                if (io_dict[add_node_name]["merge_1x1"]):
                    add_ops = io_dict[add_node_name]["ops"]
                    ow_ops = io_dict[add_node_name]["ow_ops_out"]
                else:
                    add_ops = io_dict[add_node_name]["ich_ops"]
                    ow_ops = io_dict[add_node_name]["ow_ops_out"]
                node["add_ops"] = add_ops

                print("#### Add tensor read/write rate for", name, "read", node["ich_ops"], "write", node["add_ops"])
                if (node["ich_ops"] > node["add_ops"]) or (node["ow_ops"] != ow_ops):
                    node["adjust_add"] = True
                    node["adjust_add_ops"] = find_common_mult(node["ich_ops"],node["add_ops"])
                    print("#### Found add tensor read/write rate for", name, "read", node["ich_ops"], "write", node["add_ops"], "to avoid bottleneck")
                    print("#### Balancing add tensor for", name, "from", node["add_ops"], "to", node["adjust_add_ops"], "to avoid bottleneck")
                else:
                    node["adjust_add"] = False
            else:
                node["adjust_add"] = False
    
    # print_layers = ["conv", "pool"]
    # for name, node in io_dict.items():
    #     if node["type"] in print_layers:
    #         print(f'{name} -> och: {node["och"]} par {node["ops"]}, ich: {node["ich"]} par {node["ich_ops"]}')
            
    #         # Final layer of classification
    #         if "is_1x1" in node.keys():
    #             if node["is_1x1"]:
    #                 continue
    #         node["ow_ops"] = 2

    return io_dict
