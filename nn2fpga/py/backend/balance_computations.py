import os
import sys
import pulp
from pulp.apis import PULP_CBC_CMD
import json
import math
from backend.ilp_utils import find_divisors
from backend.ilp_utils import find_range
from backend.ilp_utils import generate_valid_combinations
from backend.ilp_utils import find_higher_mult
from backend.ilp_utils import find_lower_mult
from backend.ilp_utils import find_common_mult
from backend.graph import extract_connections

def parallel_ops_number(io_dict, board="ULTRA96v2", prj_root="/tmp"):
    
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

    NUM_PORTS = (board_res["bram"] + board_res["uram"])
    NUM_DSP = board_res["dsp"]
    
    # Find the highest number of computations done by a single convolution.
    max_total = 1
    worst_index = 0
    iter = 0
    print_layers = ["conv", "pool"]
    for node_name, node_info in io_dict.items():
        if node_info["type"] in print_layers:
            print(node_info["total"])
            if max_total > node_info["total"]:
                max_total = node_info["total"]
                worst_index = iter
            iter +=1

    total_computations = 0
    index = 0
    layers_info = []
    for node_name, node_info in io_dict.items():
        if node_info["type"] in print_layers:
            
            kernel = node_info["fw"] * node_info["fh"]
            value = node_info["total"] / max_total
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
                    "total": node_info["total"],
                    "kernel": kernel,
                    "merge_1x1": merge_1x1,
                    "value": value,
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

    # Generating all the valid combinations of ich_par and och_par for each layer.
    valid_par_solutions = []
    for layer in layers_info:
        max_och_par = layer["och"]
        max_ich_par = layer["ich"]
        
        # Depthwise parallelization cannot be parallelized on output channel.
        if (layer["depth"] or layer["type"] == "pool"):
            max_och_par = 1

        # In case of merged convolutions, take into account the gcd of the
        # maximum parallelization of och of the two. The transformation is
        # described in the graph optimization section of the paper. 
        if (layer["merge_1x1"]):
            max_och_par = math.gcd(layer["och"], layer["och_1x1"])

        valid_par_solutions.append(generate_valid_combinations(max_och_par, max_ich_par))

    # valid_tot_par_solutions stores the total parallelization for each valid
    # solution and it is useful to use lpDot to compute the parallelization
    # chosen for a layer
    valid_tot_par_solutions = []
    for i, par_sol in enumerate(valid_par_solutions):
        valid_tot_par_solutions.append([])
        for single_par in par_sol:
            valid_tot_par_solutions[i].append(single_par[0] * single_par[1])

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

    ####### Problem formulated as ILP #######
    # Maximize throughput of worst layer
    prob = pulp.LpProblem("Parallel_ops", pulp.LpMaximize)
    
    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    # Objective function: maximize the parallelization of the heaviest layer.
    # The decision variable "layer_binary_variables[worst_index][i]" select only
    # one combination of ich and och for the worst layer. The multiplication
    # between the two parameters represent the level of parallelization
    prob += (
        pulp.lpDot(layer_binary_variables[worst_index].values(), valid_tot_par_solutions[worst_index]),
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
        pulp.lpSum([layer["kernel"] * pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_tot_par_solutions[layer['index']]) for layer in layers_info_unmerged]) <= NUM_DSP,
        f"DSP_constraint"
    )

    # Constraint: The total number of memory ports used to achieve the chosen
    # parallelization should be lower than the available ones. The number of
    # ports used for each layer is computed as (filter_size * och_par * ich_par
    # * bits) / bandwidth_mem.
    prob += (
        pulp.lpSum([layer["kernel"] * layer["bits"] / 72 *
                    pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    valid_tot_par_solutions[layer['index']]) for layer in layers_info_unmerged]) <= NUM_PORTS,
        f"PORTS_constraint"
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
                    valid_tot_par_solutions[layer_index]) * layers_info[layer_index]["value"] -
            pulp.lpDot(layer_binary_variables[worst_index].values(),
                    valid_tot_par_solutions[worst_index]) >= 0,
            f"Throughtput_constraint_layer_{layer_index}"
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
                    [layer_binary_variables[layer_index - 1][i] * tuple[1]
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

    prob.solve(PULP_CBC_CMD(msg=0))
    prob.writeLP(prj_root + "/parallel_ops1.lp")
    if (prob.status == pulp.LpStatusInfeasible):
        print("Problem unfeasible")
        exit(0)

    # Recovering the values of the paralellism for each layer from the binary variables.
    parallel_op = {}
    for i, layer in enumerate(valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]

    # Retriving only the parallelism combinations for lower throughput to save resources in fast layers
    layer_binary_variables = []
    clamped_valid_par_solutions = []
    clamped_valid_tot_par_solutions = []
    for i, solution_set in enumerate(valid_par_solutions):
        clamped_valid_par_solutions.append([])
        clamped_valid_tot_par_solutions.append([])
        for combination in solution_set:
            tot_par = combination[0] * combination[1]
            chosen_par = parallel_op[layers_info[i]['name']][0] * parallel_op[layers_info[i]['name']][1]
            if  (i != worst_index):
                if (tot_par <= chosen_par):
                    clamped_valid_par_solutions[i].append(combination)
                    clamped_valid_tot_par_solutions[i].append(combination[0] * combination[1])
            else:
                if (tot_par == chosen_par):
                    clamped_valid_par_solutions[i].append(combination)
                    clamped_valid_tot_par_solutions[i].append(combination[0] * combination[1])

        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(clamped_valid_par_solutions[i])), cat="Binary"))
    
    # Minimize resource usage
    prob_min = pulp.LpProblem("Resource_usage", pulp.LpMinimize)
    
    # Objective function: minimize the DSPs required to run the whole network.
    prob_min += (
        pulp.lpSum([layer["kernel"] * pulp.lpDot(layer_binary_variables[layer['index']].values(),
                    clamped_valid_tot_par_solutions[layer['index']]) for layer in layers_info_unmerged]),
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
    #
    #    par worst         par layer  
    #  -------------- <= --------------
    #    iter worst        iter layer   
    for layer_index in [x["index"] for x in layers_info]:
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    clamped_valid_tot_par_solutions[layer_index]) * layers_info[layer_index]["value"] -
            pulp.lpDot(layer_binary_variables[worst_index].values(),
                    clamped_valid_tot_par_solutions[worst_index]) >= 0,
            f"Throughtput_constraint_layer_{layer_index}"
        )
    
    # Constraints: To avoid bottlenecks the write/read bandwidth of consecutive
    # layers should be balanced. The write bandwidth is computed as (och_par *
    # ich_par) / (ich). The read bandwidth is computed as (och_par * ich_par) /
    # (och). For depthwise convolution the write bandwidth is ich_par
    prob_min += (
        1.0 - 
        ( pulp.lpDot(layer_binary_variables[0].values(),
            clamped_valid_tot_par_solutions[0]) / layers_info[0]["och"] ) >= 0,
        f"ich_constraint_layer_0"
    )
    for layer_index in [x["index"] for x in layers_info[1:]]:
        if layers_info[layer_index - 1]["depth"] or layers_info[layer_index - 1]["type"] == "pool":
            prob_min += (
                pulp.lpSum(
                    [layer_binary_variables[layer_index - 1][i] * tuple[1]
                    for i, tuple in enumerate(clamped_valid_par_solutions[layer_index - 1])]
                ) -
                ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                    clamped_valid_tot_par_solutions[layer_index]) / layers_info[layer_index]["och"] ) >= 0,
                f"ich_constraint_layer_{layer_index}"
            )
        else:
            prob_min += (
                ( pulp.lpDot(layer_binary_variables[layer_index - 1].values(),
                    clamped_valid_tot_par_solutions[layer_index - 1]) / layers_info[layer_index - 1]["ich"] ) - 
                ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                    clamped_valid_tot_par_solutions[layer_index]) / layers_info[layer_index]["och"] ) >= 0,
                f"ich_constraint_layer_{layer_index}"
            )
    
    prob_min.solve(PULP_CBC_CMD(msg=0))
    
    parallel_op = {}
    for i, layer in enumerate(clamped_valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]

    ich_ops = parallel_op[layers_info[worst_index]["name"]][1]
    och_ops = parallel_op[layers_info[worst_index]["name"]][0]
    ich = layers_info[worst_index]["ich"]
    och = layers_info[worst_index]["och"]
    oh = layers_info[worst_index]["oh"]
    ow = layers_info[worst_index]["ow"]
    ow_ops = 1
    pipeline_iterations = (och / och_ops) * (ich / ich_ops) * ow * oh
    for i, layer in enumerate(layers_info):
        ich_ops = parallel_op[layer["name"]][1]
        ich = layer["ich"]
        ow_ops = 1
        input_dimension = ich * layer["iw"] * layer["ih"] // (ich_ops * ow_ops)
        print(f"{layer['ich']} lb_iter:{input_dimension}, conv_iter:{pipeline_iterations}, {input_dimension//pipeline_iterations}")

    # Reporting the parallelization chosen for each layer.
    with open(f"{prj_root}/{board}_par.rpt", "w") as f:
        print(f"Worst layer: {layers_info[worst_index]['name']}", file = f)
        DSPs = 0
        PORTs = 0
        for layer in layers_info:
            ich_ops = parallel_op[layer['name']][1]
            och_ops = parallel_op[layer['name']][0]
            bits = layer["bits"][0]
            dsp = layer["kernel"] * och_ops * ich_ops
            if layer["type"] == "pool":
                dsp = 0
            iter = int(1 / (ich_ops * och_ops * layer["total"]))
            port = math.ceil(layer["kernel"] * bits * och_ops * ich_ops / 72)
            PORTs += port
            DSPs += dsp
            print(f"{layer['name']} [{layer['ich']}][{layer['och']}] -> ich_ops: {ich_ops}, och_ops: {och_ops}, DSPs: {dsp}, PORTs: {port}, Iter: {iter}", file=f)
            if (layer["merge_1x1"]):
                bits = layer["bits"][1]
                dsp = och_ops * ich_ops
                iter = int(1 / (ich_ops * och_ops * layer["total"]))
                port = math.ceil(bits * och_ops * ich_ops / 72)
                PORTs += port
                DSPs += dsp
                print(f"{layer['name']}_merge [{layer['ich']}][{layer['och']}] -> ich_ops: {ich_ops}, och_ops: {och_ops}, DSPs: {dsp}, PORTs: {port}, Iter: {iter}", file=f)
        print(f"Totals: DSPs {DSPs}, Ports: {PORTs}", file=f)
        for i, layer in enumerate(layers_info):
            if layer["depth"] or layer["type"] == "pool":
                Bw = parallel_op[layer['name']][0]
                Br = (parallel_op[layer['name']][0] * parallel_op[layer['name']][1]) / layer['och']
                print(f"{layer['name']} Br = {Br} Bw = {Bw}", file=f)
            else:
                Bw = (parallel_op[layer['name']][0] * parallel_op[layer['name']][1]) / layer['ich']
                Br = (parallel_op[layer['name']][0] * parallel_op[layer['name']][1]) / layer['och']
                print(f"{layer['name']} Br = {Br} Bw = {Bw}", file=f)

    
    return parallel_op

def ilp(io_dict, off_chip_storage, model, board="ULTRA96v2", double_packing=True, prj_root="/tmp"):

    parallel_ops = parallel_ops_number(io_dict, board, prj_root=prj_root)
    io_connect = extract_connections(model, io_dict)

    for node_name, ops in parallel_ops.items():
        # The pool layer accept only parallelization over ich
        if (io_dict[node_name]["type"] == "pool"):
            io_dict[node_name]["ops"] = ops[1]
            io_dict[node_name]["ich_ops"] = ops[0]
        else:
            io_dict[node_name]["ops"] = ops[0]
            io_dict[node_name]["ich_ops"] = ops[1]
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
            input_dimension = node["ich"] * node["iw"] * node["ih"] // ich_ops
            pipeline_iterations = (och / och_ops) * (ich / ich_ops) * node["ow"] * node["oh"]
            line_ops = int(input_dimension // pipeline_iterations)
            if line_ops == 0:
                line_ops = 1
            print(name, line_ops, node["ich"], node["ich_ops"])
            if node["ich_ops"] < line_ops:
                io_dict[name]["line_ops"] = line_ops
                print(f"#### Changing line_ops for {name} to {line_ops} to avoid bottleneck")
            else:
                io_dict[name]["line_ops"] = ich_ops
    
    #TODO: Avoiding cycling twice because of pool layers
    for node_name, ops in parallel_ops.items():
        output_name = io_dict[node_name]["output"][0]
        output_node_name = io_connect[output_name][1][0]
        if output_node_name != "consume_stream":
            io_dict[output_node_name]["in_ops"] = ops

    for name, node in io_dict.items():
        if "ops" in node:
            output_name = io_dict[name]["output"][0]
            output_node_name = io_connect[output_name][1][0]
            ops = node["ops"]
            if "depth" in node:
                if node["depth"]:
                    ops = node["ich_ops"]
            if output_node_name != "consume_stream":
                io_dict[output_node_name]["in_ops"] = ops

    for name, node in io_dict.items():
        if "in_ops" not in node:
            node["in_ops"] = 1

        if "conv" in node["type"]:
            # FIX: adding this check to avoid problems in merged pipelines
            # with same inputs but different output channels
            if node["merge_1x1"]:
                if node["och_1x1"] < node["ops"]:
                    node["ops_1x1"] = node["och_1x1"]
                    # Propagate to merged weights and bias parameters
                    start_index = 2
                    if "has_bias" in node.keys():
                        if node["has_bias"]:
                            start_index = 3

                    for i in range(start_index, len(node["input"])):
                        input_name = node["input"][i]
                        input_node_name = io_connect[input_name][0][0]
                        io_dict[input_node_name]["ops"] = node["ops_1x1"]
                        io_dict[input_node_name]["och"] = node["och_1x1"]
                else:
                    node["ops_1x1"] = node["ops"]
            else:
                node["ops_1x1"] = node["ops"]
    
    # Avoiding line buffer to be the bottleneck in case of strides
    for name, node in io_dict.items():
        # print ops and ich_ops for conv and pool layers
        if node["type"] == "conv":
            output_name = node["output"][0]
            output_node_name = io_connect[output_name][1][0]
            if output_node_name != "consume_stream":
                if node["depth"]:
                    io_dict[output_node_name]["in_ops"] = node["ich_ops"]
                else:
                    io_dict[output_node_name]["in_ops"] = node["ops"]

    # Input produce stream ops
    print_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in print_layers:
            # check if the input tensor is produced by a produce_stream node
            input_name = node["input"][0]
            input_node_name = io_connect[input_name][0][0]
            if io_dict[input_node_name]["type"] == "produce":
                io_dict[input_node_name]["ops"] = node["line_ops"]
                io_dict[name]["in_ops"] = node["line_ops"]
        
    # Check for necessary bandwidth adjustements for the line buffer
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
                node["adjust_line_buffer"] = False
    
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
                else:
                    add_ops = io_dict[add_node_name]["ich_ops"]
                node["add_ops"] = add_ops

                print("#### Add tensor read/write rate for", name, "read", node["ich_ops"], "write", node["add_ops"])
                if (node["ich_ops"] > node["add_ops"]):
                    node["adjust_add"] = True
                    node["adjust_add_ops"] = find_common_mult(node["ich_ops"],node["add_ops"])
                    print("#### Found add tensor read/write rate for", name, "read", node["ich_ops"], "write", node["add_ops"], "to avoid bottleneck")
                    print("#### Balancing add tensor for", name, "from", node["add_ops"], "to", node["adjust_add_ops"], "to avoid bottleneck")
                else:
                    node["adjust_add"] = False
            else:
                node["adjust_add"] = False
    
    print_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in print_layers:
            print(f'{name} -> och: {node["och"]} par {node["ops"]}, ich: {node["ich"]} par {node["ich_ops"]}')
            # Testing resnet8
            if "is_1x1" in node.keys():
                if node["is_1x1"]:
                    continue
            node["ow_ops"] = 1

    return io_dict
