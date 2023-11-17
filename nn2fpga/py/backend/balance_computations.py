import os
import sys
import pulp
import json
from pulp import const
import math
from backend.ilp_utils import find_divisors
from backend.ilp_utils import find_range
from backend.ilp_utils import find_higher_mult
from backend.ilp_utils import find_lower_mult
from backend.ilp_utils import find_common_mult
from backend.graph import extract_connections

def parallel_ops_number(layers_info, worst_index, board="ULTRA96v2", prj_root="/tmp"):

    if (board == "ULTRA96v2"):
        NUM_DSP = 400
    elif (board == "ZC706"):
        NUM_DSP = 900
    elif (board == "PYNQ"):
        NUM_DSP = 220
    elif (board == "KRIA"):
        NUM_DSP = 1500
        # NUM_DSP = 3000
    elif (board == "ZCU102"):
        NUM_DSP = 2000
    elif (board == "U280"):
        NUM_DSP = 2492
        # NUM_DSP = 9024
    elif (board == "U250"):
        NUM_DSP = 12288
    elif (board == "U55C"):
        NUM_DSP = 9024

    # # Opening JSON file
    # file_path = f"{prj_root}/../nn2fpga/boards/{board}.json"
    # with open(file_path) as f:
    #     board_dict = json.load(f)

    # # Right now consider the board as a monolithic block 
    # board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}
    # for block in board_dict['resource']:
    #     for res in block.keys():
    #         if res in board_res:
    #             board_res[res] += block[res]

    # # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # # 36 bits, in which 4 bits are reserved to ECC code
    # SIZE_BRAM18 = ((18 * 1024) / 36) * 32
    
    # # Useful space in BRAM36, composed by two BRAM18.
    # SIZE_BRAM36 = SIZE_BRAM18 * 2
    
    # # Useful space in URAM. Each URAM is 288kb with a maximum word width of 72
    # # bits.
    # SIZE_URAM = (288 * 1024)

    num_layers = len(layers_info)

    # Counting merged 1x1 layers
    for i in range(num_layers):
        if layers_info[i]["merge_1x1"]:
            NUM_DSP -= 1

    ####### Problem formulated as LP #######
    # prob = pulp.LpProblem("Parallel ops", pulp.LpMaximize)
    # # Variable to be optimized: Throughtput expressed as the number of filters
    # # computed in parallel by the heaviest layer
    # worst_par = pulp.LpVariable("Worst layer parallelization", 1, None, pulp.LpInteger)
    # prob += worst_par, "Bottleneck layer parallelization"
    # # Constraints: Sum of DSPs needed for each layer to keep throughtput
    # # constant should be less than the threshold. The parallelism of each layer
    # # is computed using the equality:
    # #
    # #    iter worst        iter layer  
    # #  -------------- =  --------------
    # #     par worst        par layer   
    # prob += pulp.lpSum([worst_par * layers_info[i]["kernel"] / layers_info[i]["value"] for i in range(num_layers)]) <= NUM_DSP
    # # Function to be maximized
    # prob.writeLP(prj_root + "/parallel_ops.lp")
    # prob.solve()
    # print("Status:", pulp.LpStatus[prob.status])
    # ilp_value = int(worst_par.value())
    ########################################
    
    # ilp_value = math.floor(board_res['dsp'] * (1 / layers_info[worst_index]['total'])/sum(
    #     layers_info[i]['kernel'] / layers_info[i]['total'] for i in range(num_layers)))
    ilp_value = math.floor(NUM_DSP * (1 / layers_info[worst_index]['total'])/sum(
        layers_info[i]['kernel'] / layers_info[i]['total'] for i in range(num_layers)))
    
    # if mem_bind[i] = 0 then BRAM
    # prob += pulp.lpSum([(mem_bind[i] * worst_par * layers_info[i]["kernel"] / layers_info[i]["value"]
    #                    for i in range(num_layers)) for i in range(num_layers)]) <= board_res["uram"]
    # prob += pulp.lpSum([(abs(mem_bind[i] - 1) * worst_par * layers_info[i]["kernel"] / layers_info[i]
    #                    ["value"] for i in range(num_layers)) for i in range(num_layers)]) <= board_res["bram"]

    #mem_bind = pulp.LpVariable.dicts("Memory binding", range(num_layers), pulp.LpBinary)

    # For each layers compute the possible parallelization factors
    all_divisors, layers_divisors, layers_offset = find_divisors(layers_info)

    # Searching for an integer divisor of the number of DSPs that is the closest to the
    # number of operations that should be executed in parallel
    low_range, low_index, high_range, _ = find_range(
        all_divisors[layers_offset[worst_index]:layers_offset[worst_index]+layers_divisors[worst_index]],
        ilp_value
    )

    low_index += layers_offset[worst_index]
    # print(all_divisors[layers_offset[i]:layers_offset[i]+layers_divisors[i]], data, low_range, high_range, layers_info[best_index][5], layers_info[best_index][6])
    parallel_op = {}
    
    # The parallelization factor for the bottleneck layer is the lower nearest
    # value between the possible ones.
    worst_par = low_range

    # Based on the chosen parallalization, the layers are updated with their
    # respective parallelization factors
    for i in range(num_layers):
        layer_par = worst_par / layers_info[i]["value"]

        low_range, _ , high_range, _ = find_range(
            all_divisors[layers_offset[i]:layers_offset[i]+layers_divisors[i]],
            math.ceil(layer_par)
        )
        parallel_op[layers_info[i]["name"]] = high_range
        # print(all_divisors[layers_offset[i]:layers_offset[i]+layers_divisors[i]], data, low_range, high_range, layers_info[i][5], layers_info[i][6])

    print(f"DSP before: {sum(parallel_op.values())}")
    return parallel_op

def ilp(io_dict, off_chip_storage, model, board="ULTRA96v2", double_packing=True, prj_root="/tmp"):

    layers_info = []

    # Find the highest number of operation done by a single convolution.
    max_total = 1
    worst_index = 0
    iterator = 0
    for node_name, node_info in io_dict.items():
        if 'conv' in node_info["type"]:
            print(node_info["total"])
            if max_total > node_info["total"]:
                max_total = node_info["total"]
                worst_index = iterator
            iterator += 1

    total_computations = 0
    for node_name, node_info in io_dict.items():
        if 'conv' in node_info["type"]:
            # print(node_info)
            value = node_info["total"]/max_total
            # value = 2**value
            # print(node_name, node_info["total"], value)
            total_computations += node_info["total_log"]
            
            if node_info["depth"]:
                max_par = node_info["ich"]
            else:
                max_par = node_info["och"]*node_info["ich"]

            layers_info.append(
                {
                    "name": node_name,
                    "total": node_info["total"],
                    "kernel": node_info["kernel"],
                    "img_ch": node_info["img_ch"],
                    "merge_1x1": node_info["merge_1x1"],
                    "value": value,
                    "max_par": max_par,
                    "max_par_1x1": max_par,
                    "bits": node_info["bits"],
                    "n_weights": node_info["ih"] * node_info["iw"] * node_info["ich"] * node_info["och"]
                }
            )

            if (node_info["merge_1x1"]):
                layers_info[-1]["max_par_1x1"] = node_info["och_1x1"] * node_info["ich"]
                # FIX: removing fix to try clamping ops to 1x1 layers
                # layers_info[-1].append(node_info["och"])
    
    # print("Total computations:", total_computations)

    parallel_ops = parallel_ops_number(layers_info, worst_index, board, prj_root=prj_root)

    print(parallel_ops)

    io_connect = extract_connections(model, io_dict)

    for node_name, ops in parallel_ops.items():
        print(node_name, io_dict[node_name]["ich"], io_dict[node_name]["och"], ops)
        if (not io_dict[node_name]["depth"]):
            och_ops = find_higher_mult(io_dict[node_name]["och"], ops)
        else:
            och_ops = 1
        io_dict[node_name]["ops"] = och_ops
        io_dict[node_name]["ich_ops"] = ops//och_ops
        # io_dict[node_name]["ich_ops"] = ops
        io_dict[node_name]["dp"] = False

        # Evaluating neccessary output channels to avoid losing performance
        io_dict[node_name]["out_par"] = int(io_dict[node_name]["oh"] * io_dict[node_name]["ow"] * io_dict[node_name]["och"] * io_dict[node_name]["total"])
        if io_dict[node_name]["out_par"] == 0:
            io_dict[node_name]["out_par"] = 1

    # FIX: INCREASING OPS FOR DEPTH
    for name, node in io_dict.items():
        if "ops" in node:
            if "depth" in node:
                if node["depth"]:
                    input_dimension = node["ich"]*node["iw"]*node["ih"]
                    pipeline_iterations = node["och"]*node["ow"]*node["oh"]
                    ich_ops = math.ceil(input_dimension/pipeline_iterations)
                    if node["ich_ops"] < ich_ops:
                        io_dict[name]["ich_ops"] = ich_ops
                        print("#### Changing ich_ops for", name, "to", io_dict[name]["ich_ops"], "to avoid bottleneck")

    #TODO: Avoiding cycling twice because of pool layers
    for node_name, ops in parallel_ops.items():
        output_name = io_dict[node_name]["output"][0]
        output_node_name = io_connect[output_name][1][0]
        if output_node_name != "consume_stream":
            io_dict[output_node_name]["in_ops"] = ops
            if "pool" in io_dict[output_node_name]["type"]:
                io_dict[output_node_name]["ops"] = ops
                io_dict[output_node_name]["ich_ops"] = ops

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
                if "pool" in io_dict[output_node_name]["type"]:
                    io_dict[output_node_name]["ops"] = ops
                    io_dict[output_node_name]["ich_ops"] = ops

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
    
    # Input produce stream ops
    print_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        # print ops and ich_ops for conv and pool layers
        if node["type"] in print_layers:
            print(f"{name}, {node['ops']}, {node['ich_ops']}, [{node['och']}][{node['ich']}]")
            # check if the input tensor is produced by a produce_stream node
            input_name = node["input"][0]
            input_node_name = io_connect[input_name][0][0]
            if io_dict[input_node_name]["type"] == "produce":
                io_dict[input_node_name]["ops"] = node["ich_ops"]
                node["in_ops"] = node["ich_ops"]
        
    # Avoiding line buffer to be the bottleneck in case of strides
    for name, node in io_dict.items():
        # print ops and ich_ops for conv and pool layers
        if node["type"] == "conv":
            # Trading off ops for ich_ops to avoid problems in line buffer
            cycles_line_buffer = node["ih"]*node["iw"]*node["ich"] // node["ich_ops"]
            if not node["depth"]:
                cycles_computation = (node["oh"]*node["ow"]*node["och"]*node["ich"]) // (node["ops"]*node["ich_ops"])
            elif node["depth"]:
                cycles_computation = (node["oh"]*node["ow"]*node["och"]) // (node["ich_ops"])
            else:
                continue

            # If mult_factor is lower or equal than 1, then the line buffer is not the bottleneck
            mult_factor = math.ceil(cycles_line_buffer/cycles_computation)
            if mult_factor <= 1:
                continue
            
            # If not paralllizing on the input channels
            if node["depth"]:
                mult_factor = find_higher_mult(io_dict[name]["ich"]//node["ich_ops"], mult_factor)
                node["ich_ops"] = mult_factor*node["ich_ops"]
            else:
                mult_factor = find_higher_mult(io_dict[name]["ich"]//node["ich_ops"], mult_factor)
                node["ich_ops"] = mult_factor*node["ich_ops"]
                node["ops"] = find_higher_mult(node["ops"], node["ops"]//mult_factor)
                if node["ops"] == 0:
                    node["ops"] = 1

            output_name = node["output"][0]
            output_node_name = io_connect[output_name][1][0]
            if output_node_name != "consume_stream":
                if node["depth"]:
                    io_dict[output_node_name]["in_ops"] = node["ich_ops"]
                else:
                    io_dict[output_node_name]["in_ops"] = node["ops"]
        
    # # Fastening previous layer to provide enough bandwidth to the next one
    # print_layers = ["conv", "pool"]
    # for name, node in io_dict.items():
    #     if node["type"] in print_layers:
    #         # check if the input tensor is produced by a produce_stream node
    #         input_name = node["input"][0]
    #         input_node_name = io_connect[input_name][0][0]
    #         input_node = io_dict[input_node_name]
    #         ich_par = True
    #         ich_par = ich_par and input_node["type"] == "conv"
    #         if ich_par:
    #             ich_par = ich_par and input_node["depth"]
    #         if ich_par:
    #             dim_str = "ich"
    #             ops_str = "ich_ops"
    #         else:
    #             dim_str = "och"
    #             ops_str = "ops"
    #         if input_node[ops_str] < node["ich_ops"]:
    #             mult_factor = find_higher_mult(input_node[dim_str]//input_node[ops_str], node["ich_ops"])
    #             mult_factor = input_node[ops_str] * mult_factor
    #             input_node[ops_str] = mult_factor
    #             node["in_ops"] = mult_factor
    #             print("#### Changing ops for", input_node_name, "to", node["in_ops"], "to avoid bottleneck")
        
    # Fastening previous layer to provide enough bandwidth to the next one
    print_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in print_layers:
            # check if the input tensor is produced by a produce_stream node
            input_name = node["input"][0]
            input_node_name = io_connect[input_name][0][0]
            input_node = io_dict[input_node_name]
            ich_par = True
            ich_par = ich_par and input_node["type"] == "conv"
            if ich_par:
                ich_par = ich_par and input_node["depth"]
            if ich_par:
                dim_str = "ich"
                ops_str = "ich_ops"
            else:
                dim_str = "och"
                ops_str = "ops"
            if input_node[ops_str] < node["ich_ops"]:
                if (node["och"]//node["ops"]) > node["ich_ops"]:
                    node["ops"] = find_lower_mult(node["ops"]*node["ich_ops"], node["och"])
                    node["ich_ops"] = 1
                    print("#### Changing ops for", name, "to", node["ops"], "to avoid bottleneck")
                    output_name = node["output"][0]
                    output_node_name = io_connect[output_name][1][0]
                    io_dict[output_node_name]["in_ops"] = node["ops"]

    # Check for necessary bandwidth adjustements for the line buffer
    line_buffer_layers = ["conv", "pool"]
    for name, node in io_dict.items():
        if node["type"] in line_buffer_layers:
            if (node["in_ops"] % node["ich_ops"]) != 0:
                node["adjust_line_buffer"] = True
                node["adjust_ops"] = find_common_mult(node["in_ops"],node["ich_ops"])
                print("#### Found line buffer read/write rate for", name, "read", node["in_ops"], "write", node["ich_ops"], "to avoid bottleneck")
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
        # print ops and ich_ops for conv and pool layers
        if node["type"] in print_layers:
            print(name, node["ops"], node["ich_ops"])

    return io_dict
