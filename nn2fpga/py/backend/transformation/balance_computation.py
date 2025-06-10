import pulp
import math
import json
import time
import numpy as np
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from backend.util.quant_utils import get_quant_params
from onnx import helper, NodeProto
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate

PARALLELIZABLE_LAYERS = ["Conv", "GlobalAveragePool", "GlobalMaxPool", "AveragePool", "MaxPool", "ProduceStream", "ConsumeStream"]

def read_board_info(board, prj_root):
    """ Read the board json file and returns a dictionary with the available resources"""
    
    # Remove the part before NN2FPGA from the path
    file_path = prj_root.split("NN2FPGA")[0]
    file_path = f"{file_path}/NN2FPGA/nn2fpga/boards/{board}.json"

    # Opening JSON file with board resources
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block 
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}
    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    board_res["axi_bitwidth"] = board_dict["axi_bitwidth"]
    
    return board_res

def extract_quant_bitwidth(node: NodeProto, model: ModelWrapper) -> int:
    """ Extracts the bitwidth of the quantization parameters from a Quant node. """
    quant_params = get_quant_params(node, model)
    if quant_params["bitwidth"] is not None:
        return int(quant_params["bitwidth"])

def packing_feature(operands_bitwidth, par, silvia_packing):
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
        if (silvia_packing):
            if (par[0] % 4 == 0):
                return 4, (4, 1)
            if (par[2] % 4 == 0):
                return 4, (1, 4)
        else:
            if (par[0] % 2 == 0 and par[2] % 2 == 0):
                return 4, (2, 2)
        if (par[2] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    return 1, (1, 1)

def compute_bram_layer(weight_bits, weight_number, parallelism):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    bram9 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=9)
    bram18 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=18)
    bram36 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=36)
    bram72 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=72)
    
    return min(bram9, bram18, bram36, bram72)

def bram_consumption(weight_bits, weight_number, parallelism, WIDTH=36):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # 36 bits, in which 4 bits are reserved to ECC code
    SIZE_BRAM18 = (18 * 1024)
    
    # Useful space in BRAM36, composed by two BRAM18.
    SIZE_BRAM36 = SIZE_BRAM18 * 2

    WIDTH_BRAM36 = WIDTH

    # Assuming is implemented using LUTRAM
    if (weight_number * weight_bits) <= SIZE_BRAM36:
        return 0
    
    very_long_word = parallelism * weight_bits
    mem_width = very_long_word // WIDTH_BRAM36
    mem_width_rem = very_long_word % WIDTH_BRAM36
    word_depth = weight_number // parallelism
    mem_depth = int(math.ceil(word_depth / (SIZE_BRAM36 // WIDTH_BRAM36)))
    tot_bram = mem_width * mem_depth

    rem_bram = 0
    if (mem_width_rem > 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 72)))
    elif (mem_width_rem > 18 and mem_width_rem <= 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 36)))
    elif (mem_width_rem > 8 and mem_width_rem <= 18):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 18)))
    elif (mem_width_rem > 0 and mem_width_rem <= 8):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 9)))
    
    tot_bram += rem_bram
    
    return tot_bram

def generate_valid_combinations(och, ich, iw, och_clip=2**10, ich_clip=2**10, iw_clip=2**10, op_clip=2**20):
    """ Generate valid combinations of parallelization over ich, och and ow """
    combinations = []

    def divisors(n, clip):
        return [i for i in range(1, n + 1) if (n % i == 0 and i <= clip)]
    
    for div_och in divisors(och, och_clip):
        for div_ich in divisors(ich, ich_clip):
            for div_iw in divisors(iw, iw_clip):
                if (div_och * div_ich * div_iw <= op_clip):
                    combinations.append((div_och, div_ich, div_iw))
    return combinations 

def generate_architectures(layers_info: list, NUM_DSP: int, axi_bitwidth: int, silvia_packing: bool) -> list:
    """Given a list of layers, generate all the valid parallelization for each layer. """
    
    valid_par_solutions = []
    iw_clip = 4  # Heuristic clip for input width
    och_clip = 20  # Heuristic clip for output channels
    for layer in layers_info:
        max_och_par = layer["och"]
        max_ich_par = layer["ich"]
        max_ow_par = layer["ow"]

        # Depthwise-like layers cannot be parallelized over output channels.
        if (layer["depth"]):
            max_och_par = 1
        
        if (layer["type"] == "Conv"):
            
            # Clipping the maximum parallelization to the available DSPs, since it is not
            # possible that one layer uses all the DSPs. Considering also the packing.
            op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), (max_och_par, max_ich_par, max_ow_par), silvia_packing)
            op_clip = (NUM_DSP / layer["kernel"]) * op_per_dsp
        elif (layer["type"] in ["ProduceStream", "ConsumeStream"]):
            
            # Clipping the maximum parallelization of ProduceStream and ConsumeStream to the bandwidth of the AXI bus.
            op_clip = axi_bitwidth // layer["act_bits"]
        
        valid_par_solutions.append(generate_valid_combinations(
            och=max_och_par, ich=max_ich_par, iw=max_ow_par, iw_clip=iw_clip, och_clip=och_clip, op_clip=op_clip))

    return valid_par_solutions

def layers_extractions(model: ModelWrapper) -> list:
    """ Extracts computation-intensive layers from the model and returns their useful information
    for balancing the computation load across the model.
    """ 

    layers_info = []
    for node in model.graph.node:
        if node.op_type in PARALLELIZABLE_LAYERS:

            input_shape = model.get_tensor_shape(node.input[0])
            output_shape = model.get_tensor_shape(node.output[0])
        

            # Ensure input and output shapes are 4D (NCHW format)
            input_shape = [1] * (4 - len(input_shape)) + input_shape
            output_shape = [1] * (4 - len(output_shape)) + output_shape

            if node.op_type == "Conv":
                kernel_shape = get_by_name(node.attribute, "kernel_shape").ints
                kernel = int(math.prod(kernel_shape))
                group = get_by_name(node.attribute, "group").i
                ops = (
                    output_shape[1]
                    * output_shape[2]
                    * output_shape[3]
                    * input_shape[1]
                    * kernel
                    // group
                )
                weight_bits = extract_quant_bitwidth(
                    model.find_producer(node.input[1]), model
                )
                act_bits = extract_quant_bitwidth(
                    model.find_producer(node.input[0]), model
                )
                depth = group == input_shape[1] 

            elif node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
                kernel_shape = (input_shape[2], input_shape[3])
                kernel = int(math.prod(kernel_shape))
                depth = True
                ops = math.prod(input_shape)
                act_bits = extract_quant_bitwidth(
                    model.find_producer(node.input[0]), model
                )
                weight_bits = 0

            elif node.op_type in ["AveragePool", "MaxPool"]:
                kernel_shape = get_by_name(node.attribute, "kernel_shape").ints
                kernel = int(math.prod(kernel_shape))
                depth = True
                ops = math.prod(output_shape) * kernel
                act_bits = extract_quant_bitwidth(
                    model.find_producer(node.input[0]), model
                )
                weight_bits = 0

            elif node.op_type == "ConsumeStream":
                kernel = 1
                depth = True
                ops = math.prod(output_shape)
                weight_bits = 0
                act_bits = extract_quant_bitwidth(
                    model.find_producer(node.input[0]), model
                )

            elif node.op_type == "ProduceStream":
                kernel = 1
                depth = True
                ops = math.prod(input_shape)
                weight_bits = 0
                act_bits = extract_quant_bitwidth(
                    model.find_consumer(node.output[0]), model
                )

            layers_info.append(
                {
                    "type": node.op_type,
                    "name": node.name,
                    "total": ops,
                    "kernel": kernel,
                    "weight_bits": weight_bits,
                    "act_bits": act_bits,
                    "ich": input_shape[1],
                    "ih": input_shape[2],
                    "iw": input_shape[3],
                    "och": output_shape[1],
                    "oh": output_shape[2],
                    "ow": output_shape[3],
                    "depth": depth,
                }
            )

    return layers_info

def parallelismILP(layers_info, valid_par_solutions, NUM_DSP, NUM_PORTS, silvia_packing, prj_root="/tmp"):
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
    for par_sol, layer in zip(valid_par_solutions, layers_info):
        valid_iter_linebuffer.append([])
        layer_iter = layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in par_sol:
            valid_iter_linebuffer[-1].append(
                layer_iter // (single_par[1] * single_par[2])
            )

    # valid_iter_solutions stores the number of iteration for each valid
    # parallelization.
    valid_iter_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_iter_solutions.append([])
        layer_iter = layer["total"]
        for single_par in layer_par:
            unroll_factor = layer["kernel"] * np.prod(single_par)
            valid_iter_solutions[-1].append(layer_iter // unroll_factor)

    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing
    valid_dsp_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_dsp_solutions.append([])

        for single_par in layer_par:
            if (layer["type"] == "Conv"):

                # For Conv layers, the unrolling is done over the output width, output channels and input channels.
                # The DSPs considered are coming from the MAC operation, considering the packing.
                op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), single_par, silvia_packing)
                dsp_used = (np.prod(single_par) * layer["kernel"]) / op_per_dsp

            elif (layer["type"] in ["GlobalAveragePool", "AveragePool"]):

                # GlobalAveragePool and AveragePool are unrolled over the output width and input channels.
                # The DSPs considered are coming from the division operation. Each single integer division requires 2 DSPs.
                dsp_used = (np.prod(single_par)) * 2

            else:
                # All the other layers do not involve DSPs, so they are not considered.
                dsp_used = 0

            valid_dsp_solutions[-1].append(dsp_used)

    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_bram_solutions.append([])
        n_weights = 0

        if (layer["type"] == "Conv"):
            n_weights = layer["ich"] * layer["och"] * layer["kernel"]

            if (layer["depth"]):
                n_weights = layer["ich"] * layer["kernel"]

        for single_par in layer_par:
            bram_used = compute_bram_layer(layer["weight_bits"], n_weights, np.prod(single_par[:2]) * layer["kernel"])
            valid_bram_solutions[-1].append(bram_used)

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
    for layer_index, layer in enumerate(layers_info):
        constraints_counter += 1
        ones = [1] * len(layer_binary_variables[layer_index])
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}",
        )

    # Constraint: The total number of DSPs used to achieve the chosen
    # parallelization should be lower than the available ones.
    constraints_counter += 1
    prob += (
        pulp.lpSum(
            [
                pulp.lpDot(layer_binary_variables[i].values(), valid_dsp_solutions[i])
                for i, layer in enumerate(layers_info)
            ]
        )
        <= NUM_DSP,
        f"DSP_constraint",
    )

    # Constraint: The total number of BRAMs used to achieve the chosen
    # parallelization should be lower than the available ones.
    constraints_counter += 1
    prob += (
        pulp.lpSum(
            [
                pulp.lpDot(layer_binary_variables[i].values(), valid_bram_solutions[i])
                for i, layer in enumerate(layers_info)
            ]
        )
        <= NUM_PORTS,
        f"BRAM_constraint",
    )

    # Constraints: The latency of each layer should be equal or lower to the minimization variable.
    for layer_index, layer in enumerate(layers_info):
        constraints_counter += 1
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= var,
            f"Latency_constraint_layer_{layer_index}"
        )

    # Constraints: The latency of each line buffer should be equal or lower to the minimization variable.
    for layer_index, layer in enumerate(layers_info):
        constraints_counter += 1
        prob += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= var,
            f"Linebuffer_constraint_layer_{layer_index}"
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

def print_report(layers_info, layer_par, n_variables, n_constraints, model_II, time_spent, silvia_packing, generate_report_file, prj_root="/tmp"):
    with open(generate_report_file, "a+") as f:
        print("=" * 40, file=f)
        print("== Parallelization report", file=f)
        print("=" * 40, file=f)
        print(f"Number of variables: \t\t\t{n_variables}", file=f)
        print(f"Number of constraints:\t\t\t{n_constraints}", file=f)
        print(f"Time to solve: \t\t\t\t\t{time_spent:.2f}s", file=f)
        print(f"Initiation interval: \t\t\t{model_II}cc", file=f)
        print(
            f"Theorical throughput @ 200MHz: \t{1000000000.0 / (model_II * 5):.2f}FPS\n",
            file=f,
        )
        table_data = []

        # header row
        header = [
            "Layer name",
            "ICH",
            "OCH",
            "OW",
            "ich_ops",
            "och_ops",
            "ow_ops",
            "DSPs",
            "PORTs",
            "Iter",
        ]
        table_data.append(header)

        DSPs = 0
        PORTs = 0
        for layer in layers_info:
            pack = False
            ow_ops = layer_par[layer["name"]][2]
            ich_ops = layer_par[layer["name"]][1]
            och_ops = layer_par[layer["name"]][0]

            port = dsp = 0
            if (layer["type"] == "Conv"):
                bits = layer["weight_bits"]
                dsp = layer["kernel"] * och_ops * ich_ops * ow_ops

                op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), layer_par[layer['name']], silvia_packing)
                pack = str(op_per_dsp)
                dsp = dsp // op_per_dsp
                weights = layer["ich"] * layer["och"] * layer["kernel"]
                if layer["depth"]:
                    weights = layer["ich"] * layer["kernel"]

                port = compute_bram_layer(bits, weights,  och_ops * ich_ops * layer["kernel"])

            iter = int(layer["total"] / (ich_ops * och_ops * ow_ops * layer["kernel"]))
            PORTs += port
            DSPs += dsp

            string_dsp = f"{dsp}"
            if pack:
                string_dsp += f" ({pack})"

            name = layer['name']

            row_data = [
                name,
                layer['ich'],
                layer['och'],
                layer['ow'],
                ich_ops,
                och_ops,
                ow_ops,
                string_dsp,
                port,
                iter
            ]

            table_data.append(row_data)

        footer = ["Totals", "", "", "", "", "", "", DSPs, PORTs, ""]
        table_data.append(footer)

        # Print the tabulated data to the file
        f.write(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print("\n", file=f)

class BalanceComputation(Transformation):
    """
    This transformation balances the computation load across the model by distributing
    resourcses evenly among the model's operations.
    """

    def __init__(self, silvia_packing: bool = False, nn2fpga_root: str = "/tmp"):
        """
        Initializes the BalanceComputation transformation.
        Args:
            silvia_packing (bool): If True, uses Silvia packing for DSPs.
            nn2fpga_root (str): The root directory of nn2FPGA.
        """
        super().__init__()
        self.nn2fpga_root = nn2fpga_root
        self.silvia_packing = silvia_packing

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """ Applies the transformation to the model.
        
        Args:
            model (ModelWrapper): The ONNX model to transform, wrapped in QONNX ModelWrapper.
        
        Returns:
            tuple: A tuple containing the transformed model and a boolean indicating if the transformation was applied.
        """

        board_res = read_board_info(
            board=model.get_metadata_prop("board_name"),
            prj_root=self.nn2fpga_root
        )

        NUM_PORTS = (board_res["bram"] + board_res["uram"])
        NUM_DSP = board_res["dsp"]
        NUM_PORTS = int(NUM_PORTS * 0.85) * 2

        # Extract layers information
        layers_info = layers_extractions(model)

        # Generate valid parallelization solutions for each layer
        valid_par_solutions = generate_architectures(
            layers_info, NUM_DSP, board_res["axi_bitwidth"], self.silvia_packing
        )

        # Balance the computation load across the model using ILP
        layer_par, model_II, n_variables, n_constraints, time_spent = parallelismILP(
            layers_info,
            valid_par_solutions,
            NUM_DSP,
            NUM_PORTS,
            self.silvia_packing,
            self.nn2fpga_root,
        )

        # Print the report
        generate_report_file = f"{self.nn2fpga_root}/balance_computation.rpt"
        print_report(
            layers_info,
            layer_par,
            n_variables,
            n_constraints,
            model_II,
            time_spent,
            self.silvia_packing,
            generate_report_file,
            prj_root=self.nn2fpga_root,
        )

        print(f"Balanced model with II {model_II} using {n_variables} variables and {n_constraints} constraints in {time_spent:.2f}s")
        return (model, False)
