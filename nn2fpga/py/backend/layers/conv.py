import os
import sys
#import onnx
import qonnx
import math
from onnx import numpy_helper
import numpy as np
import backend.quant
from backend.layers.quant import get_quant_type, get_quant_constant
from backend.layers import weights
from backend.layers.layer import Layer

class Conv2D(Layer):

    name = "Conv2D"
    inputs = []
    outputs = []

    # Attributes of the layer
    pads = [0, 0, 0, 0] # Default padding in the top, bottom, left, and right directions 
    kernel = [1, 1] # Default kernel size
    strides = [1, 1] # Default strides in the h and w directions
    group = 1 # Default group

    # Tensors shapes
    input_tensor_shape = None
    output_tensor_shape = None
    weights_tensor_shape = None
    bias_tensor_shape = None

    # Graph optimizations
    merged_relu = None
    merged_leakyrelu = None
    merged_add = None
    merged_pointwise = None
    merged_forward = False

    # Tiling factors
    tiling_factors = [1, 1, 1, 1] # Default tiling factors in the n, c, h, and w directions 

    # Loops order
    loops_order = None

    def __init__(self, onnx_node, onnx_graph):
        self.inputs = onnx_node.input
        self.outputs = onnx_node.output
        self.name = onnx_node.name
        attributes = getattr(onnx_node, 'attribute')

        for attribute in attributes:
            if getattr(attribute, 'name') == "kernel_shape":
                self.kernel = getattr(attribute, 'ints')
            if getattr(attribute, 'name') == "strides":
                self.strides = getattr(attribute, 'ints')
            if getattr(attribute, 'name') == "pads":
                self.pads = getattr(attribute, 'ints')
            if getattr(attribute, 'name') == "group":
                self.group = getattr(attribute, 'i')
        
        self.input_tensor_shape = onnx_graph[self.inputs[0]].get_tensor_shape()
        self.output_tensor_shape = onnx_graph[self.outputs[0]].get_tensor_shape()
        self.weights_tensor_shape = onnx_graph[self.inputs[1]].get_tensor_shape()
        if len(self.inputs) > 2:
            self.bias_tensor_shape = onnx_graph[self.inputs[2]].get_tensor_shape()

    def has_bias(self):
        return self.bias_tensor_shape is not None

    def get_total_mac(self):
        return np.prod(self.output_tensor_shape) * (self.input_tensor_shape[1] / self.group) * np.prod(self.kernel)

    def get_input_tensor_size(self):
        return np.prod(self.input_tensor_shape)

    def get_weights_tensor_size(self):
        return np.prod(self.weights_tensor_shape)

    def parse(self):
        return super().parse()
        

def chain_builder(node):
    """ Builds the DSPs chain for the accumulation process """
    
    # Computing Packing guard bits for accumulation process
    if node["och_pack"] > 1:
        a_d_bits = node["weight_quant"]["bits"]
        b_bits = node["input_quant"]["bits"]
    else:
        a_d_bits = node["input_quant"]["bits"]
        b_bits = node["weight_quant"]["bits"]

    end_simd_bit = 27 - 1 - a_d_bits

    # number of partial results which must be padded in the 27-bits word
    n_partial = node["och_pack"] * node["ow_pack"] // 2
    
    if (n_partial == 0):
        n_partial = 1
    guard_bits = (end_simd_bit - (a_d_bits + b_bits) * n_partial) // n_partial

    op_group = 1 if (node["depth"] == 1) else node["ich_ops"]

    # Computing the number of DSP chains needed for the accumulation process. 
    # We must ensure that the accumulator will not overflow. As example, with
    # 2 guard bits, the maximum number of sums without requiring an additional
    # bit is 7.
    max_acc = (2 ** (guard_bits + 1)) - 1
    dsp_chains = int(np.ceil(node["kernel"] * op_group / max_acc))
    log2_dsp_chains = int(math.ceil(np.log2(dsp_chains)))

    # Mask selects where the partial results needs to be accumulated. It is used
    # to avoid using a modulo operation inside the convolution.
    mask = (1 << log2_dsp_chains) - 1
    n_acc = 2 ** log2_dsp_chains 

    print(f"simd: {n_acc}, max_acc {max_acc}, guard_bits {guard_bits}, mask: {mask} with {node['kernel'] * op_group} adds.")
    return n_acc, mask, guard_bits

def info(io_dict, node, node_name, init_info, tensors_info):

    attributes = getattr(node, "attribute")
    inputs = getattr(node, "input")
    input_shape = tensors_info[node.input[0]].tensor_type.shape
    output_shape = tensors_info[node.output[0]].tensor_type.shape

    for i, attribute in enumerate(attributes):
        if getattr(attribute, 'name') == "kernel_shape":
            kernel_index = i
        if getattr(attribute, 'name') == "strides":
            strides_index = i
        if getattr(attribute, 'name') == "pads":
            pads_index = i
        if getattr(attribute, 'name') == "group":
            group_index = i

    weight_name = inputs[1] 
            
    if (len(inputs) > 2):
        bias_name = inputs[2]
    
    # Check if kernel strides and pads exist and if not set terminate
    if 'kernel_index' not in locals():
        print("Kernel index not found in conv")
        exit(0)
    if 'strides_index' not in locals():
        print("Strides index not found in conv")
        exit(0)
    if 'pads_index' not in locals():
        print("Pads index not found in conv")
        exit(0)

    ich      = getattr(input_shape, 'dim')[1].dim_value
    ih       = getattr(input_shape, 'dim')[2].dim_value
    iw       = getattr(input_shape, 'dim')[3].dim_value
    och      = getattr(output_shape, 'dim')[1].dim_value
    oh       = getattr(output_shape, 'dim')[2].dim_value
    ow       = getattr(output_shape, 'dim')[3].dim_value
    fh       = getattr(attributes[kernel_index], 'ints')[0]
    fw       = getattr(attributes[kernel_index], 'ints')[1]
    stride   = getattr(attributes[strides_index], 'ints')[0]
    pad      = getattr(attributes[pads_index], 'ints')[0]
    kernel   = fh * fw
    img_ch   = ich * och
    relu     = False
    add      = False
    leakyrelu     = False
    in_scale_factor = [None]
    in_bits = [None]
    in_signed = [None]

    # Check if groups exist and if not set to 1
    if 'group_index' not in locals():
        group = 1
    else:
        group = getattr(attributes[group_index], 'i')

    # Mark depthwise convolutions
    depth = (group == och)

    # Total number of window operations
    total = oh * ow * och * (ich / group) 

    io_dict[node_name]["depth"] = depth
    io_dict[node_name]["group"] = group
    io_dict[node_name]["ich"]    = ich
    io_dict[node_name]["ih"]     = ih
    io_dict[node_name]["iw"]     = iw
    io_dict[node_name]["och"]    = och
    io_dict[node_name]["oh"]     = oh
    io_dict[node_name]["ow"]     = ow
    io_dict[node_name]["fh"]     = fh
    io_dict[node_name]["fw"]     = fw
    io_dict[node_name]["stride"] = stride
    io_dict[node_name]["pad"]    = pad
    io_dict[node_name]["total"]  = total
    io_dict[node_name]["kernel"] = kernel
    io_dict[node_name]["img_ch"] = img_ch
    io_dict[node_name]["reuse"]  = 1
    io_dict[node_name]["relu"]   = relu
    io_dict[node_name]["add"]    = add
    io_dict[node_name]["leakyrelu"]   = leakyrelu
    io_dict[node_name]["scale_factor"] = 0
    io_dict[node_name]["in_scale_factor"] = in_scale_factor
    io_dict[node_name]["bits"]    = 0
    io_dict[node_name]["in_bits"] = in_bits
    io_dict[node_name]["in_signed"] = in_signed
    io_dict[node_name]["type"]   = 'conv'
    io_dict[node_name]["wbias"]  = len(node.input) > 2
    io_dict[node_name]["bbits"]  = 0
    io_dict[node_name]["wbits"]  = []
    io_dict[node_name]["wsigned"]  = []
    io_dict[node_name]["actbits"] = []
    io_dict[node_name]["wscale"]  = []
    io_dict[node_name]["actscale"] = []
    io_dict[node_name]["actsigned"] = []
    io_dict[node_name]["ops"] = 1
    io_dict[node_name]["in_ops"] = 1
    io_dict[node_name]["ich_ops"] = 1
    io_dict[node_name]["ow_ops"] = 1
    io_dict[node_name]["ow_ops_out"] = 1
    io_dict[node_name]["weights_name"] = [weight_name]
    io_dict[node_name]["has_forward"] = False
    io_dict[node_name]["merge_1x1"] = False
    io_dict[node_name]["start_comp_layer"] = False
    io_dict[node_name]["ops_out"] = 1

    # Supported quantizations for a convolutional layer
    io_dict[node_name]["input_quant"] = None            # Input quantization
    io_dict[node_name]["conv_output_quant"] = None      # Convolution output quantization
    io_dict[node_name]["output_quant"] = None           # After-merge output quantization
    
    if 'bias_name' in locals():
        io_dict[node_name]["bias_name"] = [bias_name]
        io_dict[node_name]["has_bias"] = True
    else:
        io_dict[node_name]["bias_name"] = None
        io_dict[node_name]["has_bias"] = False 


    return io_dict

def get_input_name(node):
    return node["input"][0]

def get_add_name(node):
    return node["input"][1]

def get_forward_name(node):
    return node["output"][1]

def get_merge_1x1_name(node):
    return node["output"][1]

def get_output_name(node):
    return node["output"][0]

def parse_comp(name, node, streaming_params=False):
    input_name  = get_input_name(node)
    input_type_name = input_name.replace("_skip", "")
    weight_name = f"{name}_weight"

    # If no batchnorm merge then there is no bias
    has_bias = node["has_bias"]
    if (has_bias):
        bias_name = f"{name}_bias"

    if (node["merge_1x1"]):
        weight_1x1_name = f"{name}_merge_weight"
        if node["merge_node"]["has_bias"]:
            bias_1x1_name = f"{name}_merge_bias"

    if (node["add"]):
        add_name = get_add_name(node)
        add_base_type_name = add_name.replace("_skip", "")
        add_type_name = add_name

    output_name = get_output_name(node)
    output_type_name = output_name.replace("_skip", "")
    if (node["has_forward"]):
        forward_name = get_forward_name(node)
    if (node["merge_1x1"]):
        output_1x1_name = get_merge_1x1_name(node)

    block = {}
    block["func"] = "conv_comp_wrap"

    # Template parameters
    block["template"] = []
    if (node["pad"] == 0) and (node["ow_ops"] == 1):
        block["template"].append("t_%s_lb_struct" % input_type_name)
        block["template"].append("t_%s_lb" % input_type_name)
        block["template"].append("t_%s_reduce" % input_type_name)
    else:
        block["template"].append("t_%s_window_struct" % input_type_name)
        block["template"].append("t_%s_window" % input_type_name)
        block["template"].append("t_%s_reduce" % input_type_name)
    block["template"].append("t_%s" % input_type_name)
    block["template"].append("t_%s" % weight_name)
    block["template"].append("t_%s_mem" % weight_name)
    if (has_bias):
        block["template"].append("t_%s" % bias_name)
        block["template"].append("t_%s_mem" % bias_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    if (node["add"]):
        block["template"].append("t_%s_struct" % add_type_name)
        block["template"].append("t_%s_vector" % add_type_name)
        block["template"].append(f"t_{add_base_type_name}")
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    if (node["has_forward"]):
        block["template"].append("t_%s_struct" % forward_name)
    else:
        block["template"].append("std::nullptr_t")

    if (node["input_quant"] is not None):
        block["template"].append("t_%s_mod" % input_type_name)
    else:
        block["template"].append("t_%s" % input_type_name)

    if (node["merge_1x1"]):
        block["template"].append("t_%s_1x1" % input_type_name)
        block["template"].append("t_%s" % weight_1x1_name)
        block["template"].append("t_%s_mem" % weight_1x1_name)
        block["template"].append("t_%s" % bias_1x1_name)
        block["template"].append("t_%s_mem" % bias_1x1_name)
    else:
        block["template"].append("t_%s" % input_type_name)
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    block["template"].append("t_%s_acc_struct" % output_name)
    block["template"].append("t_%s_acc" % output_name)

    if (node["merge_1x1"]):
        block["template"].append("t_%s_acc_struct" % output_1x1_name)
        block["template"].append("t_%s_acc" % output_1x1_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    block["template"].append("t_%s_struct" % output_type_name)
    block["template"].append("t_%s_vector" % output_type_name)
    block["template"].append("t_%s" % output_type_name)
    if (node["conv_output_quant"] is not None):
        block["template"].append("t_%s_conv_quant" % output_type_name)
    else:
        block["template"].append("std::nullptr_t")
    if (node["add"] and node["add_node"]["output_quant"] is not None):
        block["template"].append("t_%s_add_quant" % output_type_name)
    else:
        block["template"].append("std::nullptr_t")
    if (node["merge_1x1"]):
        block["template"].append("t_%s_struct" % output_1x1_name)
        block["template"].append("t_%s" % output_1x1_name)
    else:
        block["template"].append("std::nullptr_t")
        block["template"].append("std::nullptr_t")

    block["template"].append("t_params_stream")
    
    # Not a beatiful solution, we cannot modify och since it is used also for the bias
    # size. Must correct this when introducing groups.
    if (node["depth"]):
        block["template"].append({"name" : "1", "comment" : f"ich divided by groups"})
    else:
        block["template"].append(f"c_{name}_ich")
    
    block["template"].append("c_%s_ich" % name)
    block["template"].append("c_%s_och" % name)
    block["template"].append("c_%s_och_1x1" % name)
    block["template"].append("c_%s_oh" % name)
    block["template"].append("c_%s_ow" % name)
    block["template"].append("c_%s_fh" % name)
    block["template"].append("c_%s_fw" % name)
    block["template"].append("c_%s_index" % name)
    block["template"].append("c_%s_stride" % name)
    block["template"].append("c_%s_ops" % name)
    block["template"].append("c_%s_ops_out" % name)
    # block["template"].append("c_%s_ops_1x1" % name)
    block["template"].append("c_%s_ich_ops" % name)
    
    if (node["add"]):
        if "adjust_add_ops" in node:
            block["template"].append({"name" : node["adjust_add_ops"], "comment" : "add ops"})
        else:
            block["template"].append("c_%s_add_ops" % add_name)
    else:
        block["template"].append({"name" : "1", "comment" : "add ops"})
    
    block["template"].append("c_%s_ow_ops" % name)
    block["template"].append("c_%s_ow_ops_out" % name)

    bias_ops = weights.bias_ops_calc(
        node["ich_ops"],
        node["ops"],
        node["depth"]
    )
    
    block["template"].append({"name" : bias_ops, "comment" : "bias_ops"})
    block["template"].append("c_%s_reuse" % name)
    block["template"].append("c_%s_ow_pack" % name)
    block["template"].append("c_%s_och_pack" % name)

    ##############################################################################
    # PACKING: providing info on quantization from template because
    # ap_fixed methods are not available at compile time and the
    # synthesizer gives an error
    abits, aibits = get_quant_constant(node["input_quant"]["signed"], node["input_quant"]["bits"], node["input_quant"]["scale_factor"])
    wbits, wibits = get_quant_constant(node["weight_quant"]["signed"], node["weight_quant"]["bits"], node["weight_quant"]["scale_factor"])

    n_acc, mask, guard_bits = chain_builder(node)
    
    block["template"].append({"name": f"{abits}", "comment": "act bits"})
    block["template"].append({"name": f"{aibits}", "comment": "act integer bits"})
    block["template"].append({"name": f"{wbits}", "comment": "weight bits"})
    block["template"].append({"name": f"{wibits}", "comment": "weight integer bits"})
    block["template"].append({"name": f"{guard_bits}", "comment": "guard bits"})
    block["template"].append({"name": f"{n_acc}", "comment": "accumulators in parallel"})
    block["template"].append({"name": f"{mask}", "comment": "mask bits"})

    #############################################################################
    # PACKING: providing info on quantization from template because
    # ap_fixed methods are not available at compile time and the
    # synthesizer gives an error.

    if (node["merge_1x1"]):
        abits, aibits = get_quant_constant(node["merge_node"]["input_quant"]["signed"], node["merge_node"]["input_quant"]["bits"], node["merge_node"]["input_quant"]["scale_factor"])
        wbits, wibits = get_quant_constant(node["merge_node"]["weight_quant"]["signed"], node["merge_node"]["weight_quant"]["bits"], node["merge_node"]["weight_quant"]["scale_factor"])
        n_acc, mask, guard_bits = chain_builder(node["merge_node"])
    else:
        abits = 0
        aibits = 0
        wbits = 0
        wibits = 0
        guard_bits = 2
        n_acc = 1
        mask = 0

    block["template"].append({"name": f"{abits}", "comment": "act bits 1x1"})
    block["template"].append({"name": f"{aibits}", "comment": "act integer bits 1x1"})
    block["template"].append({"name": f"{wbits}", "comment": "weight bits 1x1"})
    block["template"].append({"name": f"{wibits}", "comment": "weight integer bits 1x1"})
    block["template"].append({"name": f"{guard_bits}", "comment": "guard bits 1x1"})
    block["template"].append({"name": f"{n_acc}", "comment": "accumulators in parallel 1x1"})
    block["template"].append({"name": f"{mask}", "comment": "mask bits 1x1"})
    
    width_stream = 8
    block["template"].append({"name" : node["shift_cycles"], "comment" : "shift cycles"})
    block["template"].append({"name" : width_stream, "comment" : "width stream"})
    block["template"].append({"name" : int(np.ceil(node["weight_quant"]["bits"] / width_stream)), "comment" : "weight reads per data"})
    
    if (node["has_bias"]):
        block["template"].append({"name" : int(np.ceil(node["bias_quant"]["bits"] / width_stream)), "comment" : "bias reads per data"})
    else:
        block["template"].append({"name" : 0, "comment" : "bias reads per data"})
    
    if (node["merge_1x1"]):
        block["template"].append({"name" : int(np.ceil(node["merge_node"]["weight_quant"]["bits"] / width_stream)), "comment" : "weight reads per data 1x1"})
        if (node["merge_node"]["has_bias"]):
            block["template"].append({"name" : int(np.ceil(node["merge_node"]["bias_quant"]["bits"] / width_stream)), "comment" : "bias reads per data 1x1"})
        else:
            block["template"].append({"name" : 0, "comment" : "bias reads per data 1x1"})
    else:
        block["template"].append({"name" : 0, "comment" : "weight reads per data 1x1"})
        block["template"].append({"name" : 0, "comment" : "bias reads per data 1x1"})
    
    block["template"].append("c_%s_relu" % name)
    block["template"].append("c_%s_leakyrelu" % name)
    if (node["depth"]):
        block["template"].append({"name" : "1", "comment" : "depth"})
    else:
        block["template"].append({"name" : "0", "comment" : "depth"})

    # Computing accumulator bits for worst case scenario
    actbits = node["input_quant"]["bits"]
    actscale = node["input_quant"]["scale_factor"]
    if (node["depth"]):
        acc_bits = actbits + node["weight_quant"]["bits"] + math.ceil(math.log2(node["kernel"]))
    else:
        acc_bits = actbits + node["weight_quant"]["bits"] + math.ceil(math.log2(node["kernel"] * node["ich"]))

    if (has_bias):
        acc_bits += 1
    if (node["add"]):
        acc_bits += 1

    acc_type = get_quant_type(True, acc_bits, actscale + node["weight_quant"]["scale_factor"], acc_reg=True)
    block["defines"] = {}
    block["defines"]["t_%s_acc" % output_name] = ["type", acc_type]
    block["defines"]["t_%s_acc_struct" % output_name] = [
        "struct",
        [["data", "t_%s_acc" % output_name], ["last", "bool"]]
    ]

    input_reduce_type = "std::array<t_%s, %0d>" % (input_name, node["line_ops"])
    block["defines"]["t_%s_reduce" % input_name] = ["type", input_reduce_type]
    input_window_type = "std::array<t_%s_reduce, %0d>" % (input_name, node["fh"]*(node["fw"]+(node["ow_ops"]-1)*node["stride"]))
    block["defines"]["t_%s_window" % input_name] = ["type", input_window_type]
    block["defines"]["t_%s_window_struct" % input_name] = [
        "struct",
        [["data", "t_%s_window" % input_name], ["last", "bool"]]
    ]
    input_lb_type = "std::array<t_%s_reduce, %0d>" % (input_name, 1)
    block["defines"]["t_%s_lb" % input_name] = ["type", input_lb_type]
    block["defines"]["t_%s_lb_struct" % input_name] = [
        "struct",
        [["data", "t_%s_lb" % input_name], ["last", "bool"]]
    ]
    if (node["has_forward"]):
        block["defines"]["t_%s_vector" % forward_name] = ["type", input_reduce_type]
        block["defines"]["t_%s_struct" % forward_name] = [
            "struct",
            [["data", "std::array<t_%s_reduce, 1>" % input_name]]
        ]
        block["defines"][f"t_{forward_name}"] = ["alias", f"t_{input_name}"]


    # Output type declarations
    output_type = get_quant_type(node["output_quant"]["signed"], node["output_quant"]["bits"], node["output_quant"]["scale_factor"])
    block["defines"]["t_%s" % output_name] = ["type", output_type]
    
    if (node["conv_output_quant"] is not None):
        output_type_mask = get_quant_type(node["conv_output_quant"]["signed"], node["conv_output_quant"]["bits"], node["conv_output_quant"]["scale_factor"])
        block["defines"]["t_%s_conv_quant" % output_name] = ["type", output_type_mask]

    if (node["add"] and node["add_node"]["output_quant"] is not None):
        output_type_clip = get_quant_type(node["add_node"]["output_quant"]["signed"], node["add_node"]["output_quant"]["bits"],  node["add_node"]["output_quant"]["scale_factor"])
        block["defines"]["t_%s_add_quant" % output_name] = ["type", output_type_clip]
    
    output_ops = node["ops_out"]
    output_vector_type = "std::array<t_%s, %0d>" % (output_name, output_ops)
    block["defines"]["t_%s_vector" % output_name] = ["type", output_vector_type]
    block["defines"]["t_%s_struct" % output_name] = [
        "struct",
        [["data", "std::array<t_%s_vector, 1>" % output_name], ["last", "bool"]]
    ]

    if (node["merge_1x1"]):
        output_type_1x1 = get_quant_type(node["merge_node"]["output_quant"]["signed"], node["merge_node"]["output_quant"]["bits"], node["merge_node"]["output_quant"]["scale_factor"])
        block["defines"]["t_%s" % output_1x1_name] = ["type", output_type_1x1]
        output_vector_type = "std::array<t_%s, %0d>" % (output_1x1_name, node["ops_out"])
        block["defines"]["t_%s_vector" % output_1x1_name] = ["type", output_vector_type]
        block["defines"]["t_%s_struct" % output_1x1_name] = [
            "struct",
            [["data", "std::array<t_%s_vector, 1>" % output_1x1_name]]
        ]

    # Input type declarations
    input_type_mod = get_quant_type(node["input_quant"]["signed"], node["input_quant"]["bits"], node["input_quant"]["scale_factor"])
    block["defines"]["t_%s_mod" % input_name] = ["type", input_type_mod]

    if (node["merge_1x1"]):

        input_type_mod_1x1 = get_quant_type(node["merge_node"]["input_quant"]["signed"], node["merge_node"]["input_quant"]["bits"], node["merge_node"]["input_quant"]["scale_factor"])
        block["defines"]["t_%s_1x1" % input_name] = ["type", input_type_mod_1x1]

        # Computing accumulator bits for worst case scenario
        actbits = node["merge_node"]["input_quant"]["bits"]
        actscale = node["merge_node"]["input_quant"]["scale_factor"]
        if (node["depth"]):
            acc_bits = actbits + node["merge_node"]["weight_quant"]["bits"]
        else:
            acc_bits = actbits + node["merge_node"]["weight_quant"]["bits"] + math.ceil(math.log2(node["ich"]))
        if (node["merge_node"]["has_bias"]):
            acc_bits += 1
        acc_type_1x1 = get_quant_type(True, acc_bits, actscale + node["merge_node"]["weight_quant"]["scale_factor"], acc_reg=True)

        block["defines"]["t_%s_acc" % output_1x1_name] = ["type", acc_type_1x1]
        block["defines"]["t_%s_acc_struct" % output_1x1_name] = [
            "struct",
            [["data", "t_%s_acc" % output_1x1_name], ["last", "bool"]]
        ]


    block["defines"]["c_%s_ich" % name]            = ["const", node["ich"]]
    block["defines"]["c_%s_och" % name]            = ["const", node["och"]]
    if (node["merge_1x1"]):
        block["defines"]["c_%s_och_1x1" % name]        = ["const", node["merge_node"]["och"]]
    else:
        block["defines"]["c_%s_och_1x1" % name]        = ["const", node["och"]]
    block["defines"]["c_%s_iw" % name]             = ["const", node["iw"]]
    block["defines"]["c_%s_ih" % name]             = ["const", node["ih"]]
    block["defines"]["c_%s_fw" % name]             = ["const", node["fw"]]
    block["defines"]["c_%s_fh" % name]             = ["const", node["fh"]]
    block["defines"]["c_%s_ow" % name]             = ["const", node["ow"]]
    block["defines"]["c_%s_oh" % name]             = ["const", node["oh"]]
    block["defines"]["c_%s_relu" % name]           = ["const", int(node["relu"])]
    block["defines"]["c_%s_leakyrelu" % name]      = ["const", int(node["leakyrelu"])]
    block["defines"]["c_%s_stride" % name]         = ["const", node["stride"]]
    block["defines"]["c_%s_pad" % name]            = ["const", node["pad"]]
    block["defines"]["c_%s_ops" % name]            = ["const", node["ops"]]
    block["defines"]["c_%s_ops_out" % name]        = ["const", node["ops_out"]]
    # block["defines"]["c_%s_ops_1x1" % name]        = ["const", node["ops_1x1"]]

    ### in_ops is not used by anyone!
    block["defines"]["c_%s_in_ops" % name]         = ["const", node["in_ops"]]
    block["defines"]["c_%s_ich_ops" % name]        = ["const", node["ich_ops"]]
    block["defines"]["c_%s_index" % name]          = ["const", node["kernel"]]
    block["defines"]["c_%s_reuse" % name]          = ["const", node["reuse"]]
    block["defines"]["c_%s_ow_ops" % name]         = ["const", node["ow_ops"]]
    block["defines"]["c_%s_ow_ops_out" % name]     = ["const", node["ow_ops_out"]]
    block["defines"]["c_%s_ow_pack" % name]        = ["const", node["ow_pack"]]
    block["defines"]["c_%s_och_pack" % name]       = ["const", node["och_pack"]]
    
    if (node["has_forward"]):
        block["defines"]["c_%s_add_ops" % forward_name]         = ["const", node["ich_ops"]]
    elif (node["merge_1x1"]):
        block["defines"]["c_%s_add_ops" % output_1x1_name]         = ["const", node["ops_out"]]
    else:
        block["defines"][f"c_{output_name}_add_ops"]         = ["const", node["ops_out"]]

    block["args"] = []
    
    ### Params stream args
    block["args"].append(f"s_{node['shift_params_connections']['in']}_out")
    block["args"].append(f"s_{name}_init_flag")
    if node["shift_params_connections"]["out"] == "null":
        block["args"].append("(hls::stream<t_params_stream>*)(nullptr)")
    else:
        block["args"].append(f"s_{name}_out")
    ### End params stream args
    
    ### Memory args
    if (has_bias):
        block["args"].append(f"c_{bias_name}")
    else:
        block["args"].append(f"nullptr")

    block["args"].append(f"c_{weight_name}")
    
    if (has_bias and node["merge_1x1"]):
        block["args"].append(f"c_{bias_1x1_name}")
    else:
        block["args"].append("nullptr")
    
    if (node["merge_1x1"]):
        block["args"].append(f"c_{weight_1x1_name}")
    else:
        block["args"].append("nullptr")
    ### End memory args

    ### Conv stream args
    if (node["pad"] == 0) and (node["ow_ops"] == 1):
        block["args"].append("s_%s_pre_pad" % input_name)
    else:
        block["args"].append("s_%s_compute" % input_name)

    # block["args"].append("s_%s" % weight_name)
    # if (has_bias):
    #     block["args"].append("s_%s" % bias_name)
    # else:
    #     block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    # if (node["merge_1x1"]):
    #     block["args"].append("s_%s" % weight_1x1_name)
    #     block["args"].append("s_%s" % bias_1x1_name)
    # else:
    #     block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")
    #     block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    if (node["add"]):
        block["args"].append("s_%s" % add_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    if (node["has_forward"]):
        block["args"].append("s_%s" % forward_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")

    block["args"].append("s_%s" % output_name)

    if (node["merge_1x1"]):
        block["args"].append("s_%s" % output_1x1_name)
    else:
        block["args"].append("(hls::stream<std::nullptr_t>*)(nullptr)")
    ### End conv stream args

    block["output"] = []
    block["output"].append("s_%s" % output_name)

    ### Variable declaration
    block["declare"] = []
    declare = {}
    declare["name"] = "s_%s" % output_name
    declare["type"] = "t_%s_struct" % output_name
    declare["is_array"] = True
    declare["dim"] = node["ow_ops_out"]
    block["declare"].append(declare)

    if (node["merge_1x1"]):
        declare = {}
        declare["name"] = "s_%s" % output_1x1_name
        declare["type"] = "t_%s_struct" % output_1x1_name
        declare["is_array"] = True
        declare["dim"] = node["ow_ops_out"]
        block["declare"].append(declare)

    if (node["has_forward"]):
        declare = {}
        declare["name"] = "s_%s" % forward_name
        declare["type"] = "t_%s_struct" % forward_name
        declare["is_array"] = True
        declare["dim"] = node["ow_ops_out"]
        block["declare"].append(declare)
    
    declare = {}
    declare["name"] = f"static c_{weight_name}"
    declare["type"] = f"t_{weight_name}_mem"
    declare["is_array"] = True
    declare["is_const"] = False
    # size = weight_node["values"].shape
    declare["size"] = weights.mem_shape_calc(node, node["fh"], node["fw"], False)
    # declare["init"] = weight_node["values"]
    declare["form"] = "float"
    block["declare"].append(declare)
    
    if (has_bias):
        declare = {}
        declare["name"] = f"static c_{bias_name}"
        declare["type"] = f"t_{bias_name}_mem"
        declare["is_array"] = True
        declare["is_const"] = False
        declare["size"] = weights.mem_shape_calc(node, node["fh"], node["fw"], True)
        declare["form"] = "float"
        block["declare"].append(declare)
    
    if (node["merge_1x1"]):
        declare = {}
        declare["name"] = f"static c_{weight_1x1_name}"
        declare["type"] = f"t_{weight_1x1_name}_mem"
        declare["is_array"] = True
        declare["is_const"] = False
        declare["size"] = weights.mem_shape_calc(node, 1, 1, False)
        declare["form"] = "float"
        block["declare"].append(declare)
    
    if (node["merge_1x1"] and node["merge_node"]["has_bias"]):
        declare = {}
        declare["name"] = f"static c_{bias_1x1_name}"
        declare["type"] = f"t_{bias_1x1_name}_mem"
        declare["is_array"] = True
        declare["is_const"] = False
        declare["size"] = weights.mem_shape_calc(node, 1, 1, True)
        declare["form"] = "float"
        block["declare"].append(declare)

    declare = {}
    declare["name"] = f"s_{name}_init_flag"
    declare["type"] = "static bool"
    declare["is_array"] = False
    declare["is_const"] = False
    declare["size"] = 1
    declare["init_value"] = "false"
    block["declare"].append(declare)

    if node["shift_params_connections"]["out"] != "null":
        declare = {}
        declare["name"] = f"s_{name}_out"
        declare["type"] = f"t_params_stream"
        declare["is_array"] = True
        declare["dim"] = 1
        block["declare"].append(declare)
    ### Finish variable declaration

    block["pragma"] = []
   
    # Binding memory to URAM storage
    if node["weight_quant"]["uram_storage"]:
        pragma = {}
        pragma["name"] = "bind_storage"
        options = [
            ["variable", f"c_{weight_name}"],
            ["impl", "uram"],
            ["type", "ram_s2p"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
    
    if (node["merge_1x1"]):
        pragma = {}
        if (node["merge_node"]["weight_quant"]["uram_storage"]):
            pragma["name"] = "bind_storage"
            options = [
                ["variable", f"c_{weight_1x1_name}"],
                ["impl", "uram"],
                ["type", "ram_s2p"]
            ]
            pragma["options"] = options
            block["pragma"].append(pragma)

    # depth = int(node["och"]/node["ops"] + 1)
    # Completely reshaping weights and bias memory
    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", f"c_{weight_name}"],
        ["dim", 3],
        ["type", "complete"]
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    
    pragma = {}
    pragma["name"] = "array_reshape"
    options = [
        ["variable", f"c_{weight_name}"],
        ["dim", 1],
        ["type", "complete"]
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)

    pragma = {} 
    pragma["name"] = "array_partition"
    options = [
        ["variable", f"c_{weight_name}"],
        ["off", "true"],
    ]
    pragma["options"] = options
    block["pragma"].append(pragma)
    
    if (has_bias):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", f"c_{bias_name}"],
            ["dim", 2],
            ["type", "complete"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
        
        pragma = {} 
        pragma["name"] = "array_partition"
        options = [
            ["variable", f"c_{bias_name}"],
            ["off", "true"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
    
    if (node["merge_1x1"]):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", f"c_{weight_1x1_name}"],
            ["dim", 3],
            ["type", "complete"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
        
        pragma = {} 
        pragma["name"] = "array_partition"
        options = [
            ["variable", f"c_{weight_1x1_name}"],
            ["off", "true"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
    
    if (node["merge_1x1"] and node["merge_node"]["has_bias"]):
        pragma = {}
        pragma["name"] = "array_reshape"
        options = [
            ["variable", f"c_{bias_1x1_name}"],
            ["dim", 2],
            ["type", "complete"]
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)
        
        pragma = {} 
        pragma["name"] = "array_partition"
        options = [
            ["variable", f"c_{bias_1x1_name}"],
            ["off", "true"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    if (node["has_forward"]):
        # First two lines
        depth = node["depth_forward"]
        pragma = {}
        pragma["name"] = "stream"
        pragma_name = "s_%s" % forward_name
        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    # Dimension of the stream in output of the conv, covers a burst over och, ow_ops_out may be wrong
    depth = int(node["och"] / node["ops_out"]) * node["ow_ops_out"] + 1
    
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

    # FIX: Adding pragma to bind storage to SRL
    # if the depth of the fifo is small enough to
    # not justify the use of BRAM
    # if (depth < 64):
    #     pragma = {}
    #     pragma["name"] = "bind_storage"
    #     options = [
    #         ["variable", "s_%s" % output_name],
    #         ["impl", "SRL"],
    #         ["type", "fifo"]
    #     ]
    #     pragma["options"] = options

    #     block["pragma"].append(pragma)


    if (node["merge_1x1"]):
        pragma = {}
        pragma["name"] = "stream"
        pragma_name = "s_%s" % (output_1x1_name)

        # depth = node["ow"]*int(node["och"]/node["ops"])*(node["fh"]-1)-node["ich"]
        depth = node["depth_1x1"]

        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

        # FIX: Adding pragma to bind storage to SRL
        # if the depth of the fifo is small enough to
        # not justify the use of BRAM
        # if (depth < 64):
        #     pragma = {}
        #     pragma["name"] = "bind_storage"
        #     options = [
        #         ["variable", "s_%s" % output_1x1_name],
        #         ["impl", "SRL"],
        #     ["type", "fifo"]
        #     ]
        #     pragma["options"] = options

        #     block["pragma"].append(pragma)



    return [block]

def parse(name, node, streaming_params=False):

    return parse_comp(name, node, streaming_params)