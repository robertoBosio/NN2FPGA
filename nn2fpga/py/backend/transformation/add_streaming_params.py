from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
import qonnx.custom_op.general.quant as qonnx_quant
from onnx import NodeProto, TensorProto, helper
from collections import deque
from backend.util.par_utils import get_par_attributes, set_par_attributes, check_par_attributes
from backend.util.quant_utils import get_quant_params
import backend.transformation as transformation
import numpy as np
import os

NODE_WITH_PARAMS = [
    "Conv", 
    "Gemm",
    "MatMul",
    ]

# Save original references before monkey-patching
_original_min_int = qonnx_quant.min_int
_original_max_int = qonnx_quant.max_int

def _np_min_int(signed: bool, narrow_range: bool, bit_width: int) -> np.generic:
    val = _original_min_int(signed, narrow_range, bit_width)
    return np.array(val)

def _np_max_int(signed: bool, narrow_range: bool, bit_width: int) -> np.generic:
    val = _original_max_int(signed, narrow_range, bit_width)
    return np.array(val)

def safe_int_quant_call(*args, **kwargs):
    # Monkey-patch min/max to return NumPy-safe scalars
    qonnx_quant.min_int = _np_min_int
    qonnx_quant.max_int = _np_max_int

    try:
        return qonnx_quant.quant(*args, **kwargs)
    finally:
        # Restore originals
        qonnx_quant.min_int = _original_min_int
        qonnx_quant.max_int = _original_max_int

def quant_array(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode):
    """ Quantize an input tensor to a specified bitwidth and return the quantized tensor."""

    # Let QONNX handle the quantization. This function return the dequantized tensor.
    inp_tensor = safe_int_quant_call(
        inp_tensor,
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode
    )

    # Moving from a dequantized tensor to a quantized tensor, knowing that clipping
    # and rounding have already been applied.
    inp_tensor = inp_tensor / scale
    inp_tensor = inp_tensor + zeropt
    return inp_tensor.astype(np.int32)  # Convert to uint32 for packing

def pack_values_to_int32words(arr: np.ndarray, bitwidth: int) -> np.ndarray:
    """
    Packs values from the input array into 32-bit words, ensuring that each value
    fits entirely within a word (no value is split between two words). Uses padding
    as needed.

    Args:
        arr (np.ndarray): Input array of unsigned integers.
        bitwidth (int): Number of bits per value. Must be <= 32.

    Returns:
        np.ndarray: Packed 32-bit words as a 1D array of dtype=np.uint32.
    """
    if bitwidth > 32 or bitwidth <= 0:
        raise ValueError("bitwidth must be between 1 and 32")

    arr = arr.flatten()  # Ensure the input is a 1D array
    values_per_word = 32 // bitwidth  # Max number of values per word
    padded_len = int(((len(arr) + values_per_word - 1) // values_per_word) * values_per_word)
    padded_arr = np.zeros(padded_len, dtype=np.uint32)
    padded_arr[:len(arr)] = arr

    packed = []
    for i in range(0, padded_len, values_per_word):
        word = 0
        for j in range(values_per_word):
            word |= (padded_arr[i + j] & ((1 << bitwidth) - 1)) << (bitwidth * j)
        packed.append(word)

    return np.array(packed, dtype=np.uint32)


def get_param_mem_from_node(model: ModelWrapper, node: NodeProto) -> np.ndarray:
    """
    Get the memory representation of the parameters of a node that requires streaming.
    The memory representation is a 1D array of 32-bit unsigned integers."""

    mem = np.array([], dtype=np.uint32)

    if node.op_type == "Conv":

        if len(node.input) > 2:
            bias_quant_node = model.find_producer(node.input[2])
            bias_params = get_quant_params(bias_quant_node, model)
            bias_array = model.get_initializer(bias_quant_node.input[0])
            quant_bias_array = quant_array(
                bias_array,
                scale=bias_params["scale"],
                zeropt=bias_params["zeropt"],
                bitwidth=bias_params["bitwidth"],
                signed=bias_params["signed"],
                narrow=bias_params["narrow"],
                rounding_mode=bias_params["rounding_mode"]
            )
            mem = np.concatenate((mem, pack_values_to_int32words(quant_bias_array, bias_params["bitwidth"])))

        weight_quant_node = model.find_producer(node.input[1])
        weight_params = get_quant_params(weight_quant_node, model)

        weight_array = model.get_initializer(weight_quant_node.input[0])
        quant_weight_array = quant_array(
            weight_array,
            scale=weight_params["scale"],
            zeropt=weight_params["zeropt"],
            bitwidth=weight_params["bitwidth"],
            signed=weight_params["signed"],
            narrow=weight_params["narrow"],
            rounding_mode=weight_params["rounding_mode"]
        )

        mem = np.concatenate((mem, pack_values_to_int32words(quant_weight_array, weight_params["bitwidth"])))

    return mem
    

class AddStreamingParams(Transformation):
    """ A transformation pass that adds the logic to handle streaming parameters at startup.
    Each node with parameters will have an associated ParamStream node that is in charge of
    streaming the parameters to the node.
    """ 
    def __init__(self, nn2fpga_root: str = "/tmp"):
        """
        Initializes the AddStreamingParams transformation.
        Args:
            nn2fpga_root (str): The root directory of nn2FPGA.
        """
        super().__init__()
        self.nn2fpga_root = nn2fpga_root

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        sequential_streaming = list()
        uint32_mem = np.array([], dtype=np.uint32)
        
        # Find all nodes with parameters that need streaming
        # and collect them in a list.
        for node in model.graph.node:
            if node.op_type in NODE_WITH_PARAMS:
                node_mem = get_param_mem_from_node(model, node)
                uint32_mem = np.concatenate(
                    (uint32_mem, node_mem)
                )
                sequential_streaming.append((node, node_mem.size))

        if len(sequential_streaming) == 0:
            return (model, False)
        
        # Add an input to the model for the streaming parameters.
        param_stream_input = helper.make_tensor_value_info(
            "param_stream", 
            TensorProto.INT32,
            [len(uint32_mem)]
        )
        model.graph.input.append(param_stream_input)

        # Create a ParamStream node for each node with parameters.
        input_stream = [param_stream_input.name]
        params_to_shift = uint32_mem.size
        for node, params_size in sequential_streaming[:-1]:
            node_par = get_par_attributes(node)
            param_stream_node_name = node.name + "_param"
            output_stream = [f"{node.name}_stream"]
            output_stream.append(f"{node.name}_shift_out")

            # Create the ParamStream node.
            param_stream_node = helper.make_node(
                "ParamStream",
                inputs=input_stream,
                outputs=output_stream,
                name=param_stream_node_name,
                ich_par=node_par.get("ich_par", None),
                och_par=node_par.get("och_par", None),
            )

            # Add the new input to the node with parameters.
            node.input.append(output_stream[0])

            # Add the ParamStream node to the graph.
            model.graph.node.append(param_stream_node)
            input_stream = [output_stream[1]]  # The next input is the first output of the current ParamStream node

            params_to_shift -= params_size
            model.set_tensor_shape(
                output_stream[0],
                [params_size]
            )
            model.set_tensor_shape(
                output_stream[1],
                [params_to_shift]
            )
        else:
            # For the last node we do not need the shift_out output,
            # we just need the stream output.

            node, params_size = sequential_streaming[-1]
            node_par = get_par_attributes(node)
            param_stream_node_name = node.name + "_param"
            output_stream = [f"{node.name}_stream"]

            # Create the ParamStream node.
            param_stream_node = helper.make_node(
                "ParamStream",
                inputs=input_stream,
                outputs=output_stream,
                name=param_stream_node_name,
                ich_par=node_par.get("ich_par", None),
                och_par=node_par.get("och_par", None),
            )

            # Add the new input to the node with parameters.
            node.input.append(output_stream[0])

            # Add the ParamStream node to the graph.
            model.graph.node.append(param_stream_node)
            
            model.set_tensor_shape(
                output_stream[0],
                [params_size]
            )
        
        # Sort the graph.
        model = model.transform(SortGraph())


        os.system(f"mkdir -p {self.nn2fpga_root}/params/")
        np.save(f"{self.nn2fpga_root}/params/streaming_params.npy", uint32_mem)
        
        # For c++ testbench, we need to save the parameters in a binary file.
        with open(f"{self.nn2fpga_root}/params/streaming_params.bin", "wb") as file:
            file.write(uint32_mem.tobytes())
        return (model, False)

            


            