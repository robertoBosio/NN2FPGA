from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.custom_op.general.quant as qonnx_quant
from onnx import NodeProto, TensorProto, helper
from backend.util.par_utils import get_par_attributes
from backend.core.tensor_quant import TensorQuant
import numpy as np
from qonnx.custom_op.registry import getCustomOp

NODE_WITH_PARAMS = [
    "StreamingConv",
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
    """Quantize an input tensor to a specified bitwidth and return the quantized tensor."""

    # Let QONNX handle the quantization. This function return the dequantized tensor.
    inp_tensor = safe_int_quant_call(
        inp_tensor,
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode,
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
    bitwidth = int(bitwidth)
    if bitwidth > 32 or bitwidth <= 0:
        raise ValueError("bitwidth must be between 1 and 32")

    arr = arr.flatten()  # Ensure the input is a 1D array
    values_per_word = 32 // bitwidth  # Max number of values per word
    padded_len = int(
        ((len(arr) + values_per_word - 1) // values_per_word) * values_per_word
    )
    padded_arr = np.zeros(padded_len, dtype=np.uint32)
    padded_arr[: len(arr)] = arr

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
    custom_node = getCustomOp(node)
    if node.op_type == "StreamingConv":

        if len(node.input) > 5:
            quant_bias_array = quant_array(
                model.get_initializer(node.input[5]),
                scale=model.get_initializer(node.input[6]),
                zeropt=model.get_initializer(node.input[7]),
                bitwidth=model.get_initializer(node.input[8]),
                signed=custom_node.get_nodeattr("b_signed"),
                narrow=custom_node.get_nodeattr("b_narrow"),
                rounding_mode=custom_node.get_nodeattr("b_rounding_mode"),
            )
            mem = np.concatenate(
                (mem, pack_values_to_int32words(quant_bias_array, model.get_initializer(node.input[8]).item()))
            )

        weight_array = model.get_initializer(node.input[1])
        quant_weight_array = quant_array(
            weight_array,
            scale=model.get_initializer(node.input[2]),
            zeropt=model.get_initializer(node.input[3]),
            bitwidth=model.get_initializer(node.input[4]),
            signed=custom_node.get_nodeattr("w_signed"),
            narrow=custom_node.get_nodeattr("w_narrow"),
            rounding_mode=custom_node.get_nodeattr("w_rounding_mode"),
        )

        mem = np.concatenate(
            (mem, pack_values_to_int32words(quant_weight_array, model.get_initializer(node.input[4]).item()))
        )

    return mem

class AddStreamingParams(Transformation):
    """A transformation pass that adds the logic to handle streaming parameters at startup.
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
                uint32_mem = np.concatenate((uint32_mem, node_mem))
                sequential_streaming.append((node, node_mem.size))

        if len(sequential_streaming) == 0:
            return (model, False)

        # Add an input to the model for the streaming parameters.
        # Add also an initializer since it's constant.
        # The 'const_' string in the name is mandatory to recognize the initializer
        # as a special in the simulation flow.
        param_stream_input = helper.make_tensor_value_info(
            "const_param_stream", TensorProto.INT32, [len(uint32_mem)]
        )
        model.set_initializer("const_param_stream", uint32_mem)

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
            input_stream = [
                output_stream[1]
            ]  # The next input is the first output of the current ParamStream node

            params_to_shift -= params_size
            model.set_tensor_shape(output_stream[0], [params_size])
            model.set_tensor_shape(output_stream[1], [params_to_shift])
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

            model.set_tensor_shape(output_stream[0], [params_size])

        # Sort the graph.
        model = model.transform(SortGraph())

        # os.system(f"mkdir -p {self.nn2fpga_root}/params/")
        # np.save(f"{self.nn2fpga_root}/params/streaming_params.npy", uint32_mem)

        # # For c++ testbench, we need to save the parameters in a binary file.
        # with open(f"{self.nn2fpga_root}/params/streaming_params.bin", "wb") as file:
        #     file.write(uint32_mem.tobytes())
        return (model, False)
