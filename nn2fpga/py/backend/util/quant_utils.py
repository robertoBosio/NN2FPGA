from onnx import numpy_helper, helper
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import remove_by_name
from qonnx.custom_op.registry import getCustomOp
import numpy as np

def get_quant_params(node: NodeProto, model: ModelWrapper) -> dict:
    """Get quantization parameters from a quantization node."""
    
    scale = zeropt = bitwidth = None

    if len(node.input) > 1:
        scale = model.get_initializer(node.input[1])
    if len(node.input) > 2:
        zeropt = model.get_initializer(node.input[2])
    if len(node.input) > 3:
        bitwidth = model.get_initializer(node.input[3])
        if bitwidth.size > 1:
            raise ValueError(
                f"Bitwidth for node {node.name} is not a scalar: {bitwidth}"
            )
        bitwidth = int(bitwidth.item())
    signed = getCustomOp(node).get_nodeattr("signed")
    narrow = getCustomOp(node).get_nodeattr("narrow")
    rounding_mode = getCustomOp(node).get_nodeattr("rounding_mode")
    return dict(
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode,
    )

def is_constant_input_node(model: ModelWrapper, node: NodeProto) -> bool:
    """Check if the node has only constant inputs.
    It is used to distinguish between Quant nodes on the activation and
    Quant nodes on the parameters (weights and biases).
    """
    init_names = {init.name for init in model.graph.initializer}
    return all(inp in init_names for inp in node.input)
