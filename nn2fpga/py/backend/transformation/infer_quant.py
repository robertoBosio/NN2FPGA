from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.transformation.propagate_quant import QUANT_INVARIANT_NODES
import numpy as np


class InferQuant(Transformation):
    """Infer quantization parameters for nodes without quantization parameters that are quant invariant."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # Find all nodes in the model
        nodes = model.graph.node
        for node in nodes:
            if node.op_type not in QUANT_INVARIANT_NODES:
                continue

            # Check if the node already has quantization parameters in output
            output_quant = model.get_tensor_datatype(node.output[0])
            if output_quant is not None:
                continue  # Skip nodes that already have quantization parameters

            # Infer quantization parameters based on the producer node
            input_quant = model.get_tensor_datatype(node.input[0])
            if input_quant is not None:
                # If the input has quantization parameters, set them for the output
                model.set_tensor_datatype(node.output[0], input_quant)

        return (model, False)
