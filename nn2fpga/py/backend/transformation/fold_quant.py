from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.quant_utils import (
    get_quant_params,
    is_constant_input_node,
)
from backend.core.tensor_quant import (
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
)
from backend.core.tensor_quant import TensorQuant
import numpy as np


class FoldQuant(Transformation):
    """Fold Quant node into tensors datatype."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        not_folded = 0
        folded = 0

        # Find all Quant nodes in the model
        quants = model.get_nodes_by_op_type("Quant")
        for quant in quants:

            if is_constant_input_node(model, quant):
                continue  # Skip Quant nodes on the parameters (weights and biases)

            # Get quantization parameters
            quant_params = get_quant_params(quant, model)

            expected_tensor_quant = TensorQuant(
                bitwidth=quant_params["bitwidth"],
                signed=quant_params["signed"],
                scale=quant_params["scale"],
                zeropt=quant_params["zeropt"],
                narrow_range=quant_params["narrow"],
                rounding=quant_params["rounding_mode"],
            )
            current_output_tensor_quant = get_custom_tensor_datatype(
                model, quant.output[0]
            )

            # Check if the output of the Quant node has already a quantization
            # If it does, check if they are different
            if (
                current_output_tensor_quant is not None
                and current_output_tensor_quant.get_canonical_name()
                != expected_tensor_quant.get_canonical_name()
            ):

                raise ValueError(
                    f"Quant node {quant.name} has a different quantization parameter than the output tensor {quant.output[0]}.\n"
                    f"Expected: {expected_tensor_quant.get_canonical_name()}, "
                    f"current_output: {current_output_tensor_quant.get_canonical_name() if current_output_tensor_quant else 'None'}"
                )

            # Set the quantization parameters on the output tensor
            set_custom_tensor_datatype(model, quant.output[0], expected_tensor_quant)

            # Check if the input of the Quant node has already a quantization
            current_input_tensor_quant = get_custom_tensor_datatype(
                model, quant.input[0]
            )

            if (
                current_input_tensor_quant is not None
                and current_input_tensor_quant.get_canonical_name()
                != expected_tensor_quant.get_canonical_name()
            ):
                # This case can happen if there are multiple Quant nodes with different parameters. It is not an error, but we cannot fold the Quant node,
                # and thsu we have to implement it in the hardware.
                not_folded += 1
                continue

            # Bypass and remove the Quant node
            producer = model.find_producer(quant.input[0])
            for i, out in enumerate(producer.output):
                if out == quant.input[0]:
                    producer.output[i] = quant.output[0]

            # Remove the Quant node from the graph
            graph.node.remove(quant)
            folded += 1

        print(
            f"Folded {folded} Quant nodes into tensor datatype."
        )
        if not_folded > 0:
            print(
                f"Could not fold {not_folded} Quant nodes due to multiple quantization parameters on the same tensor."
            )

        return (model, False)
