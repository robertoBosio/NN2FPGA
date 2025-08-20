from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import remove_by_name
from backend.core.tensor_quant import is_constant_input_node
from onnx import helper, NodeProto
import numpy as np
import logging
logger = logging.getLogger(__name__)

def reshape_initializer(model: ModelWrapper, node: NodeProto, new_shape: tuple) -> None:
    """
    Given a weight Quant node name, modifies its associated initializer from
    [out_ch, in_ch] to [out_ch, in_ch, 1, 1] in-place in the model.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
        node (onnx.NodeProto): The Quant node containing the weight initializer.

    """

    if node.op_type != "Quant":
        raise ValueError(f"Node {node.name}, is not a Quant node.")

    weight_input_name = node.input[0]
    weight_array = model.get_initializer(weight_input_name)
    scale_array = model.get_initializer(node.input[1])
    zeropt_array = model.get_initializer(node.input[2])
    if weight_array is None:
        raise ValueError(f"Initializer {weight_input_name} not found in the model.")
    if scale_array is None:
        raise ValueError(
            f"Scales initializer {node.input[1]} not found in the model."
        )
    if zeropt_array is None:
        raise ValueError(
            f"Zero points initializer {node.input[2]} not found in the model."
        )

    if weight_array.ndim != 2:
        raise ValueError(
            f"Expected 2D weight initializer, got shape {weight_array.shape}"
        )

    # Reshape to 4D [out_ch, in_ch, 1, 1]
    weight_4d = weight_array.reshape(new_shape)

    # Replace initializer
    model.set_initializer(weight_input_name, weight_4d)

    # Per-channel quantization.
    if scale_array.size > 1:
        scale_4d = scale_array.reshape(new_shape)
        model.set_initializer(node.input[1], scale_4d)

    # Asymmetric quantization.
    if zeropt_array.size > 1:
        zeropt_4d = zeropt_array.reshape(new_shape)
        model.set_initializer(node.input[2], zeropt_4d)

def remove_flatten_reshape(model: ModelWrapper, node: NodeProto) -> None:
    """
    Removes Flatten or Reshape nodes from the model, as they are not needed
    after converting fully connected layers to convolutional layers.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
        node (onnx.NodeProto): The Flatten or Reshape node to be removed.
    """
    graph = model.graph

    # Connect the inputs of the Flatten/Reshape node to its consumers
    for input in node.input:
        producer = model.find_producer(input)
        for i, producer_output in enumerate(producer.output):
            if producer_output == input:
                # Replace the input with the Flatten/Reshape node's input
                producer.output[i] = node.output[0]

    # Remove the Flatten/Reshape node from the graph
    graph.node.remove(node)

def remove_all_shape_info(model: ModelWrapper) -> None:
    """
    Removes all inferrable shape/type information from the model's
    graph.output and and graph.value_info to enable re-inference.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
    """
    for tensor in model.get_all_tensor_names():
        if tensor not in [init.name for init in model.graph.initializer]:
            remove_by_name(model.graph.value_info, tensor)

    new_outputs = []
    for out in model.graph.output:
        if not out.type.HasField("tensor_type"):
            new_outputs.append(out)
            continue

        tensor_type = out.type.tensor_type
        elem_type = tensor_type.elem_type

        # Create output without shape
        new_vi = helper.make_tensor_value_info(
            name=out.name,
            elem_type=elem_type,
            shape=None  # No shape -> makes it completely unspecified
        )
        new_outputs.append(new_vi)

    # Replace graph outputs with shape-less versions
    del model.graph.output[:]
    model.graph.output.extend(new_outputs)


class FoldReshapeIntoInitializer(Transformation):
    """
    Fold Reshape nodes into initializers for weights and biases of Quant nodes.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        nodes_to_remove = []
        for node in model.graph.node:
            if node.op_type in ["Flatten", "Reshape"]:
                quant_node = model.find_producer(node.input[0])

                if (
                    quant_node is not None
                    and quant_node.op_type == "Quant"
                    and is_constant_input_node(model, quant_node)
                ):
                    new_shape = model.get_tensor_shape(node.output[0])
                    old_shape = model.get_tensor_shape(quant_node.input[0])

                    if new_shape is None:
                        raise ValueError(
                            f"Cannot infer shape for node {node.name} output {node.output[0]}"
                        )

                    if old_shape is None:
                        raise ValueError(
                            f"Cannot infer shape for node {quant_node.name} input {quant_node.input[0]}"
                        )

                    if np.prod(new_shape) != np.prod(old_shape):
                        raise ValueError(
                            f"Shape mismatch: {new_shape} vs {old_shape} for node {node.name}"
                        )

                    # Modify the weight initializer to match the new shape
                    reshape_initializer(model, quant_node, new_shape)
                    nodes_to_remove.append(node)

                    logger.info(
                        f"Folded Reshape node {node.name} into initializer of Quant node {quant_node.name}."
                    )

        # Remove Flatten/Reshape nodes after folding
        for node in nodes_to_remove:
            remove_flatten_reshape(model, node)

        return (model, False)
