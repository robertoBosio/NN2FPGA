from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import remove_by_name
from qonnx.transformation.infer_shapes import InferShapes
from onnx import helper, numpy_helper, NodeProto
import numpy as np


def extend_weight_initializer_to_4d(model: ModelWrapper, node: NodeProto) -> None:
    """
    Given a weight Quant node name, modifies its associated initializer from
    [out_ch, in_ch] to [out_ch, in_ch, 1, 1] in-place in the model.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
        node (onnx.NodeProto): The Quant node containing the weight initializer.

    """

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
    weight_4d = weight_array.reshape(
        (weight_array.shape[0], weight_array.shape[1], 1, 1)
    )

    # Replace initializer
    model.set_initializer(weight_input_name, weight_4d)

    # Per-channel quantization.
    if scale_array.size > 1:
        scale_4d = scale_array.reshape(
            (scale_array.shape[0], scale_array.shape[1], 1, 1)
        )
        model.set_initializer(node.input[1], scale_4d)
    
    # Asymmetric quantization.
    if zeropt_array.size > 1:
        zeropt_4d = zeropt_array.reshape(
            (zeropt_array.shape[0], zeropt_array.shape[1], 1, 1)
        )
        model.set_initializer(node.input[2], zeropt_4d)


def replace_gemm_with_conv(model: ModelWrapper, node: NodeProto) -> None:
    """
    Replaces a Gemm node with a Conv node using the same inputs,
    assuming the model is reshaped for 4D activations and weights.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
        node (onnx.NodeProto): The Gemm node to be replaced.
    """
    graph = model.graph

    # Extract inputs and output
    act_input = node.input[0]  # [N, C, 1, 1] expected
    weight_input = node.input[1]  # [out_ch, in_ch, 1, 1]
    bias_input = node.input[2] if len(node.input) > 2 else None
    output = node.output[0]

    # Build Conv node
    conv_inputs = [act_input, weight_input]
    if bias_input:
        conv_inputs.append(bias_input)

    conv_node = helper.make_node(
        "Conv",
        inputs=conv_inputs,
        outputs=[output],
        name=node.name + "_as_conv",
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    # Replace node in graph
    new_nodes = []
    for generic_node in graph.node:
        if generic_node.name == node.name:
            new_nodes.append(conv_node)
        else:
            new_nodes.append(generic_node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)


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
    for output in node.output:
        consumers = model.find_consumers(output)
        for consumer in consumers:
            for i, consumer_input in enumerate(consumer.input):
                if consumer_input == output:
                    # Replace the input with the Flatten/Reshape node's input
                    consumer.input[i] = node.input[0]

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

class FullyConnectedToConv(Transformation):
    """
    A transformation that converts fully connected layers (Flatten/Reshape -> Quant -> Gemm) to convolutional layers
    by replacing the fully connected nodes with their convolutional equivalents.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for node in model.graph.node:

            # Identify the subgraph pattern: Flatten/Reshape -> Quant -> Gemm
            if node.op_type in ["Flatten", "Reshape"]:
                quant_node = model.find_consumer(node.output[0])

                if quant_node is not None and quant_node.op_type == "Quant":
                    gemm_node = model.find_consumer(quant_node.output[0])

                    if gemm_node is not None and gemm_node.op_type == "Gemm":

                        gemm_weight_node = model.find_producer(gemm_node.input[1])
                        extend_weight_initializer_to_4d(model, gemm_weight_node)

                        # Create a new Conv node
                        replace_gemm_with_conv(model, gemm_node)

                        # Remove the Flatten/Reshape and Quant nodes
                        remove_flatten_reshape(model, node)

        remove_all_shape_info(model)
        return (model, False)
