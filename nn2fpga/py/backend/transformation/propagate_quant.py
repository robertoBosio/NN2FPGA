from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, GiveUniqueParameterTensors
from onnx import numpy_helper, helper, NodeProto

QUANT_INVARIANT_NODES = [
    "Relu",
    "Reshape",
    "Flatten",
    "Identity",
    "MaxPool",
    "GlobalMaxPool",
    "Concat",
    "Transpose",
    "Split"
]

def get_non_constant_inputs(node: NodeProto, model: ModelWrapper) -> list[str]:
    """Get non-constant inputs of a node."""
    init_dict = {init.name: init for init in model.model.graph.initializer}
    return [inp for inp in node.input if inp not in init_dict]

def get_non_constant_outputs(node: NodeProto, model: ModelWrapper) -> list[str]:
    """Get non-constant outputs of a node."""
    init_dict = {init.name: init for init in model.model.graph.initializer}
    return [out for out in node.output if out not in init_dict]

def get_quant_params(node: NodeProto, model: ModelWrapper) -> dict:
    """ Get quantization parameters from a quantization node. """
    init_dict = {init.name: init for init in model.model.graph.initializer}
    def get_scalar(name):
        arr = numpy_helper.to_array(init_dict[name])
        return arr.item() if arr.size == 1 else None

    # Handle scale, zeropt, bitwidth
    scale = zeropt = bitwidth = None

    if len(node.input) > 1 and node.input[1] in init_dict:
        scale = get_scalar(node.input[1])
    if len(node.input) > 2 and node.input[2] in init_dict:
        zeropt = get_scalar(node.input[2])
    if len(node.input) > 3 and node.input[3] in init_dict:
        bitwidth = get_scalar(node.input[3])
    attr_dict = {a.name: a.i for a in node.attribute}
    signed = attr_dict.get("signed", 0)
    narrow = attr_dict.get("narrow", 0)
    return dict(scale=scale, zeropt=zeropt, bitwidth=bitwidth,
                signed=signed, narrow=narrow)

def same_quant(q1: dict, q2: dict) -> bool:
    """ Check if two quantization parameters are the same. """
    return all(q1[k] == q2[k] for k in q1)

class PropagateQuant(Transformation):
    """Propagate quantization parameters through quantization invariant nodes. """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        is_changed = False

        for i, node in enumerate(graph.node):
            if node.op_type not in QUANT_INVARIANT_NODES:
                continue

            producers = [model.find_producer(inp) for inp in get_non_constant_inputs(node, model)]
            consumers = list()
            for consumer in get_non_constant_outputs(node, model):
                consumers.extend(model.find_consumers(consumer))

            if (
                len(producers) == 0
                or len(consumers) == 0
                or any(p is None for p in producers)
                or any(c is None for c in consumers)
            ):
                # If any producer or consumer is None, skip this node
                continue

            # To propagate forward a quantization, all the inputs must be quantized
            # and with the same quantization parameters, while the consumers must not be quantized
            all_producer_quantized = all(
                p.op_type in ["IntQuant", "Quant"] for p in producers
            )
            if all_producer_quantized:
                for producer in producers[1:]:
                    if not same_quant(get_quant_params(producers[0], model), get_quant_params(producer, model)):
                        all_producer_quantized = False
                        break
            any_consumers_quantized = any(
                c.op_type in ["IntQuant", "Quant"] for c in consumers
            )
            forward_quantization_possible = all_producer_quantized and not any_consumers_quantized
            
            if forward_quantization_possible:
                # If the producer is quantized and consumers are not, propagate quantization

                quant_node = producers[0]
                for out in get_non_constant_outputs(node, model):
                    consumers = model.find_consumers(out)
                    print(f"Propagating forward quantization of {quant_node.name} to tensor {out}")

                    # Insert new quant node after current node
                    new_quant_output = out + "_quant_forward_propagated"
                    new_quant_node = helper.make_node(
                        quant_node.op_type,
                        inputs=[out, quant_node.input[1], quant_node.input[2], quant_node.input[3]],  # Use the same quantization parameters
                        outputs=[new_quant_output],
                        name=quant_node.name + "_forward_propagated",
                        domain=quant_node.domain,
                    )
                    for attr in quant_node.attribute:
                        new_quant_node.attribute.append(attr)

                    # Rewire the consumer to use the new quantized output
                    for consumer in consumers:
                        for j, inp in enumerate(consumer.input):
                            if inp == out:
                                consumer.input[j] = new_quant_output

                    # Insert the new quant node after the current node
                    graph.node.insert(i + 1, new_quant_node)
                    is_changed = True

            # To propagate backward a quantization, all the outputs must be quantized
            # and with the same quantization parameters, while the producers must not be quantized.
            all_consumers_quantized = all(
                c.op_type in ["IntQuant", "Quant"] for c in consumers
            )
            if all_consumers_quantized:
                for consumer in consumers[1:]:
                    if not same_quant(get_quant_params(consumers[0], model), get_quant_params(consumer, model)):
                        all_consumers_quantized = False
                        break
            any_producer_quantized = any(
                p.op_type in ["IntQuant", "Quant"] for p in producers
            )
            backward_quantization_possible = all_consumers_quantized and not any_producer_quantized

            if backward_quantization_possible:
                quant_node = consumers[0]

                for inp in get_non_constant_inputs(node, model):
                    producer = model.find_producer(inp)

                    print(f"Propagating backward quantization of {quant_node.name} to tensor {inp}")

                    # Insert new quant node before current node
                    new_quant_output = inp + "_quant_backward_propagated"
                    new_quant_node = helper.make_node(
                        quant_node.op_type,
                        inputs=[inp, quant_node.input[1], quant_node.input[2], quant_node.input[3]],  # Use the same quantization parameters
                        outputs=[new_quant_output],
                        name=quant_node.name + "_backward_propagated",
                        domain=quant_node.domain,
                    )
                    for attr in quant_node.attribute:
                        new_quant_node.attribute.append(attr)

                    # Rewire the producer to use the new quantized output
                    for j, node_input in enumerate(node.input):
                        if node_input == inp:
                            node.input[j] = new_quant_output

                    # Insert the new quant node before the current node
                    graph.node.insert(i, new_quant_node)
                    is_changed = True

        if is_changed:
            # If any changes were made, sort the graph to maintain a valid topological order
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(GiveUniqueParameterTensors())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        return (model, is_changed)
