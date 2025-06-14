from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from onnx import numpy_helper, helper, NodeProto
from backend.util.quant_utils import get_quant_params, is_constant_input_node, get_quant_attributes, set_quant_attributes
import numpy as np

def check_node_has_folded_quant(node: NodeProto, model: ModelWrapper, direction: str) -> bool:
    """Check if a node has already folded quantization parameters."""
    pass


class FoldQuant(Transformation):
    """ Fold quantization parameters into layers. """
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        is_everything_folded = True

        # Find all Quant nodes in the model
        quants = model.get_nodes_by_op_type("Quant")
        for quant in quants:

            if is_constant_input_node(model, quant):
                continue  # Skip Quant nodes on the parameters (weights and biases)

            # Get quantization parameters
            quant_params = get_quant_params(quant, model)

            # Fold into the producer node the parameters
            producer = model.find_producer(quant.input[0])
            quant_params_dict = {}
            for param_name, param_value in quant_params.items():
                if param_value is not None:
                    value = param_value.item() if param_value is not None and isinstance(param_value, np.ndarray) else param_value
                    quant_params_dict[param_name] = value

            if producer is None:
                print("No producer found for Quant node:", quant.name)
                continue

            if producer.op_type == "Quant":
                print("Skipping folding of nested Quant node:", quant.name)
                is_everything_folded = False
                continue

            # Check if the producer already has the quantization parameters
            # If it does, check if they are different
            # If they are different, skip folding
            producer_attr_dict = get_quant_attributes(producer, "out")
            producer_already_has_quant = any(
                param_value is not None for param_value in producer_attr_dict.values()
            )
            producer_already_has_different_quant = False
            if producer_already_has_quant:
                producer_already_has_different_quant = any(
                    not np.isclose(
                        quant_params_dict[param_name], producer_attr_dict[param_name]
                    )
                    for param_name in quant_params_dict.keys()
                )
            
            if not producer_already_has_quant:
                set_quant_attributes(producer, "out", quant_params_dict)
            elif producer_already_has_quant and producer_already_has_different_quant:
                print(f"Producer {producer.name} already has different quantization parameters for Quant node {quant.name}. Skipping folding.")
                is_everything_folded = False
                continue

            consumer = model.find_consumer(quant.output[0])
            if consumer.op_type == "Quant":
                print(f"Skipping folding into consumer {consumer.name} as it is a Quant node.")
                is_everything_folded = False
                continue

            # Check if the producer already has the quantization parameters
            # If it does, check if they are different
            # If they are different, skip folding
            consumer_attr_dict = get_quant_attributes(consumer, "in")
            consumer_already_has_quant = any(
                param_value is not None for param_value in consumer_attr_dict.values()
            )
            consumer_already_has_different_quant = False
            if consumer_already_has_quant:
                consumer_already_has_different_quant = any(
                    not np.isclose(
                        quant_params_dict[param_name], consumer_attr_dict[param_name]
                    )
                    for param_name in quant_params_dict.keys()
                )
            
            if not consumer_already_has_quant:
                set_quant_attributes(consumer, "in", quant_params_dict)
            elif consumer_already_has_quant and consumer_already_has_different_quant:
                print(f"consumer {consumer.name} already has different quantization parameters for Quant node {quant.name}. Skipping folding.")
                is_everything_folded = False
                continue
        
            print(f"Folding Quant node {quant.name} into producer {producer.name} and consumer {consumer.name}.")

            # Remove the Quant node
            for i, inp in enumerate(consumer.input):
                if inp == quant.output[0]:
                    consumer.input[i] = producer.output[0]
            graph.node.remove(quant)
                

        return (model, False)
