from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.quant_utils import get_quant_params
from onnx import numpy_helper

class RemoveRedundantQuant(Transformation):
    """Remove consecutive IntQuant nodes with identical quantization parameters."""

    def same_quant(self, q1, q2):
        return all(q1[k] == q2[k] for k in q1)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.model.graph
        new_nodes = []
        skip_outputs = {}
        run_again = False

        for i, node in enumerate(graph.node):
            if node.op_type != "IntQuant" and node.op_type != "Quant":
                new_nodes.append(node)
                continue

            if len(new_nodes) == 0 or (new_nodes[-1].op_type != "IntQuant" and new_nodes[-1].op_type != "Quant"):
                new_nodes.append(node)
                continue

            prev_node = new_nodes[-1]

            # Check consecutive
            if node.input[0] != prev_node.output[0]:
                new_nodes.append(node)
                continue

            # Compare quantization parameters
            q1 = get_quant_params(prev_node, model)
            q2 = get_quant_params(node, model)

            if self.same_quant(q1, q2):
                # Remove node: patch all consumers of node.output[0]
                print(f"Removing redundant quant node {node.name} with identical quant params to {prev_node.name}")
                skip_outputs[node.output[0]] = prev_node.output[0]
                continue
            else:
                new_nodes.append(node)

        # Update inputs of downstream nodes
        for node in new_nodes:
            for i, inp in enumerate(node.input):
                if inp in skip_outputs:
                    node.input[i] = skip_outputs[inp]

        # Also patch model outputs if needed
        for output in graph.output:
            if output.name in skip_outputs:
                output.name = skip_outputs[output.name]

        if skip_outputs:
            run_again = True

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        return (model, run_again)
