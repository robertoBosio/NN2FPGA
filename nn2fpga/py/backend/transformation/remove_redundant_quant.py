from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name
from onnx import numpy_helper

class RemoveRedundantQuant(Transformation):
    """Remove consecutive IntQuant nodes with identical quantization parameters."""

    def get_quant_params(self, node, model):
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

    def same_quant(self, q1, q2):
        return all(q1[k] == q2[k] for k in q1)

    def apply(self, model):
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
            q1 = self.get_quant_params(prev_node, model)
            q2 = self.get_quant_params(node, model)

            if self.same_quant(q1, q2):
                # Remove node: patch all consumers of node.output[0]
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
