from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper

class InsertProduceStream(Transformation):
    """
    Inserts a ProduceStream node for each input tensor in the model.
    This node will stream the input tensor read from an AXI Lite interface.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        new_nodes = []
        for i, inp in enumerate(model.graph.input):
            orig_input_name = inp.name
            produce_stream_output = f"{orig_input_name}_streamed"

            # Create the custom node ProduceStream
            produce_node = helper.make_node(
                op_type="ProduceStream",
                domain="backend.custom_op",
                inputs=[orig_input_name],
                outputs=[produce_stream_output],
                normalize=0,
                name=f"ProduceStream_{i}",
            )

            # Replace all uses of this input
            for node in model.graph.node:
                node_inputs = list(node.input)
                for j, node_in in enumerate(node_inputs):
                    if node_in == orig_input_name:
                        node.input[j] = produce_stream_output

            new_nodes.append(produce_node)

        # Insert all new nodes at the beginning
        for node in reversed(new_nodes):
            model.graph.node.insert(0, node)

        return (model, False)
