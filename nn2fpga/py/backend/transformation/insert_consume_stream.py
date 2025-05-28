from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.custom_op.consumestream import ConsumeStream
from onnx import helper

class InsertConsumeStream(Transformation):
    """
    Inserts a ConsumeStream node for each output tensor in the model.
    This node will transform the streaming tensor into an AXI Lite compatible one.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        new_nodes = []
        for i, out in enumerate(model.graph.output):
            orig_output_name = out.name
            consume_stream_output = f"{orig_output_name}_streamed"

            # Create the custom node ConsumeStream
            consume_node = helper.make_node(
                op_type="ConsumeStream",
                domain="backend.custom_op",
                outputs=[orig_output_name],
                inputs=[consume_stream_output],
                ow_ops=1,
                och_ops=1,
                name=f"ConsumeStream_{i}"
            )

            # Replace all uses of this input
            for node in model.graph.node:
                node_outputs = list(node.output)
                for j, node_out in enumerate(node_outputs):
                    if node_out == orig_output_name:
                        node.output[j] = consume_stream_output

            new_nodes.append(consume_node)

        # Insert all new nodes at the beginning
        for node in new_nodes:
            model.graph.node.append(node)

        return (model, False)