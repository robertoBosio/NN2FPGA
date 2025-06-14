from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.custom_op.consumestream import ConsumeStream
from onnx import helper
from qonnx.util.basic import get_by_name

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
                outputs=[consume_stream_output],
                inputs=[orig_output_name],
                name=f"ConsumeStream_{i}"
            )

            get_by_name(model.graph.output, orig_output_name).name = consume_stream_output 
            new_nodes.append(consume_node)

        # Insert all new nodes at the beginning
        for node in new_nodes:
            model.graph.node.append(node)

        return (model, False)