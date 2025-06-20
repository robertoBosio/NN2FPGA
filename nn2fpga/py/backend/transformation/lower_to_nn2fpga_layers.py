from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.transformation.supported_partition import FPGA_SUPPORTED_OPS
from onnx import helper
from backend.custom_op.streamingconv import StreamingConv


class LowerToNN2FPGALayers(Transformation):
    """Lower the model to NN2FPGA layers."""

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """
        Apply the transformation to lower the model to NN2FPGA layers.

        Args:
            model (ModelWrapper): The model to transform.

        Returns:
            tuple: A tuple containing the transformed model and a boolean indicating if the transformation needs to be reapplied.
        """

        node_index = 0
        for node in model.graph.node:
            node_index += 1
            if node.op_type == "Conv":
                
                nn2fpga_node = StreamingConv.from_onnx_node(node)
                model.graph.node.insert(node_index, nn2fpga_node)
                model.graph.node.remove(node)
                

        return (model, False)
