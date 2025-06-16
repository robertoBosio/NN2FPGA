from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from onnx import numpy_helper, helper, NodeProto
from backend.util.quant_utils import is_constant_input_node, get_quant_attributes, set_quant_attributes, check_quant_attributes
from backend.transformation.propagate_quant import QUANT_INVARIANT_NODES
import numpy as np

class InferQuant(Transformation):
    """ Infer quantization parameters for nodes without quantization parameters that are quant invariant. """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # Find all nodes in the model
        nodes = model.graph.node
        for node in nodes:
            if node.op_type not in QUANT_INVARIANT_NODES:
                continue

            if check_quant_attributes(node, "out") and check_quant_attributes(node, "in"):
                continue  # Skip nodes that already have quantization parameters

            # Infer quantization parameters based on the producer node
            predecessors = model.find_direct_predecessors(node)
            if predecessors is None or len(predecessors) > 1:
                continue

            producer = predecessors.pop()
            if check_quant_attributes(producer, "out"):
                # If the producer has quantization parameters, infer them for the current node
                set_quant_attributes(node, "in", get_quant_attributes(producer, "out"))
                set_quant_attributes(node, "out", get_quant_attributes(producer, "out"))
        

        return (model, False)