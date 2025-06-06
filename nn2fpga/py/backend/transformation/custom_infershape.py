from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

class CustomInferShapes(Transformation):
    """
    A shape‐inference pass that hides the nodes in nn2fpga domain by
    substituting them with their shape‐compatible ONNX equivalent,
    runs onnx.shape_inference, and then restores the original nn2fpga nodes.
    Similar to qonnx.transformation.infer_shapes.InferShapes.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        subs = []
        node_ind = 0
        for node in model.graph.node:
            node_ind += 1
            if "backend" in node.domain:
                
                # Replace nn2fpga custom op with its shape-compatible ONNX equivalent
                nn2fpga_node = getCustomOp(node)
                onnx_node = nn2fpga_node.make_shape_compatible_op(model)

                if isinstance(onnx_node, list):
                    # If the shape-compatible op returns a list (TensorDuplicator), we need to handle each node separately
                    node_list = []
                    for i, onnx_sub_node in enumerate(onnx_node):
                        node_list.append(onnx_sub_node)
                        model.graph.node.insert(node_ind + 1, onnx_sub_node)
                    subs.append([node, node_list, node_ind])

                else:
                    # If it's a single node, we can handle it directly
                    subs.append([node, onnx_node, node_ind])
                    model.graph.node.insert(node_ind, onnx_node)
                
                model.graph.node.remove(node)

        inferred = model.transform(InferShapes())

        # Restore the original nn2fpga nodes
        for nn2fpga_node, onnx_node, node_ind in subs:
            inferred.graph.node.insert(node_ind, nn2fpga_node)
            if isinstance(onnx_node, list):
                # If the shape-compatible op returned a list, we need to restore each node
                for sub_node in onnx_node:
                    inferred.graph.node.remove(sub_node)
            else:
                # If it's a single node, we can remove it directly
                inferred.graph.node.remove(onnx_node)

        return (inferred, False)
