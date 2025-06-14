from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper

class TensorDuplicator(CustomOp):
    """ Node duplicating a tensor to ensure that each consumer gets a separate copy. """

    def get_nodeattr_types(self):
        return {
            "copies": ("i", True, 2),
        }

    def infer_node_datatype(self, model: ModelWrapper):
        node = self.onnx_node
        in_dtype = model.get_tensor_datatype(node.input[0])
        for out in node.output:
            model.set_tensor_datatype(out, in_dtype)
    
    def make_shape_compatible_op(self, model: ModelWrapper):
        node = self.onnx_node
        shape_compatible_nodes = [] 
        for i in range(self.get_nodeattr("copies")):
            identity_node = helper.make_node(
                "Identity",
                inputs=[node.input[0]],
                outputs=[node.output[i]],
                name=f"{node.name}_identity_{i}"
            )
            shape_compatible_nodes.append(identity_node)
        return shape_compatible_nodes

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_name = node.input[0]
        input_val = context[input_name]
        num_copies = self.get_nodeattr("copies")

        for i in range(num_copies):
            out_name = node.output[i]
            context[out_name] = input_val.copy()
    
    def verify_node(self):
        pass