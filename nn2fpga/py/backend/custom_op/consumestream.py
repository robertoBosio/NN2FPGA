from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp

class ConsumeStream(CustomOp):
    """ Node consuming a streaming tensor to an axi lite interface. """
    
    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Identity", [node.input[0]], [node.output[0]])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        context[out_name] = inp

    def verify_node(self):
        pass