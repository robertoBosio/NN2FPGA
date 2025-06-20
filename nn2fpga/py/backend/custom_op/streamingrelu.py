import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper

class StreamingRelu(CustomOp):
    """ Node implementing the output-stationary convolution operation. """

    @staticmethod
    def from_onnx_node(onnx_node):
        """ Create a StreamingRelu instance from an ONNX node.

        Args:
            onnx_node: The ONNX node to convert.

        Returns:
            StreamingRelu: An instance of the StreamingRelu class.
        """
        if onnx_node.op_type != "Relu":
            raise ValueError(f"Expected Relu node to be converted in StreamingRelu, got {onnx_node.op_type}")

        return helper.make_node(
            "StreamingRelu",
            domain="backend.custom_op",
            inputs=onnx_node.input,
            outputs=onnx_node.output,
            name=onnx_node.name,
        )

    def get_nodeattr_types(self):
        return {
            # Custom attributes for parallelization of StreamingRelu
            "och_par" : ("i", False, 1),
            "w_par" : ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Relu",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard conv node to compute the result
        node = self.onnx_node
        node_conv = helper.make_node(
            "Relu",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_globaleaveragepool = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_relu = qonnx_make_model(graph_globaleaveragepool, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_relu.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass
