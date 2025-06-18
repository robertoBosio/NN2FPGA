import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model

class StreamingConv(CustomOp):
    """ Node implementing the output-stationary convolution operation. """

    def get_nodeattr_types(self):
        return {
            # Standard ONNX attributes for Conv
            "dilation": ("ints", True, [1, 1]),
            "group": ("i", True, 1),
            "kernel_shape": ("ints", True, 1),
            "pads": ("ints", True, [0, 0]),
            "strides": ("ints", True, 1),

            # Custom attributes for quantization of StreamingConv
            "bitwidth_in" : ("i", False, 8),
            "bitwidth_out" : ("i", False, 8),
            "narrow_in" : ("i", False, 0),
            "narrow_out" : ("i", False, 0),
            "rounding_mode_in" : ("s", False, "ROUND"),
            "rounding_mode_out" : ("s", False, "ROUND"),
            "scale_in" : ("f", False, 1.0),
            "scale_out" : ("f", False, 1.0),
            "signed_in" : ("i", False, 0),
            "signed_out" : ("i", False, 0),
            "zeropt_in" : ("i", False, 0),
            "zeropt_out" : ("i", False, 0),

            # Custom attributes for parallelization of StreamingConv
            "ich_par" : ("i", False, 1),
            "och_par" : ("i", False, 1),
            "w_par" : ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Conv",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            dilation=self.get_nodeattr("dilation"),
            group=self.get_nodeattr("group"),
            kernel_shape=self.get_nodeattr("kernel_shape"),
            pads=self.get_nodeattr("pads"),
            strides=self.get_nodeattr("strides"),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard conv node to compute the result
        node = self.onnx_node
        node_conv = helper.make_node(
            "Conv",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            dilation=self.get_nodeattr("dilation"),
            group=self.get_nodeattr("group"),
            kernel_shape=self.get_nodeattr("kernel_shape"),
            pads=self.get_nodeattr("pads"),
            strides=self.get_nodeattr("strides"),
        )
        
        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        graph_conv = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_avgpool = qonnx_make_model(graph_conv, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_avgpool.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass