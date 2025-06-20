import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper

class StreamingConv(CustomOp):
    """ Node implementing the output-stationary convolution operation. """

    @staticmethod
    def from_onnx_node(onnx_node):
        """ Create a StreamingConv instance from an ONNX node.

        Args:
            onnx_node: The ONNX node to convert.

        Returns:
            StreamingConv: An instance of the StreamingConv class.
        """
        if onnx_node.op_type != "Conv":
            raise ValueError(f"Expected Conv node to be converted in StreamingConv, got {onnx_node.op_type}")

        # Read the attributes from the ONNX node
        attr_dict = {attr.name: helper.get_attribute_value(attr) for attr in onnx_node.attribute}
        dilations = attr_dict.get("dilations", [1, 1])
        group = attr_dict.get("group", 1)
        kernel_shape = attr_dict.get("kernel_shape", [1, 1])
        pads = attr_dict.get("pads", [0, 0, 0, 0])
        strides = attr_dict.get("strides", [1, 1])

        return helper.make_node(
            "StreamingConv",
            domain="backend.custom_op",
            inputs=onnx_node.input,
            outputs=onnx_node.output,
            name=onnx_node.name,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

    def get_nodeattr_types(self):
        return {
            # Standard ONNX attributes for Conv
            "dilations": ("ints", True, [1, 1]),
            "group": ("i", True, 1),
            "kernel_shape": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0]),
            "strides": ("ints", True, [1, 1]),

            # Custom attributes for parallelization of StreamingConv
            "ich_par" : ("i", False, 1),
            "och_par" : ("i", False, 1),
            "w_par" : ("i", False, 1),

            # Custom attributes for zero point folding into bias
            "asym_folding": ("i", False, 0),  # 0: no folding, 1: fold zeropt into bias
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Conv",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            dilations=self.get_nodeattr("dilations"),
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
            dilations=self.get_nodeattr("dilations"),
            group=self.get_nodeattr("group"),
            kernel_shape=self.get_nodeattr("kernel_shape"),
            pads=self.get_nodeattr("pads"),
            strides=self.get_nodeattr("strides"),
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        weight_values = context[node.input[1]]
        if len(node.input) > 2:
            bias_values = context[node.input[2]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        weight = helper.make_tensor_value_info(
            node.input[1], TensorProto.FLOAT, weight_values.shape
        )
        if len(node.input) > 2:
            bias = helper.make_tensor_value_info(
                node.input[2], TensorProto.FLOAT, bias_values.shape
            )

        graph_conv = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inp, weight, bias] if len(node.input) > 2 else [inp, weight],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_conv = qonnx_make_model(graph_conv, **onnx_kwargs)
        if len(node.input) > 2:
            idict = {node.input[0]: inp_values, 
                     node.input[1]: weight_values, 
                     node.input[2]: bias_values}
        else:
            idict = {node.input[0]: inp_values, 
                    node.input[1]: weight_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_conv.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass
