import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.custom_op.register_rewrite_rule import register_rules
from onnxscript.rewriter import pattern

class StreamingConv(CustomOp):
    """ Node implementing the output-stationary convolution operation. """

    @staticmethod
    def pattern(
        op,
        x,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
        w_value,
        w_scale,
        w_zeropt,
        w_bitwidth,
        b_value,
        b_scale,
        b_zeropt,
        b_bitwidth,
        w_signed,
        w_narrow,
        w_rounding_mode,
        b_signed,
        b_narrow,
        b_rounding_mode,
    ):
        w_quant = op.Quant(
            w_value,
            w_scale,
            w_zeropt,
            w_bitwidth,
            signed=w_signed,
            narrow=w_narrow,
            rounding_mode=w_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        b_quant = op.Quant(
            b_value,
            b_scale,
            b_zeropt,
            b_bitwidth,
            signed=b_signed,
            narrow=b_narrow,
            rounding_mode=b_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        y = op.Conv(
            x,
            w_quant,
            b_quant,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            _allow_other_attributes=True,
        )
        return y

    @staticmethod
    def rewrite(
        op,
        x,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
        w_value,
        w_scale,
        w_zeropt,
        w_bitwidth,
        b_value,
        b_scale,
        b_zeropt,
        b_bitwidth,
        w_signed,
        w_narrow,
        w_rounding_mode,
        b_signed,
        b_narrow,
        b_rounding_mode,
    ):

        return op.StreamingConv(
            x,
            w_value,
            w_scale,
            w_zeropt,
            w_bitwidth,
            b_value,
            b_scale,
            b_zeropt,
            b_bitwidth,
            w_signed=w_signed.value,
            w_narrow=w_narrow.value,
            w_rounding_mode=w_rounding_mode.value,
            b_signed=b_signed.value,
            b_narrow=b_narrow.value,
            b_rounding_mode=b_rounding_mode.value,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            _domain="backend.custom_op",
        )

    @register_rules
    def _rewriter_rules():
        return [
            pattern.RewriteRule(
                StreamingConv.pattern,
                StreamingConv.rewrite,
            )
        ]

    def get_nodeattr_types(self):
        return {
            # Standard ONNX attributes for Conv
            "dilations": ("ints", True, [1, 1]),
            "group": ("i", True, 1),
            "kernel_shape": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0]),
            "strides": ("ints", True, [1, 1]),

            # Custom attributes for quantization of weights
            "w_signed": ("i", True, 0),  # 0: unsigned, 1: signed
            "w_narrow": ("i", True, 0),  # 0: full range, 1: narrow range
            "w_rounding_mode": ("s", True, "ROUND"),

            # Custom attributes for quantization of bias
            "b_signed": ("i", False, 0),  # 0: unsigned, 1: signed
            "b_narrow": ("i", False, 0),  # 0: full range, 1: narrow range
            "b_rounding_mode": ("s", False, "ROUND"),

            # Custom attributes for parallelization of StreamingConv
            "ich_par" : ("i", False, 1),
            "och_par" : ("i", False, 1),
            "w_par" : ("i", False, 1),

            # Custom attributes for zero point folding into bias
            "asym_folding": ("i", False, 0),  # 0: no folding, 1: fold zeropt into bias
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        input_list = []
        if len(node.input) == 5:
            # Conv without bias
            input_list = [node.input[0], node.input[1]] + node.input[2:5]
        elif len(node.input) == 9:
            # Conv with bias
            input_list = [node.input[0], node.input[1], node.input[6]] + node.input[2:6] + node.input[7:9]
        else:
            raise ValueError(
                f"Unexpected number of inputs for StreamingConv node {node.name}: {len(node.input)}"
            )

        return helper.make_node(
            "Conv",
            inputs=input_list,
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
            inputs=(
                [node.input[0], node.input[1]]
                if len(node.input) == 5
                else [node.input[0], node.input[1], node.input[6]]
            ),
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
        if len(node.input) > 5:
            bias_values = context[node.input[6]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        weight = helper.make_tensor_value_info(
            node.input[1], TensorProto.FLOAT, weight_values.shape
        )
        if len(node.input) > 5:
            bias = helper.make_tensor_value_info(
                node.input[6], TensorProto.FLOAT, bias_values.shape
            )

        graph_conv = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inp, weight, bias] if len(node.input) > 5 else [inp, weight],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_conv = qonnx_make_model(graph_conv, **onnx_kwargs)
        if len(node.input) > 5:
            idict = {node.input[0]: inp_values,
                     node.input[1]: weight_values,
                     node.input[6]: bias_values}
        else:
            idict = {node.input[0]: inp_values, 
                    node.input[1]: weight_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_conv.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass
