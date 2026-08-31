from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.transforms.lower_to_nn2fpga_layers import LowerToNN2FPGALayers


def _build_sigmoid_model() -> ModelWrapper:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])

    sigmoid = helper.make_node(
        "Sigmoid",
        inputs=["input"],
        outputs=["output"],
        name="sigmoid_node",
    )

    graph = helper.make_graph(
        [sigmoid],
        name="sigmoid_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _build_hardsigmoid_mul_model(alpha: float = 0.1666666716337204, b_value: float = 1.0001220703125) -> ModelWrapper:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])

    B = helper.make_tensor("B", TensorProto.FLOAT, [], [b_value])

    hardsigmoid = helper.make_node(
        "HardSigmoid",
        inputs=["input"],
        outputs=["hardsigmoid_out"],
        name="hardsigmoid_node",
        alpha=alpha,
    )
    mul = helper.make_node(
        "Mul",
        inputs=["hardsigmoid_out", "B"],
        outputs=["output"],
        name="mul_node",
    )

    graph = helper.make_graph(
        [hardsigmoid, mul],
        name="hardsigmoid_mul_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[B],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def test_lower_real_sigmoid_to_streamingsigmoid():
    model = _build_sigmoid_model()

    model, changed = LowerToNN2FPGALayers().apply(model)

    assert changed is False
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "StreamingSigmoid"
    assert node.domain == "nn2fpga.compiler.custom_op"
    assert getCustomOp(node).get_nodeattr("impl") == "sigmoid"


def test_lower_hardsigmoid_mul_to_streamingsigmoid_legacy_impl():
    alpha = 0.1666666716337204
    b_value = 1.0001220703125
    model = _build_hardsigmoid_mul_model(alpha=alpha, b_value=b_value)

    model, changed = LowerToNN2FPGALayers().apply(model)

    assert changed is False
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    op = getCustomOp(node)
    assert node.op_type == "StreamingSigmoid"
    assert node.domain == "nn2fpga.compiler.custom_op"
    assert op.get_nodeattr("impl") == "hardsigmoid_mul"
    assert op.get_nodeattr("alpha") == alpha
    assert op.get_nodeattr("B") == b_value
