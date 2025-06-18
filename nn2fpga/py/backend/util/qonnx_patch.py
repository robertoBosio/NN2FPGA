# utils/qonnx_patch.py
from qonnx.util import basic as qonnx_basic
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import (
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
)

# Store original in case needed
original_is_finn_op = qonnx_basic.is_finn_op

# List of extra custom op domains to treat as QONNX-compatible
EXTRA_CUSTOM_DOMAINS = ["backend"]


def patched_is_finn_op(op_type):
    if original_is_finn_op(op_type):
        return True
    return any(op_type.startswith(dom) for dom in EXTRA_CUSTOM_DOMAINS)


def patch_qonnx_ops():
    """
    Monkey patch QONNX to treat custom domains as if they are QONNX ops.
    Call this once at the start of your program.
    """
    qonnx_basic.is_finn_op = patched_is_finn_op
    patch_qonnx_datatype()


def patch_qonnx_datatype():
    """
    Monkey patch QONNX's ModelWrapper to use custom tensor datatype based on TensorQuant.
    """
    ModelWrapper.set_tensor_datatype = set_custom_tensor_datatype
    ModelWrapper.get_tensor_datatype = get_custom_tensor_datatype
