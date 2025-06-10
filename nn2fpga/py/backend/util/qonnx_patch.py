# utils/qonnx_patch.py
from qonnx.util import basic as qonnx_basic

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
