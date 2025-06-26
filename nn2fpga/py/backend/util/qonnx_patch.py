from qonnx.util import basic as qonnx_basic
import numpy as np

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
    Monkey patch GenericPartition node execution with nn2fpga simulation flow.
    Call this once at the start of your program.
    """
    # Patch is_finn_op first
    qonnx_basic.is_finn_op = patched_is_finn_op

    # Now safely import GenericPartition and patch execute_node
    from qonnx.custom_op.general.genericpartition import GenericPartition
    from qonnx.core.modelwrapper import ModelWrapper
    from backend.core.simulation import simulate

    original_GenericPartition_get_nodeattr_types = GenericPartition.get_nodeattr_types
    original_GenericPartition_execute_node = GenericPartition.execute_node

    def custom_execute_node(self, context, graph):

        blob = self.get_nodeattr("blob")
        if blob is None:
            return original_GenericPartition_execute_node(self, context, graph)

        context = simulate(blob=blob, context=context)
        return context

    def get_nodeattr_types(self):
        """
        Get the attribute types for the GenericPartition node, including custom attributes.
        The custom attribute "blob" is added for nn2fpga to handle additional data, such as HLS code,
        or input data for the partition.
        This allows to embed everything needed for nn2fpga simulation directly into the ONNX model.
        """
        attr_types = original_GenericPartition_get_nodeattr_types(self)

        # Add custom attributes for nn2fpga
        attr_types.update(
            {
                "blob": ("s", False, ""),
            }
        )

        return attr_types

    GenericPartition.get_nodeattr_types = get_nodeattr_types
    GenericPartition.execute_node = custom_execute_node
