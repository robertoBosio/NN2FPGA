from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.transformation.supported_partition import FPGA_SUPPORTED_OPS
from onnx import helper
from backend.custom_op.streamingconv import StreamingConv
from backend.custom_op.streamingglobalaveragepool import StreamingGlobalAveragePool
from backend.custom_op.streamingadd import StreamingAdd
from backend.custom_op.streamingrelu import StreamingRelu

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from onnx import helper, TensorProto
from onnxscript.rewriter import pattern, rewrite
from backend.custom_op.register_rewrite_rule import collect_rules
from onnxscript import ir
import logging
logger = logging.getLogger(__name__)


class LowerToNN2FPGALayers(Transformation):
    """Lower the model to NN2FPGA layers."""

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """
        Apply the transformation to lower the model to NN2FPGA layers.

        Args:
            model (ModelWrapper): The model to transform.

        Returns:
            tuple: A tuple containing the transformed model and a boolean indicating if the transformation needs to be reapplied.
        """
        model = ir.from_proto(model.model)
        model = rewrite(model, pattern_rewrite_rules=collect_rules())
        model = ir.to_proto(model)
        model = ModelWrapper(model)

        return (model, False)
