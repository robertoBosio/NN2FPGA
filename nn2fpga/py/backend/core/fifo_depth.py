import re
from qonnx.util import basic as qonnx_basic
from onnx import TensorAnnotation, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper

class TensorFifoDepth:
    def __init__(self, depths):
        self.depths = list(map(int, depths))

    def get_canonical_name(self):
        return f"FIFO_DEPTHS[{','.join(str(d) for d in self.depths)}]"

    @staticmethod
    def from_canonical_name(s):
        m = re.fullmatch(r"FIFO_DEPTHS\[(\d+(?:,\d+)*)\]", s)
        if not m:
            raise ValueError(f"Invalid FIFO_DEPTHS string: {s}")
        depths = list(map(int, m.group(1).split(',')))
        return TensorFifoDepth(depths)

    def __repr__(self):
        return f"<TensorFifoDepth {self.get_canonical_name()}>"

def set_custom_tensor_fifo_depth(model: ModelWrapper, tensor_name: str, fifo_depths: TensorFifoDepth):
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")

    if ret is not None:
        ret_fd = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "fifo_depth", "key")
        if ret_fd is not None:
            if fifo_depths is None:
                ret_fd.Clear()
            else:
                ret_fd.value = fifo_depths.get_canonical_name()
        elif fifo_depths is not None:
            fd = StringStringEntryProto()
            fd.key = "fifo_depth"
            fd.value = fifo_depths.get_canonical_name()
            ret.quant_parameter_tensor_names.append(fd)
    elif fifo_depths is not None:
        qa = TensorAnnotation()
        qa.tensor_name = tensor_name
        fd = StringStringEntryProto()
        fd.key = "fifo_depth"
        fd.value = fifo_depths.get_canonical_name()
        qa.quant_parameter_tensor_names.append(fd)
        qnt_annotations.append(qa)

def get_custom_tensor_fifo_depth(model: ModelWrapper, tensor_name: str):
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")
    if ret is None:
        return None

    ret_fd = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "fifo_depth", "key")
    if ret_fd is None:
        return None

    try:
        return TensorFifoDepth.from_canonical_name(ret_fd.value)
    except Exception as e:
        raise ValueError(f"Invalid TensorFifoDepth string for tensor {tensor_name}: {ret_fd.value}") from e
