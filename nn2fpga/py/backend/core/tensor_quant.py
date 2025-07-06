import re
from qonnx.util import basic as qonnx_basic
from onnx import TensorAnnotation, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper

class TensorQuant:
    def __init__(self, bitwidth, signed, scale, zeropt, narrow_range=False, rounding="ROUND"):
        self.bitwidth = int(bitwidth)
        self.signed = int(signed)
        self.scale = float(scale)
        self.zeropt = int(zeropt)
        self.narrow_range = int(narrow_range)
        self.rounding = str(rounding)

    def get_canonical_name(self):
        return f"Q[{self.bitwidth},{self.signed},{self.scale},{self.zeropt},{self.narrow_range},{self.rounding}]"

    @staticmethod
    def from_canonical_name(s):
        m = re.fullmatch(
            r"Q\[(\d+),(0|1),([0-9.eE+-]+),(\d+),(0|1),(\w+)\]",
            s
        )
        if not m:
            raise ValueError(f"Invalid quantization annotation string: {s}")
        return TensorQuant(
            bitwidth=int(m.group(1)),
            signed=int(m.group(2)),
            scale=float(m.group(3)),
            zeropt=int(m.group(4)),
            narrow_range=int(m.group(5)),
            rounding=str(m.group(6))
        )

    def __repr__(self):
        return f"<TensorQuant {self.get_canonical_name()}>"

def set_custom_tensor_datatype(model: ModelWrapper, tensor_name: str, tensor_quant: TensorQuant):
    """Sets the TensorQuant of a tensor with the given name."""
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")

    if ret is not None:
        ret_dt = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "tensor_quant", "key")
        if ret_dt is not None:
            if tensor_quant is None:
                ret_dt.Clear()
            else:
                ret_dt.value = tensor_quant.get_canonical_name()
        elif tensor_quant is not None:
            dt = StringStringEntryProto()
            dt.key = "tensor_quant"
            dt.value = tensor_quant.get_canonical_name()
            ret.quant_parameter_tensor_names.append(dt)
    elif tensor_quant is not None:
        qa = TensorAnnotation()
        qa.tensor_name = tensor_name
        dt = StringStringEntryProto()
        dt.key = "tensor_quant"
        dt.value = tensor_quant.get_canonical_name()
        qa.quant_parameter_tensor_names.append(dt)
        qnt_annotations.append(qa)

def get_custom_tensor_datatype(model: ModelWrapper, tensor_name):
    """Gets the custom TensorQuant of a tensor with the given name.
    Returns None if not found.
    """
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")
    if ret is None:
        return None

    ret_dt = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "tensor_quant", "key")
    if ret_dt is None:
        return None

    try:
        return TensorQuant.from_canonical_name(ret_dt.value)
    except Exception as e:
        raise ValueError(f"Invalid TensorQuant string for tensor {tensor_name}: {ret_dt.value}") from e
