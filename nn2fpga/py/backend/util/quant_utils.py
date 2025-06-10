from onnx import numpy_helper
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper

def get_quant_params(node: NodeProto, model: ModelWrapper) -> dict:
    """ Get quantization parameters from a quantization node. """
    init_dict = {init.name: init for init in model.model.graph.initializer}
    def get_scalar(name):
        arr = numpy_helper.to_array(init_dict[name])
        return arr.item() if arr.size == 1 else None

    # Handle scale, zeropt, bitwidth
    scale = zeropt = bitwidth = None

    if len(node.input) > 1 and node.input[1] in init_dict:
        scale = get_scalar(node.input[1])
    if len(node.input) > 2 and node.input[2] in init_dict:
        zeropt = get_scalar(node.input[2])
    if len(node.input) > 3 and node.input[3] in init_dict:
        bitwidth = get_scalar(node.input[3])
    attr_dict = {a.name: a.i for a in node.attribute}
    signed = attr_dict.get("signed", 0)
    narrow = attr_dict.get("narrow", 0)
    return dict(scale=scale, zeropt=zeropt, bitwidth=bitwidth,
                signed=signed, narrow=narrow)