from onnx import numpy_helper, helper
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import remove_by_name


def get_quant_params(node: NodeProto, model: ModelWrapper) -> dict:
    """Get quantization parameters from a quantization node."""
    init_dict = {init.name: init for init in model.model.graph.initializer}

    def get_value(name):
        return numpy_helper.to_array(init_dict[name])

    # Handle scale, zeropt, bitwidth
    scale = zeropt = bitwidth = None

    if len(node.input) > 1 and node.input[1] in init_dict:
        scale = get_value(node.input[1])
    if len(node.input) > 2 and node.input[2] in init_dict:
        zeropt = get_value(node.input[2])
    if len(node.input) > 3 and node.input[3] in init_dict:
        bitwidth = get_value(node.input[3])
    attr_dict = {a.name: a.i for a in node.attribute}
    signed = attr_dict.get("signed", None)
    narrow = attr_dict.get("narrow", None)
    return dict(
        scale=scale, zeropt=zeropt, bitwidth=bitwidth, signed=signed, narrow=narrow
    )

def get_quant_attributes(node: NodeProto, direction: str) -> dict:
    """Get quantization attributes from a node with folded quantization."""
    attr_dict = {a.name: a for a in node.attribute}

    quant_attributes = {
        "scale": attr_dict.get(f"scale_{direction}", None),
        "zeropt": attr_dict.get(f"zeropt_{direction}", None),
        "bitwidth": attr_dict.get(f"bitwidth_{direction}", None),
        "signed": attr_dict.get(f"signed_{direction}", None),
        "narrow": attr_dict.get(f"narrow_{direction}", None),
    }

    quant_attributes = {
        "scale": quant_attributes["scale"].f if quant_attributes["scale"] else None,
        "zeropt": int(quant_attributes["zeropt"].i) if quant_attributes["zeropt"] else None,
        "bitwidth": int(quant_attributes["bitwidth"].i) if quant_attributes["bitwidth"] else None,
        "signed": int(quant_attributes["signed"].i) if quant_attributes["signed"] else None,
        "narrow": int(quant_attributes["narrow"].i) if quant_attributes["narrow"] else None,
    }

    return quant_attributes

def add_attribute(
    node: NodeProto, name: str, attr_type: helper.AttributeProto.AttributeType, value
) -> None:
    """Add an attribute to a node."""

    node.attribute.append(
        helper.make_attribute(key=name, value=value, attr_type=attr_type)
    )

def set_attribute(
    node: NodeProto, name: str, attr_type: helper.AttributeProto.AttributeType, value) -> None:
    """Set an attribute on a node, replacing any existing attribute with the same name."""

    remove_by_name(node.attribute, name)
    add_attribute(node, name, attr_type, value)


def set_quant_attributes(node: NodeProto, direction: str, quant_params: dict) -> None:
    """Set quantization attributes on a node."""
    add_attribute(
        node,
        f"scale_{direction}",
        helper.AttributeProto.FLOAT,
        float(quant_params["scale"]),
    )
    add_attribute(
        node,
        f"zeropt_{direction}",
        helper.AttributeProto.INT,
        int(quant_params["zeropt"]),
    )
    add_attribute(
        node,
        f"bitwidth_{direction}",
        helper.AttributeProto.INT,
        int(quant_params["bitwidth"]),
    )
    add_attribute(
        node,
        f"signed_{direction}",
        helper.AttributeProto.INT,
        int(quant_params["signed"]),
    )
    add_attribute(
        node,
        f"narrow_{direction}",
        helper.AttributeProto.INT,
        int(quant_params["narrow"]),
    )


def compare_quant_attributes(
    quant1_dict: dict,
    quant2_dict: dict,
) -> bool:
    """Compare two quantization attribute dictionaries."""
    for key in quant1_dict:
        if key not in quant2_dict:
            return False
        if quant1_dict[key] != quant2_dict[key]:
            return False
    return True


def is_constant_input_node(model: ModelWrapper, node: NodeProto) -> bool:
    """Check if the node has only constant inputs.
    It is used to distinguish between Quant nodes on the activation and
    Quant nodes on the parameters (weights and biases).
    """
    init_names = {init.name for init in model.graph.initializer}
    return all(inp in init_names for inp in node.input)
