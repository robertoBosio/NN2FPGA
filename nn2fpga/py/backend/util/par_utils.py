from onnx import NodeProto, helper


def check_par_attributes(node: NodeProto) -> bool:
    """Check if a node has parallelization attributes."""

    par_attributes = get_par_attributes(node)
    return any(par_attributes[attr] is not None for attr in par_attributes.keys())


def get_par_attributes(node: NodeProto) -> dict:
    """Get parallelization attributes from a node."""
    attr_dict = {a.name: a for a in node.attribute}

    par_attributes = {
        "ich_par": attr_dict.get(f"ich_par", None),
        "och_par": attr_dict.get(f"och_par", None),
        "w_par": attr_dict.get(f"w_par", None),
    }

    par_attributes = {
        "ich_par": (
            int(par_attributes["ich_par"].i) if par_attributes["ich_par"] else None
        ),
        "och_par": (
            int(par_attributes["och_par"].i) if par_attributes["och_par"] else None
        ),
        "w_par": int(par_attributes["w_par"].i) if par_attributes["w_par"] else None,
    }

    return par_attributes

def set_par_attributes(node: NodeProto, par_attributes: dict) -> None:
    """Set parallelization attributes on a node."""

    for attr_name, attr_value in par_attributes.items():
        node.attribute.append(
            helper.make_attribute(
                attr_name,
                attr_value,
                attr_type=helper.AttributeProto.INT
            )
        )
