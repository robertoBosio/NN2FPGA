import os
import sys
import json
import onnx
from onnx import TypeProto, TensorShapeProto

def extract_attrs(node, names):
    return {attr.name: attr for attr in node.attribute if attr.name in names}

def get_shape_from_value_info(value_info: onnx.TypeProto) -> list[int]:
    if not value_info.HasField("tensor_type"):
        raise TypeError("TypeProto does not contain tensor_type")
    return get_shape_from_type(value_info.tensor_type)

def get_shape_from_type(tensor_type: TypeProto.Tensor) -> list[int]:
    """
    Extracts the shape from an ONNX TensorTypeProto object.

    Args:
        tensor_type: An ONNX TypeProto.Tensor object.

    Returns:
        A list of integers representing the shape of the tensor.
    """
    if not isinstance(tensor_type, onnx.TypeProto.Tensor):
        raise TypeError("Expected a TypeProto.Tensor")

    shape = tensor_type.shape
    if shape is None or not isinstance(shape, TensorShapeProto):
        raise ValueError("Invalid or missing shape")

    dims = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(-1)  # symbolic dimension (can choose any placeholder)
        else:
            dims.append(-1)  # unknown dimension
    
    if len(dims) < 4:
        # Pad the shape to 4 dimensions if necessary
        dims = [1] * (4 - len(dims)) + dims

    return dims

def sanitize_string(string: str) -> str:
    """ Sanitizes a string by replacing dots with underscores.
    
    Args:
        string (str): The input string to sanitize.
    
    Returns:
        str: The sanitized string with dots replaced by underscores.
    """
    return string.replace(".", "_")

def extract_board_info(board, prj_root):
    """ Read the board json file and returns a dictionary with the available resources"""
    
    # Remove the part before NN2FPGA from the path
    file_path = prj_root.split("NN2FPGA")[0]
    file_path = f"{file_path}/NN2FPGA/nn2fpga/boards/{board}.json"

    # Opening JSON file with board resources
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block 
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}
    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    
    return board_res

def write_func(fd, info):

    if info["func"] != "memory_management":
        fd.write("\tnn2fpga::%s" % info["func"])
    else:
        fd.write("\t%s" % info["func"])
    if "template" in info.keys():
        fd.write(" <\n")
        for i, template in enumerate(info["template"]):
            if isinstance(template, dict):
                fd.write(f"\t\t{template['name']}")
            else:
                fd.write("\t\t%s" % template)
            if i < len(info["template"])-1:
                fd.write(",")
            else:
                fd.write(">")
            if isinstance(template, dict):
                fd.write(f"\t\t//{template['comment']}")
            fd.write("\n")
    fd.write("\t(\n")
    for i, arg in enumerate(info["args"]):
        fd.write("\t\t%s" %arg)
        if i < len(info["args"])-1:
            fd.write(",\n")
        else:
            fd.write("\n")
            fd.write("\t);\n")
    fd.write("\n")


def body(file_path, parsed_write, prj_root="/tmp"):

    with open(file_path, "a") as fd:
        
        for layer in parsed_write:

            write_func(fd, layer)

        fd.write("}\n")

def write_const(fd, values, i, dims, form="int"):

    if (i == (dims)):
        if form == "float":
            fd.write("%0.16f" % values)
        else:
            fd.write("%0d" % values)
    else:
        fd.write("{")
        for j in range(values.shape[0]):
            if j > 0:
                fd.write(",")
            write_const(fd, values[j, ...], i+1, dims, form)
        fd.write("}")
    
def write_declare(fd, variable):
    if ("is_pointer" not in variable.keys()):
        variable["is_pointer"] = False

    name = variable["name"]
    is_not_stream = 'is_const' in variable.keys()
    if "is_stream" in variable.keys():
        is_not_stream = not variable["is_stream"]
    
    fd.write("\t")
    type_name = variable["type"]
    if not is_not_stream:
        dim = variable["dim"]
        type_name = "hls::stream<%s>" % type_name
    else:
        if variable["is_const"]:
            fd.write("const ")

    if (variable["is_array"] and not is_not_stream):
        fd.write("%s %s[%0d]" % (type_name, name, dim))
    elif (variable["is_pointer"] and is_not_stream):
        fd.write("%s *%s" % (type_name, name))
    else:
        fd.write("%s %s" % (type_name, name))

    if (is_not_stream):
        if variable["is_array"]:
            for dim in variable["size"]:
                fd.write("[%0d]" % dim)

        if ("attribute" in variable.keys()):
            fd.write(" __attribute__((%s))" % variable["attribute"])
        
        if (variable["is_const"]):
            fd.write(" = ")
            form = "int"

            if "form" in variable.keys():
                form = variable["form"]

            if variable["is_array"]:
                write_const(fd, variable["init"], 0, len(variable["init"].shape), form)
            else:
                if form == "float":
                    fd.write("%0.16f" % variable["init"])
                elif form == "int":
                    fd.write("%0d" % variable["init"])
                else:
                    fd.write("%s" % variable["init"])

        # For non const variables with a starting value. 
        if ("init_value" in variable.keys()):
            fd.write(" = %s" % variable["init_value"])


    fd.write(";\n")

def write_pragma(fd, pragma):
    
    name = pragma["name"]
    options = pragma["options"]
    fd.write(
        "\t#pragma HLS %s" % (name)
    )

    for option in options:
        if len(option) == 1:
            fd.write(
                " %s" % (option[0])
            )
        else:
            fd.write(
                " %s=%s" % (option[0], option[1])
            )

    fd.write("\n")

def declare(file_path, parsed_write, ap_ctrl=None, inline=False, prj_root="/tmp"):
    
    with open(file_path, "a") as fd:
        
        if inline:
            # Adding inline option to put the module in dataflow
            pragma = {}
            pragma["name"] = "inline"
            options = []
            pragma["options"] = options
            write_pragma(fd, pragma)

        if ap_ctrl is not None:
            # Adding return port ap_ctrl definition
            pragma = {}
            pragma["name"] = "interface"
            options = [
                ["mode", ap_ctrl],
                ["port", "return"],
            ]
            pragma["options"] = options
            write_pragma(fd, pragma)

        for layer in parsed_write:

            for variable in layer["declare"]:
                write_declare(fd, variable)

            for pragma in layer["pragma"]:
                write_pragma(fd, pragma)

        fd.write("\n")

def write_defines(fd, values, layer_name):

    fd.write(f"\n/************************* {layer_name} *************************/\n")
    for name, value in values.items():

        if value[0] == 'const':
            fd.write(f"const int {name} = {value[1]};\n")

        if value[0] == 'const_float':
            fd.write(
                "const ap_fixed<32,16> %s = %.16f;\n" % (
                    name,
                    value[1]
                )
            )

        if value[0] == 'type':
            fd.write(
                "typedef %s %s;\n" % (
                    value[1],
                    name,
                )
            )

        if value[0] == 'struct':
            fd.write("typedef struct {\n")
            for field in value[1]:
                fd.write("\t%s %s;\n" % (field[1], field[0]))
            fd.write("} %s;\n" % name)

        if value[0] == 'alias':
            fd.write(
                "using %s=%s;\n" % (
                    name,
                    value[1],
                )
            )
    
