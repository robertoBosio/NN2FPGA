import os
import sys

def write_func(fd, info):

    if info["func"] != "memory_management":
        fd.write("\tnn2fpga::%s" % info["func"])
    else:
        fd.write("\t%s" % info["func"])
    if "template" in info.keys():
        fd.write(" <\n")
        for i, template in enumerate(info["template"]):
            fd.write("\t\t%s" % template)
            if i < len(info["template"])-1:
                fd.write(",\n")
            else:
                fd.write("\n")
                fd.write("\t>")
    fd.write(" (\n")
    for i, arg in enumerate(info["args"]):
        fd.write("\t\t%s" %arg)
        if i < len(info["args"])-1:
            fd.write(",\n")
        else:
            fd.write("\n")
            fd.write("\t);\n")
    fd.write("\n")


def body(file_name, parsed_write, prj_root="/tmp"):

    with open(prj_root + ("/cc/src/%s.cc" % file_name), "a") as fd:
        
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
    else:
        fd.write("%s %s" % (type_name, name))

    if (is_not_stream):
        if variable["is_array"]:
            for dim in variable["size"]:
                fd.write("[%0d]" % dim)

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
                else:
                    fd.write("%0d" % variable["init"])

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

def declare(file_name, parsed_write, ap_ctrl=None, inline=False, prj_root="/tmp"):

    with open(prj_root + ("/cc/src/%s.cc" % file_name), "a") as fd:
        
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

def write_defines(fd, values):

    for name, value in values.items():

        if value[0] == 'const':
            fd.write(
                "const int %s = %0d;\n" % (
                    name,
                    value[1]
                )
            )

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

def defines(file_name, parsed_write, prj_root="/tmp"):

    libraries = [
        "ap_int.h",
        "hls_stream.h",
        "hls_vector.h",
        "stdint.h",
        "ap_axi_sdata.h",
    ]

    with open(prj_root + ("/cc/include/%s.h" % file_name), "w+") as fd:
        
        fd.write("#ifndef NN2FPGA_NETWORK_H_\n")
        fd.write("#define NN2FPGA_NETWORK_H_\n")

        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        for layer in parsed_write:

            if 'defines' in layer.keys():
                write_defines(fd, layer["defines"])

        fd.write("\n")

        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &%s,\n" % (name, name))

        for layer in parsed_write:
            if "memory_management" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\tconst t_%s_st *i_data_%s,\n" % (name, name))

                for name in layer["stream_input"]:
                    fd.write("\thls::stream<t_%s_stream> &i_data_%s,\n" % (name, name))

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(");\n")

        fd.write("void memory_management(\n")

        for layer in parsed_write:
            if 'is_const' in layer.keys():
                for name in layer["input"]:
                    fd.write("\tconst t_%s_st *i_data_%s,\n" % (name, name))

                for name in layer["stream_input"]:
                    fd.write("\thls::stream<t_%s_stream> &i_data_%s,\n" % (name, name))

        for i, layer in enumerate(parsed_write):
            if 'is_const' in layer.keys():
                for j, name in enumerate(layer["output"]):
                    fd.write(
                        "\thls::stream<t_%s> s_%s[%0d]" % (
                            name,
                            name,
                            layer["size"]
                        )
                    )
                    if i < (len(parsed_write)-1) or j < (len(layer["output"])-1):
                        fd.write(",")
                    fd.write("\n")

        fd.write(");\n")

        fd.write("\n")
        fd.write("#endif")
