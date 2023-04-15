import os
import sys

def write_func(fd, info):

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


def body(file_name, parsed_write):

    with open("src/%s.cpp" % file_name, "a") as fd:
        
        for layer in parsed_write:

            write_func(fd, layer)

        fd.write("}\n")

def write_const(fd, values, i, dims):

    if (i == (dims)):
        fd.write("%0d" % values)
    else:
        fd.write("{")
        for j in range(values.shape[0]):
            if j > 0:
                fd.write(",")
            write_const(fd, values[j, ...], i+1, dims)
        fd.write("}")
    
def write_declare(fd, variable):
    
    name = variable["name"]
    is_const = 'init' in variable.keys() 
    
    type_name = variable["type"]
    if not is_const:
        dim = variable["dim"]
        type_name = "hls::stream<%s>" % type_name
    else:
        fd.write("\tconst ")

    if (variable["is_array"] and not is_const):
        fd.write("\t%s %s[%0d]" % (type_name, name, dim))
    elif not is_const:
        fd.write("\t%s %s" % (type_name, name))
    else:
        fd.write("%s_st %s" % (type_name, name))

    if (is_const):
        for dim in variable["init"].shape:
            fd.write("[%0d]" % dim)
        fd.write(" = ")
        write_const(fd, variable["init"], 0, len(variable["init"].shape))

    fd.write(";\n")

def write_pragma(fd, pragma):
    
    name = pragma["name"]
    options = pragma["options"]
    fd.write(
        "\t#pragma HLS %s" % (name)
    )

    for option in options:
        fd.write(
            " %s=%s" % (option[0], option[1])
        )

    fd.write("\n")

def declare(file_name, parsed_write, inline=False):

    with open("src/%s.cpp" % file_name, "a") as fd:
        
        for layer in parsed_write:

            for variable in layer["declare"]:
                write_declare(fd, variable)

            if inline:
                # Adding inline option to put the module in dataflow
                pragma = {}
                pragma["name"] = "inline"
                options = []
                pragma["options"] = options
                write_pragma(fd, pragma)

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

def defines(file_name, parsed_write):

    libraries = [
        "ap_int.h",
        "hls_stream.h",
        "hls_vector.h",
        "stdint.h",
        "ap_axi_sdata.h",
    ]

    with open("src/%s.hpp" % file_name, "w+") as fd:
        
        fd.write("#ifndef __NETWORK__\n")
        fd.write("#define __NETWORK__\n")

        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        for layer in parsed_write:

            if 'defines' in layer.keys():
                write_defines(fd, layer["defines"])

        fd.write("\n")

        fd.write("void %s(\n" % file_name)

        for layer in parsed_write:
            if "ProduceStream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &%s,\n" % (name, name))

        for layer in parsed_write:
            if "MemoryManagement" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\tap_int<READ_WIDTH> *i_data_%s,\n" % name)

        for layer in parsed_write:
            if "ConsumeStream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(");\n")

        fd.write("void MemoryManagement(\n")

        for layer in parsed_write:
            if 'is_const' in layer.keys():
                for name in layer["input"]:
                    fd.write("\tap_int<READ_WIDTH> *i_data_%s,\n" % (name))

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
