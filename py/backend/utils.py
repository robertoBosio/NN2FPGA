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

    if (variable["is_array"] and not is_const):
        fd.write("\t%s %s[%0d]" % (type_name, name, dim))
    else:
        fd.write("\t%s %s" % (type_name, name))

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

def declare(file_name, parsed_write):

    with open("src/%s.cpp" % file_name, "a") as fd:
        
        for layer in parsed_write:

            for variable in layer["declare"]:
                write_declare(fd, variable)

            for pragma in layer["pragma"]:
                write_pragma(fd, pragma)

        fd.write("\n")

