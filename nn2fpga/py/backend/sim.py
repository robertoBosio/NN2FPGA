from backend.utils import *
from backend.main import parse_all_main

# Packing tensor for 128 bits parallel read
def write_tb_declare(fd, variable, read_width=16, bits=8):
    # ap_int<128> arr[2] = {ap_int<128>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16), 0};

    mask = int(2**bits-1)
    name = variable["name"]
    type_name = variable["type"]

    fd.write("\tconst ")

    fd.write("%s %s" % (type_name, name))

    dim = int(variable["init"].shape[0]/read_width)
    fd.write("[%0d]" % (dim))
    fd.write(" = {")
    for i in range(dim):
        fd.write("%s(\"" % type_name)
        for pack in range(read_width-1, -1, -1):
            index = i*read_width+pack
            value = variable["init"][index]
            value = int(value) & mask
            fd.write(f'{value:x}')
        fd.write("\",")

    fd.write("};")

    values = variable["init"]

    fd.write(";\n")

def tb_declare(fd, layer_tb_declare):

    for variable in layer_tb_declare:
        write_declare(fd, variable)

    fd.write("\n")

def init(file_name, parsed_write, prj_root="."):
    with open(prj_root + "/cc/include/%sSim.hpp" % file_name, "w+") as fd:

        libraries = [
            "Network.hpp"
        ]

        fd.write("#ifndef __NETWORKSIM__\n")
        fd.write("#define __NETWORKSIM__\n")

        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        fd.write("\n")

        fd.write("void %sSim(\n" % file_name)

        for layer in parsed_write:
            if "ProduceStream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &%s,\n" % (name, name))

        for layer in parsed_write:
            if "ConsumeStream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(") {\n")


def body(file_name, parsed_write):
    with open(prj_root + "/cc/include/%sSim.hpp" % file_name, "a") as fd:

        for layer in parsed_write:

            if 'tb_declare' in layer.keys():
                tb_declare(fd, layer["tb_declare"])

        fd.write("\t%s(\n" % file_name)

        for layer in parsed_write:
            if "ProduceStream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\t\t%s,\n" % (name))

        for layer in parsed_write:
            if "MemoryManagement" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\t\tc_%s,\n" % (name))

        for layer in parsed_write:
            if "ConsumeStream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\t\to_%s\n" % (name))

        fd.write("\t);\n")


def footer(file_name, parsed_write):
    with open(prj_root + "/cc/include/%sSim.hpp" % file_name, "a") as fd:
        fd.write("}\n")
        fd.write("\n")
        fd.write("#endif")

def write(io_dict, file_name):

    parsed_write, parsed_const = parse_all_main(io_dict)
    parsed_write = parsed_write + parsed_const

    init(file_name, parsed_write)
    body(file_name, parsed_write)
    footer(file_name, parsed_write)

