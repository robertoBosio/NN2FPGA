from backend.utils import *
from backend.main import parse_all_main
from backend.layers.uram_download import *
import numpy as np

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

def init(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_sim.h" % file_name, "w+") as fd:

        libraries = [
            "%s.h" % file_name,
            "nn2fpga/debug.h"
        ]

        fd.write("#ifndef __NETWORKSIM__\n")
        fd.write("#define __NETWORKSIM__\n")

        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        fd.write("\n")

        fd.write("void networkSim(\n")

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\thls::stream<t_%s> &%s,\n" % (name, name))

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\thls::stream<t_o_%s> &o_%s\n" % (name, name))

        fd.write(") {\n")


def declare_uram_layer(parsed_write):
    
    concat_weights = None
    remove_tb_declare = []
    for i, layer in enumerate(parsed_write):

        if 'uram_input' in layer.keys():
            if len(layer["uram_input"]) > 0:
                if concat_weights is None:
                    concat_weights = layer["tb_declare"][0]["init"]
                else:
                    concat_weights = np.concatenate(
                        [
                            concat_weights,
                            layer["tb_declare"][0]["init"]
                        ]
                    )
                remove_tb_declare.append(i)

    for i in remove_tb_declare:
       parsed_write[i]["tb_declare"] = []

    uram_declare = None
    if concat_weights is not None:
        output_name = "weights"
        uram_declare = {}
        uram_declare["name"] = "c_%s" % output_name
        uram_declare["type"] = "t_%s_st" % output_name
        uram_declare["is_array"] = True
        uram_declare["is_const"] = True
        uram_declare["size"] = concat_weights.shape
        uram_declare["init"] = concat_weights
    
    dim = None
    if concat_weights is not None:
        dim = concat_weights.shape[0]

    return parsed_write, [uram_declare], dim, concat_weights


def body(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_sim.h" % file_name, "a") as fd:

        parsed_write, uram_declare, dim, concat_weights = declare_uram_layer(parsed_write)

        if uram_declare[0] is not None:
            tb_declare(fd, uram_declare)

        for layer in parsed_write:

            if 'tb_declare' in layer.keys():
                if len(layer["tb_declare"]) > 0:
                    tb_declare(fd, layer["tb_declare"])

        # Defining DMA blocks which provide input streams 
        for layer in parsed_write:
            if "memory_management" == layer["func"]:
                for name in layer["stream_input"]:
                    dma_layer = dma_func(name, dim)
                    write_defines(fd, dma_layer["defines"])
                    write_declare(fd, dma_layer["declare"][0])
                    write_func(fd, dma_layer)

        fd.write("\t%s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\t\t%s,\n" % (name))

        for layer in parsed_write:
            if "memory_management" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\t\tc_%s,\n" % (name))

                for name in layer["stream_input"]:
                    fd.write("\t\tc_%s_stream[0],\n" % (name))

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\t\to_%s\n" % (name))

        fd.write("\t);\n")
    
    if concat_weights is not None:
        os.system("mkdir -p " + prj_root + "/npy/")
        np.save(prj_root + "/npy/uram_%s.npy" % file_name, concat_weights)


def footer(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_sim.h" % file_name, "a") as fd:
        fd.write("}\n")
        fd.write("\n")
        fd.write("#endif")

def write(io_dict, file_name, prj_root="/tmp"):

    parsed_write, parsed_const = parse_all_main(io_dict)
    parsed_write = parsed_write + parsed_const

    init(file_name, parsed_write, prj_root=prj_root)
    body(file_name, parsed_write, prj_root=prj_root)
    footer(file_name, parsed_write, prj_root=prj_root)

