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

def init(file_name, parsed_write, ap_ctrl, uram_dim, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_platform.h" % file_name, "w+") as fd:

        vitis_flow = False
        if "VITIS_FLOW" in os.environ:
            if int(os.environ.get("VITIS_FLOW")) == 1:
                vitis_flow = True

        fd.write("#ifndef __NETWORK_PLATFORM_SIM__\n")
        fd.write("#define __NETWORK_PLATFORM_SIM__\n")
        fd.write("#include \"params.h\"\n")

        fd.write("\n")
        fd.write("void %s_platform(\n" % file_name)

        if (vitis_flow == True):
            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    for name in layer["input"]:
                        fd.write("\tconst size_t n_inp,\n")
                        fd.write("\tconst t_in_mem* i_%s,\n" % name)

            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    for name in layer["stream_input"]:
                        fd.write("\tconst size_t c_%s_dim,\n" %  name)
                        fd.write("\tconst t_%s_st* c_%s,\n" %  (name, name))

            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    for name in layer["output"]:
                        fd.write("\tconst size_t n_out,\n")
                        fd.write("\tt_out_mem* o_%s\n" % name)
        else:
            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    for name in layer["input"]:
                        fd.write("\thls::stream<t_%s> &c_%s_stream,\n" % (name, name))

            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    for name in layer["stream_input"]:
                        fd.write("\thls::stream<t_%s_stream> &c_%s_stream,\n" %  (name, name))

            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    for name in layer["output"]:
                        fd.write("\thls::stream<t_o_%s> &c_%s_stream\n" % (name, name))
        fd.write(");\n")
        fd.write("\n")
        fd.write("#endif /* __NETWORK_PLATFORM_SIM__ */")

    with open(prj_root + "/cc/src/%s_platform.cc" % file_name, "w+") as fd:

        vitis_flow = False
        if "VITIS_FLOW" in os.environ:
            if int(os.environ.get("VITIS_FLOW")) == 1:
                vitis_flow = True
        
        libraries = [
            "params.h",
            f"{file_name}.h"
        ]

        if (vitis_flow == True):
            libraries.append("nn2fpga/mm2s.h")
            libraries.append("nn2fpga/s2mm.h")

        for lib in libraries:
            fd.write("#include \"%s\"\n" % lib)

        fd.write("\n")
        fd.write("extern \"C++\" {\n")
        fd.write("\tvoid %s_platform(\n" % file_name)

        pragmas = []
        connectivity = []

        if (vitis_flow == True):
            pragmas.append("#pragma HLS DATAFLOW")
            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    for name in layer["input"]:
                        pragmas.append("#pragma HLS INTERFACE s_axilite port = n_inp")
                        pragmas.append("#pragma HLS INTERFACE m_axi port = i_%s bundle = mem_%s" % (name, name))
                        connectivity.append(f"sp={file_name}_platform_1.i_{name}:DDR[0]")
                        fd.write("\t\tconst size_t n_inp,\n")
                        fd.write("\t\tconst t_in_mem* i_%s,\n" % name)

            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    for name in layer["stream_input"]:
                        pragmas.append("#pragma HLS INTERFACE s_axilite port = c_%s_dim" % name)
                        pragmas.append("#pragma HLS INTERFACE m_axi port = c_%s bundle = mem_%s" % (name, name))
                        connectivity.append(f"sp={file_name}_platform_1.c_{name}:DDR[0]")
                        fd.write("\t\tconst size_t c_%s_dim,\n" %  name)
                        fd.write("\t\tconst t_%s_st* c_%s,\n" %  (name, name))

            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    for name in layer["output"]:
                        pragmas.append("#pragma HLS INTERFACE s_axilite port = n_out")
                        pragmas.append("#pragma HLS INTERFACE m_axi port = o_%s bundle = mem_%s" % (name, name))
                        connectivity.append(f"sp={file_name}_platform_1.o_{name}:DDR[0]")
                        fd.write("\t\tconst size_t n_out,\n")
                        fd.write("\t\tt_out_mem* o_%s\n" % name)
            pragmas.append("#pragma HLS INTERFACE s_axilite port = return")
        else:
            pragmas.append(f"#pragma HLS INTERFACE {ap_ctrl} port = return")
            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    for name in layer["input"]:
                        pragmas.append("#pragma HLS INTERFACE axis port = c_%s_stream" % name)
                        fd.write("\t\thls::stream<t_%s> &c_%s_stream,\n" % (name, name))

            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    for name in layer["stream_input"]:
                        pragmas.append("#pragma HLS INTERFACE axis port = c_%s_stream" % name)
                        fd.write("\t\thls::stream<t_%s_stream> &c_%s_stream,\n" %  (name, name))

            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    for name in layer["output"]:
                        pragmas.append("#pragma HLS INTERFACE axis port = c_%s_stream" % name)
                        fd.write("\t\thls::stream<t_o_%s> &c_%s_stream\n" % (name, name))

        fd.write("\t\t) {\n\n")
        for pragma in pragmas:
            fd.write(pragma + "\n")
        fd.write("\n")

    with open(prj_root + "/%s_link.cfg" % file_name, "w+") as fd:
        fd.write("[connectivity]\n")
        for conn in connectivity:
            fd.write(conn + "\n")
    


def body(file_name, parsed_write, uram_dim, prj_root="/tmp"):
    with open(prj_root + "/cc/src/%s_platform.cc" % file_name, "a") as fd:
        
        
        # Alveo boards must follow the vitis flow, which 
        vitis_flow = False
        if "VITIS_FLOW" in os.environ:
            if int(os.environ.get("VITIS_FLOW")) == 1:
                vitis_flow = True

        # In case of Alveo boards the blocks for stream to memory and memory to
        # stream are located in the kernel
        if (vitis_flow):

            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    for name in layer["stream_input"]:
                        tmp = {}
                        tmp["name"] = "c_%s_stream" % name
                        tmp["type"] = "t_%s_stream" % name
                        tmp["is_array"] = False
                        tmp["dim"] = 0
                        write_declare(fd, tmp)

            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    for name in layer["input"]:
                        tmp = {}
                        tmp["name"] = "c_%s_stream" % name
                        tmp["type"] = "t_%s" % name
                        tmp["is_array"] = False
                        tmp["dim"] = 0
                        write_declare(fd, tmp)

            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    for name in layer["output"]:
                        tmp = {}
                        tmp["name"] = "c_%s_stream" % name
                        tmp["type"] = "t_o_%s" % name
                        tmp["is_array"] = False
                        tmp["dim"] = 0
                        write_declare(fd, tmp)
            fd.write("\n")

            # Defining memory to stream function to load weights
            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    for name in layer["stream_input"]:
                        mm2s_weights_layer = {}
                        mm2s_weights_layer["func"] = "mm2s"
                        mm2s_weights_layer["args"] = []
                        mm2s_weights_layer["input"] = []
                        mm2s_weights_layer["stream_input"] = []
                        mm2s_weights_layer["uram_input"] = []
                        mm2s_weights_layer["output"] = []
                        mm2s_weights_layer["template"] = []
                        mm2s_weights_layer["template"].append("t_%s_st" % (name))
                        mm2s_weights_layer["template"].append("t_%s_stream" % (name))
                        mm2s_weights_layer["args"].append("c_%s" % name)
                        mm2s_weights_layer["args"].append("c_%s_dim" % name)
                        mm2s_weights_layer["args"].append("c_%s_stream" % name)
                        mm2s_weights_layer["declare"] = []
                        mm2s_weights_layer["defines"] = {}
                        mm2s_weights_layer["pragma"] = []

                        print("\t", end="", file=fd)
                        write_func(fd, mm2s_weights_layer)
            
            # Defining memory to stream function to load activations
            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    for name in layer["input"]:
                        mm2s_activations_layer = {}
                        mm2s_activations_layer["func"] = "mm2s"
                        mm2s_activations_layer["args"] = []
                        mm2s_activations_layer["input"] = []
                        mm2s_activations_layer["stream_input"] = []
                        mm2s_activations_layer["uram_input"] = []
                        mm2s_activations_layer["output"] = []
                        mm2s_activations_layer["template"] = []
                        mm2s_activations_layer["template"].append("t_in_mem")
                        mm2s_activations_layer["template"].append("t_%s" % (name))
                        mm2s_activations_layer["args"].append("i_%s" % name)
                        mm2s_activations_layer["args"].append("n_inp")
                        mm2s_activations_layer["args"].append("c_%s_stream" % name)
                        mm2s_activations_layer["declare"] = []
                        mm2s_activations_layer["defines"] = {}
                        mm2s_activations_layer["pragma"] = []

                        write_func(fd, mm2s_activations_layer)

        # Calling the kernel
        fd.write("\t%s(\n" % file_name)

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\t\tc_%s_stream,\n" % (name))

        for layer in parsed_write:
            if "memory_management" == layer["func"]:
                for name in layer["input"]:
                    fd.write("\t\tc_%s,\n" % (name))

                for name in layer["stream_input"]:
                    fd.write("\t\tc_%s_stream,\n" % (name))

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    fd.write("\t\tc_%s_stream\n" % (name))

        fd.write("\t);\n\n")
        
        if (vitis_flow):
            
            # Defining stream to memory function to store results
            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    for name in layer["output"]:
                        s2mm_layer = {}
                        s2mm_layer["func"] = "s2mm"
                        s2mm_layer["args"] = []
                        s2mm_layer["input"] = []
                        s2mm_layer["stream_input"] = []
                        s2mm_layer["uram_input"] = []
                        s2mm_layer["output"] = []
                        s2mm_layer["template"] = []
                        s2mm_layer["template"].append("t_out_mem")
                        s2mm_layer["template"].append("t_o_%s" % (name))
                        s2mm_layer["args"].append("o_%s" % name)
                        s2mm_layer["args"].append("n_out")
                        s2mm_layer["args"].append("c_%s_stream" % name)
                        s2mm_layer["declare"] = []
                        s2mm_layer["declare"].append(tmp)
                        s2mm_layer["defines"] = {}
                        s2mm_layer["pragma"] = []

                        write_func(fd, s2mm_layer)


def footer(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/src/%s_platform.cc" % file_name, "a") as fd:
        fd.write("\t}\n")
        fd.write("}\n")

def write(file_name, parsed_write, ap_ctrl, uram_dim, prj_root="/tmp"):

    init(file_name, parsed_write, ap_ctrl, uram_dim, prj_root=prj_root)
    body(file_name, parsed_write, uram_dim, prj_root=prj_root)
    footer(file_name, parsed_write, prj_root=prj_root)

