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


def init(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_sim.h" % file_name, "w+") as fd:

        uram_storage = False
        if "URAM_STORAGE" in os.environ:
            if int(os.environ.get("URAM_STORAGE")) == 1:
                uram_storage = True
        
        vitis_flow = False
        if "VITIS_FLOW" in os.environ:
            if int(os.environ.get("VITIS_FLOW")) == 1:
                vitis_flow = True
        
        libraries = [
            "#include \"params.h\"",
            "#include \"nn2fpga/debug.h\"",
            "#include <chrono>",
            "#include <iostream>",
            "#include <fstream>",
            f"#include \"{file_name}.h\""
        ]

        libraries.append("#include \"nn2fpga/mm2s.h\"")
        libraries.append("#include \"nn2fpga/s2mm.h\"")
        if (vitis_flow):
            libraries.append("\n#ifndef CSIM")
            libraries.append("#include \"experimental/xrt_bo.h\"")
            libraries.append("#include \"experimental/xrt_device.h\"")
            libraries.append("#include \"experimental/xrt_kernel.h\"")
            libraries.append("#include \"cmdlineparser.h\"")
            libraries.append("#endif /* CSIM */\n")

        fd.write("#ifndef __NETWORK_SIM__\n")
        fd.write("#define __NETWORK_SIM__\n")

        for lib in libraries:
            fd.write(f"{lib}\n")

        fd.write("\n")
        fd.write("std::chrono::duration<double> networkSim(\n")

        input_args = []
        # Must be passed from testbench to retrieve .xclbin file
        input_args.append("\tint argc")
        input_args.append("\tchar** argv")
        input_args.append("\tstd::string prj_root")

        # TODO: insert support for more than one array of input and output
        input_args.append("\tconst unsigned int n_inp")
        input_args.append("\tconst unsigned int n_out")

        for layer in parsed_write:
            if "produce_stream" == layer["func"]:
                for name in layer["input"]:
                    input_args.append("\tconst t_in_mem* %s" % name)

        for layer in parsed_write:
            if "consume_stream" == layer["func"]:
                for name in layer["output"]:
                    input_args.append("\tt_out_mem* o_%s" % name)

        for i, args in enumerate(input_args):
            if i == len(input_args) - 1:
                fd.write(f"{args}\n")
            else:
                fd.write(f"{args},\n")

        fd.write(") {\n")

def declare_uram_layer(parsed_write):

    # Concatenating all the weights of the layers in a single array to be stored
    # in a file 
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

    # Removing weights declaration from the dictionary.
    for i in remove_tb_declare:
       parsed_write[i]["tb_declare"] = []
    
    # Create a single variable delcaration for the concatenation of the weights
    uram_declare = None
    if concat_weights is not None:
        output_name = "weights"
        uram_declare = {}
        uram_declare["name"] = "c_%s" % output_name
        uram_declare["type"] = "t_%s_st" % output_name
        uram_declare["is_array"] = False
        uram_declare["is_pointer"] = True
        uram_declare["is_const"] = False

        # uram_declare["is_array"] = True
        # uram_declare["is_const"] = True
        # uram_declare["size"] = concat_weights.shape
        # uram_declare["init"] = concat_weights
        # uram_declare["attribute"] = "aligned(4096)"
        uram_declare["defines"] = {}
        uram_declare["defines"]["c_%s_dim" % (output_name)] = ["const", concat_weights.shape[0]]
    
    dim = None
    if concat_weights is not None:
        dim = concat_weights.shape[0]

    return parsed_write, [uram_declare], dim, concat_weights


def body(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_sim.h" % file_name, "a") as fd:

        parsed_write, uram_declare, uram_dim, concat_weights = declare_uram_layer(parsed_write)
        if uram_declare[0] is not None:
            print("\t", end="", file=fd)
            write_defines(fd, uram_declare[0]["defines"])
            tb_declare(fd, uram_declare)
            fd.write(f"\tposix_memalign((void**)&{uram_declare[0]['name']}, 4096, {uram_dim} * sizeof({uram_declare[0]['type']}));\n")
            fd.write(f"\tstd::ifstream file_weights(prj_root + \"npy/{file_name}_weights.bin\", std::ios::binary);\n")
            fd.write(f"\tfile_weights.read(reinterpret_cast<char*>({uram_declare[0]['name']}), {uram_dim} * sizeof({uram_declare[0]['type']}));\n")
            fd.write(f"\tfile_weights.close();\n\n")

        for layer in parsed_write:
            if 'tb_declare' in layer.keys():
                if len(layer["tb_declare"]) > 0:
                    tb_declare(fd, layer["tb_declare"])
        
        # Alveo boards must follow the vitis flow, which 
        vitis_flow = False
        if "VITIS_FLOW" in os.environ:
            if int(os.environ.get("VITIS_FLOW")) == 1:
                vitis_flow = True

        # List of connections for the Vitis flow
        connectivity = []
        if (vitis_flow):
            fd.write("#ifdef CSIM\n")

        # In case of embedded FPGA the blocks for stream to memory and memory to
        # stream are located in the testbench
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
                    write_func(fd, mm2s_weights_layer)

                    # Wrap the templated function for the Vitis flow
                    d_function = {}
                    d_function["includes"] = []
                    d_function["includes"].append("nn2fpga/mm2s.h")
                    d_function["includes"].append("params.h")
                    d_function["arguments"] = []
                    d_function["arguments"].append(f"const t_{name}_st* c_{name}")
                    d_function["arguments"].append(f"const unsigned int c_{name}_dim")
                    d_function["arguments"].append(f"hls::stream<t_{name}_stream>& c_{name}_stream")
                    d_function["parameters"] = []
                    d_function["parameters"].append(f"c_{name}")
                    d_function["parameters"].append(f"c_{name}_dim")
                    d_function["parameters"].append(f"c_{name}_stream")
                    d_function["function_name"] = "nn2fpga::mm2s"
                    d_function["template"] = []
                    d_function["template"].append(f"t_{name}_st")
                    d_function["template"].append(f"t_{name}_stream")
                    connectivity.append(f"sp=mm2s_weights_1.c_{name}:DDR[0]")
                    connectivity.append(f"sc=mm2s_weights_1.c_{name}_stream:{file_name}_1.i_data_weights")
                    write_templated_converted("mm2s_weights", d_function, prj_root)
        
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
                    mm2s_activations_layer["args"].append("%s" % name)
                    mm2s_activations_layer["args"].append("n_inp")
                    mm2s_activations_layer["args"].append("c_%s_stream" % name)
                    mm2s_activations_layer["declare"] = []
                    mm2s_activations_layer["defines"] = {}
                    mm2s_activations_layer["pragma"] = []

                    write_func(fd, mm2s_activations_layer)
                    
                    # Wrap the templated function for the Vitis flow
                    d_function = {}
                    d_function["includes"] = []
                    d_function["includes"].append("nn2fpga/mm2s.h")
                    d_function["includes"].append("params.h")
                    d_function["arguments"] = []
                    d_function["arguments"].append(f"const t_in_mem* {name}")
                    d_function["arguments"].append(f"const unsigned int n_inp")
                    d_function["arguments"].append(f"hls::stream<t_{name}>& c_{name}_stream")
                    d_function["parameters"] = []
                    d_function["parameters"].append(f"{name}")
                    d_function["parameters"].append(f"n_inp")
                    d_function["parameters"].append(f"c_{name}_stream")
                    d_function["function_name"] = "nn2fpga::mm2s"
                    d_function["template"] = []
                    d_function["template"].append(f"t_in_mem")
                    d_function["template"].append(f"t_{name}")
                    connectivity.append(f"sp=mm2s_activations_1.{name}:DDR[0]")
                    connectivity.append(f"sc=mm2s_activations_1.c_{name}_stream:{file_name}_1.i_inp_1")
                    write_templated_converted("mm2s_activations", d_function, prj_root)

        fd.write("\tauto start = std::chrono::high_resolution_clock::now();\n\n");
        
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
        fd.write("\tauto end = std::chrono::high_resolution_clock::now();\n\n")
        
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

                    # Wrap the templated function for the Vitis flow
                    d_function = {}
                    d_function["includes"] = []
                    d_function["includes"].append("nn2fpga/s2mm.h")
                    d_function["includes"].append("params.h")
                    d_function["arguments"] = []
                    d_function["arguments"].append(f"t_out_mem* o_{name}")
                    d_function["arguments"].append(f"const unsigned int n_out")
                    d_function["arguments"].append(f"hls::stream<t_o_{name}>& c_{name}_stream")
                    d_function["parameters"] = []
                    d_function["parameters"].append(f"o_{name}")
                    d_function["parameters"].append(f"n_out")
                    d_function["parameters"].append(f"c_{name}_stream")
                    d_function["function_name"] = "nn2fpga::s2mm"
                    d_function["template"] = []
                    d_function["template"].append(f"t_out_mem")
                    d_function["template"].append(f"t_o_{name}")
                    write_templated_converted("s2mm_outputs", d_function, prj_root)
                    connectivity.append(f"sp=s2mm_outputs_1.o_{name}:DDR[0]")
                    connectivity.append(f"sc={file_name}_1.o_outp1:s2mm_outputs_1.c_{name}_stream")
                    write_func(fd, s2mm_layer)
        
        if (vitis_flow):

            # Storing kernels name to start 
            input_kernels = []
            weights_kernels = []
            output_kernels = []

            fd.write("#else\n")

            #Opening the device and the xclbin
            fd.write("\tsda::utils::CmdLineParser parser;\n")
            fd.write("\tparser.addSwitch(\"--xclbin_file\", \"-x\", \"input binary file string\", \"\");\n")
            fd.write("\tparser.addSwitch(\"--device_id\", \"-d\", \"device index\", \"0\");\n")
            fd.write("\tparser.addSwitch(\"--n_images\", \"-n\", \"input number of images\", \"1\");\n")
            fd.write("\tparser.addSwitch(\"--upload_weights\", \"-w\", \"input upload weights flag\", \"1\");\n")
            fd.write("\tparser.parse(argc, argv);\n")
            fd.write("\tstd::string binaryFile = parser.value(\"xclbin_file\");\n")
            fd.write("\tint device_index = stoi(parser.value(\"device_id\"));\n")
            fd.write("\tint upload_weights_flag = stoi(parser.value(\"upload_weights\"));\n")
            fd.write("\tstd::cout << \"Opening the device \" << device_index << std::endl;\n")
            fd.write("\tauto device = xrt::device(device_index);\n")
            fd.write("\tstd::cout << \"Loading the xclbin \" << binaryFile << std::endl;\n")
            fd.write("\tauto uuid = device.load_xclbin(binaryFile);\n")
            fd.write("\n")

            # TODO: Implement support for more than one stream of activation 
            # TODO: Implement support for activations that are not packed in ap_uint<64>
            for layer in parsed_write:
                if "produce_stream" == layer["func"]:
                    if (len(layer["input"]) > 0):
                        fd.write(f"\tauto mm2s_activations = xrt::kernel(device, uuid, \"mm2s_activations\");\n")
                        fd.write("\tauto buff_activations = xrt::bo(device, (int*)inp_1, n_inp * 8, mm2s_activations.group_id(0));\n")
                        fd.write("\tstd::cout << \"Synching h2d inp_1\" << std::endl;\n")
                        fd.write("\tbuff_activations.sync(XCL_BO_SYNC_BO_TO_DEVICE);\n")
                        fd.write("\tauto mm2s_a = xrt::run(mm2s_activations);\n")
                        fd.write("\tmm2s_a.set_arg(0, buff_activations);\n")
                        fd.write("\tmm2s_a.set_arg(1, n_inp);\n\n")
                        input_kernels.append("mm2s_a")

            for layer in parsed_write:
                if "memory_management" == layer["func"]:
                    if (len(layer["stream_input"]) > 0):
                        fd.write(f"\tauto mm2s_weights = xrt::kernel(device, uuid, \"mm2s_weights\");\n")
                        fd.write("\tauto buff_weights = xrt::bo(device, (int*)%s, %s, mm2s_weights.group_id(0));\n" % (uram_declare[0]["name"], uram_dim))
                        fd.write("\tstd::cout << \"Synching h2d %s\" << std::endl;\n" % uram_declare[0]["name"])
                        fd.write("\tbuff_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);\n")
                        fd.write("\tauto mm2s_w = xrt::run(mm2s_weights);\n")
                        fd.write("\tmm2s_w.set_arg(0, buff_weights);\n")
                        fd.write(f"\tmm2s_w.set_arg(1, {uram_declare[0]['name']}_dim);\n\n")
                        weights_kernels.append("mm2s_w")

            for layer in parsed_write:
                if "consume_stream" == layer["func"]:
                    if (len(layer["output"]) > 0):
                        fd.write(f"\tauto s2mm_output = xrt::kernel(device, uuid, \"s2mm_outputs\");\n")
                        fd.write("\tauto buff_output = xrt::bo(device, (int*)o_outp1, n_out, s2mm_output.group_id(0));\n")
                        fd.write("\tauto s2mm_o = xrt::run(s2mm_output);\n")
                        fd.write("\ts2mm_o.set_arg(0, buff_output);\n")
                        fd.write("\ts2mm_o.set_arg(1, n_out);\n\n")
                        output_kernels.append("s2mm_o")

            for ker in output_kernels:
                fd.write(f"\t{ker}.start();\n")
            
            fd.write("\n\tauto start = std::chrono::high_resolution_clock::now();\n\n");

            if (len(weights_kernels) > 0):
                fd.write("\tif (upload_weights_flag) {\n")
                for ker in weights_kernels:
                    fd.write(f"\t\t{ker}.start();\n")
                fd.write("\t}\n")
            
            for ker in input_kernels:
                fd.write(f"\t{ker}.start();\n")

            # Calling the kernel with also the size parameters

            for ker in output_kernels:
                fd.write(f"\t{ker}.wait();\n")
            
            fd.write("\n\tauto end = std::chrono::high_resolution_clock::now();\n\n")

            fd.write("\tstd::cout << \"Synching d2h o_outp1\" << std::endl;\n")
            fd.write("\tbuff_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);\n")
            fd.write("\n")
            
            fd.write("#endif /* CSIM */\n\n")
    
        if uram_declare[0] is not None:
            fd.write(f"\tfree({uram_declare[0]['name']});\n")
        fd.write("\treturn (end - start);\n")
    
    # Write the .npy file containting the wieghts
    if concat_weights is not None:
        os.system(f"mkdir -p {prj_root}/npy/")
        np.save(f"{prj_root}/npy/uram_{file_name}.npy", concat_weights)
        
        concat_weights_uint8 = concat_weights.astype(np.uint8)
        # concat_weights_uint8.tofile(f"{prj_root}/npy/{file_name}_weights.bin")
        with open(f"{prj_root}/npy/{file_name}_weights.bin", "wb") as file:
            file.write(concat_weights_uint8.tobytes())
    
    with open(prj_root + "/%s_link.cfg" % file_name, "w+") as fd:
        fd.write("[connectivity]\n")
        for conn in connectivity:
            fd.write(conn + "\n")

def footer(file_name, parsed_write, prj_root="/tmp"):
    with open(prj_root + "/cc/include/%s_sim.h" % file_name, "a") as fd:
        fd.write("}\n")
        fd.write("\n")
        fd.write("#endif")

# With vitis it is not possible to have templated function as kernel, so we need
# to create a wrapper
# TODO: integrate extern C++ in the write_func
def write_templated_converted(filename, dict, prj_root):
    with open(f"{prj_root}/cc/src/{filename}.cc", "w") as fd:
        for inc in dict["includes"]:
            fd.write(f"#include \"{inc}\"\n")
        fd.write("\nextern \"C++\" {\n")
        fd.write(f"\tvoid {filename}(\n")
        for i, arg in enumerate(dict["arguments"]):
            if (i == len(dict["arguments"]) - 1):
                fd.write(f"\t\t{arg}\n")
            else:
                fd.write(f"\t\t{arg},\n")
        fd.write("\t) {\n\n") 
        fd.write(f"\t\t{dict['function_name']}<")
        for i, temp in enumerate(dict["template"]):
            if (i == len(dict["template"]) - 1):
                fd.write(f"{temp}>\n")
            else:
                fd.write(f"{temp}, ")
        fd.write("\t\t\t(")
        for i, par in enumerate(dict["parameters"]):
            if (i == len(dict["parameters"]) - 1):
                fd.write(f"{par});\n")
            else:
                fd.write(f"{par}, ")
        fd.write("\n\t}\n}")
             


def write(io_dict, file_name, prj_root="/tmp"):

    parsed_write, parsed_const = parse_all_main(io_dict)
    parsed_write = parsed_write + parsed_const

    init(file_name, parsed_write, prj_root=prj_root)
    body(file_name, parsed_write, prj_root=prj_root)
    footer(file_name, parsed_write, prj_root=prj_root)

