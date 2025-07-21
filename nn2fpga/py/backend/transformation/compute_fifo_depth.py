from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from backend.core.fifo_depth import (
    get_custom_tensor_fifo_depth,
    set_custom_tensor_fifo_depth,
    TensorFifoDepth
)
from backend.util.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
import os
import json
import numpy as np
import subprocess

def generate_hls_code(model: ModelWrapper, work_root: str) -> str:
    """ Generate the HLS code to execute the model in fifo-depth mode. """

    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")
    cwr.include("<fstream>")

    # Include files from the nn2FPGA library
    nn2fpga_include_dir = "/workspace/NN2FPGA/nn2fpga/library/include"
    if os.path.isdir(nn2fpga_include_dir):
        for fname in os.listdir(nn2fpga_include_dir):
            if fname.endswith(".hpp"):
                cwr.include(fname)

    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")

    for produce in model.get_nodes_by_op_type("ProduceStream"):
        for stream in getCustomOp(produce).get_input_stream_cpp(model):
            stream.primitive = stream.primitive + "&"
            function.add_argument(stream)

    for const_input_name in {init.name for init in model.graph.initializer if "const_" in init.name}:
        var = cpp_variable(f"{const_input_name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_argument(var)

    for consume in model.get_nodes_by_op_type("ConsumeStream"):
        for stream in getCustomOp(consume).get_output_stream_cpp(model):
            stream.primitive = stream.primitive + "&"
            function.add_argument(stream)

    stream_vars = []
    stream_count = 0
    for node in model.graph.node:

        # Declare the output streams, not for ConsumeStream nodes which are arguments to the top function
        if node.op_type != "ConsumeStream":
            for stream in getCustomOp(node).get_output_stream_cpp(model):
                stream_vars.append(stream)
                stream_count += stream.array[0] 
                function.add_code(f"{stream.generate_declaration()};")

        # Declare the variables used in the node
        for var in getCustomOp(node).get_variable_cpp(model):
            function.add_code(f"{var.generate_declaration()};")

        # Generate the object declaration for the custom operation
        function.add_code(getCustomOp(node).get_object_cpp(model).generate_declaration())
    
    # Declare the array of streams sizes.
    stream_sizes = cpp_variable("stream_max_size", primitive="size_t", value=[2] * stream_count) 
    function.add_code(stream_sizes.generate_initialization())

    # Declare the end flag.    
    end_flag = cpp_variable("end_flag", primitive="bool")
    function.add_code(end_flag.declaration)

    # Declare the clock cycle counter.
    clock_cycle = cpp_variable("clock_cycle", primitive="size_t", value=0)
    function.add_code(clock_cycle.generate_initialization())

    # Write the do while loop to process the data until all the processors are waiting.
    function.add_code("do {")
    function.codewriter.indent()
    function.add_code("end_flag = false;")

    # Execute a step for each node in the model.
    for node in model.graph.node:
        function.add_code(f"end_flag |= {getCustomOp(node).generate_step_call()};")
    
    # Update the fifo max size for each stream.
    iter = 0
    for stream in stream_vars:
        for _ in range(stream.array[0]):
            function.add_code(f"stream_max_size[{iter}] = std::max<size_t>({stream.name}[0].size(), stream_max_size[{iter}]);")
            iter += 1

    function.add_code("clock_cycle++;")
    function.codewriter.dedent()
    function.add_code("} while (end_flag);")

    # Add the final code to save the json report.
    function.add_code(f"std::ofstream report_file(\"{work_root}/fifo_depth.json\");")
    function.add_code("report_file << \"{\\n\";")
    function.add_code("report_file << \"\t\\\"fifo_depth\\\": {\\n\";")

    # Write the fifo depth for each stream.
    iter = 0
    for stream in stream_vars:
        function.add_code(f'report_file << "\t\t\\\"{stream.name}\\\": [" << std::endl;')
        for s in range(stream.array[0]):
            if s != stream.array[0] - 1:
                function.add_code(f'report_file << "\t\t\t" << stream_max_size[{iter}] << ",\\n";')
            else:
                function.add_code(f'report_file << "\t\t\t" << stream_max_size[{iter}] << "\\n";')
            iter += 1
        if iter != stream_count:
            function.add_code("report_file << \"\t\t],\\n\";")
        else:
            function.add_code("report_file << \"\t\t]\\n\";")

    function.add_code("report_file << \"\t},\\n\";")
    function.add_code("report_file << \"\t\\\"latency\\\": \" << clock_cycle << \"\\n\";")
    function.add_code("report_file << \"}\\n\";")
    function.add_code("report_file.close();")
    cwr.add_function_definition(function)
    return cwr.code

def generate_hls_driver(model: ModelWrapper) -> str:
    """Generate HLS driver code for the given model.
    Args:
        model (ModelWrapper): The model to generate HLS driver code for.
    Returns:
        str: The generated HLS driver code as a string.
    """
    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")
    cwr.include("utils/utils.hpp")

    # Accelerator kernel function definition
    kernel_function = cpp_function(
        name=model.get_metadata_prop("top_name"),
        return_type="void",
        qualifiers=["extern"],
    )

    # Add input and output streams to the kernel function
    for produce in model.get_nodes_by_op_type("ProduceStream"):
        input_args = getCustomOp(produce).get_input_stream_cpp(model)
        for arg in input_args:
            arg.primitive = arg.primitive + "&"
            kernel_function.add_argument(arg)

    for consume in model.get_nodes_by_op_type("ConsumeStream"):
        output_args = getCustomOp(consume).get_output_stream_cpp(model)
        for arg in output_args:
            arg.primitive = arg.primitive + "&"
            kernel_function.add_argument(arg)

    # Add the function prototype, which will be called from the main function.
    cwr.add_function_prototype(kernel_function)
    
    # Main testbench function definition
    main_function = cpp_function(
        name="main",
        return_type="int",
        arguments=[cpp_variable("argc", "int"), cpp_variable("argv", "char**")],
    )

    # Add file and streams declarations
    file_arg_idx = 1
    file_map = {}
    for produce in model.get_nodes_by_op_type("ProduceStream"):
        file_name = "file_" + str(file_arg_idx)
        file_map[produce.name] = file_name
        main_function.add_code(f"std::string {file_name} = argv[{file_arg_idx}];")
        input_args = getCustomOp(produce).get_input_stream_cpp(model)
        for arg in input_args:
            main_function.add_code(f"{arg.generate_declaration()};")
        file_arg_idx += 1
    
    for consume in model.get_nodes_by_op_type("ConsumeStream"):
        # file_name = "file_" + str(file_arg_idx)
        # file_map[consume.name] = file_name
        # main_function.add_code(f"std::string {file_name} = argv[{file_arg_idx}];")
        output_args = getCustomOp(consume).get_output_stream_cpp(model)
        for arg in output_args:
            main_function.add_code(f"{arg.generate_declaration()};")
        file_arg_idx += 1

    # Add read from file calls for input streams
    for produce in model.get_nodes_by_op_type("ProduceStream"):
        main_function.add_code(f"{getCustomOp(produce).generate_call_read_input_from_file(model, file_map[produce.name])};")

    kernel_arguments = []
    for produce in model.get_nodes_by_op_type("ProduceStream"):
        for arg in getCustomOp(produce).get_input_stream_cpp(model):
            kernel_arguments.append(arg.name)

    for consume in model.get_nodes_by_op_type("ConsumeStream"):
        for arg in getCustomOp(consume).get_output_stream_cpp(model):
            kernel_arguments.append(arg.name)

    # Add the kernel function call
    main_function.add_code(f"{kernel_function.generate_call([], *kernel_arguments)};")

    main_function.add_code("return 0;")
    cwr.add_function_definition(main_function)
    return cwr.code

def generate_tcl_script(top_name, part_name, frequency, hls_version, input_files):
    """Dump a TCL script to set up the HLS project and run the simulation."""

    argv = " ".join(input_files)
    tb_files = " ".join(input_files + ["testbench.cpp"])
    t_clk = f"{1e3 / int(frequency):.2f}ns" # Convert frequency in MHz to clock period in ns
    lines = list()
    lines.append("# Auto-generated TCL script for HLS project setup")
    lines.append("# Generated by nn2fpga simulation flow")
    lines.append("")

    # Check the HLS version to determine the correct syntax
    if float(hls_version) > 2025:
        lines.append(
            'open_component -reset "proj_{top_name}" -flow_target vivado',
        )
    else:
        lines.extend(
            [
                'open_project -reset "proj_{top_name}"',
                'open_solution -reset solution0',
            ]
        )

    lines.extend(
        [
            'add_files fifo_depth.cpp -cflags " -I/workspace/NN2FPGA/nn2fpga/library/include"',
            'add_files -tb "/workspace/NN2FPGA/deps/cnpy/cnpy.cpp"',
            'add_files -tb "{tb_files}" -cflags "-I/workspace/NN2FPGA/deps/cnpy -I/workspace/NN2FPGA/nn2fpga/library/include -lz"',
            'set_top "{top_name}"',
            'set_part {part_name}',
            'create_clock -period {t_clk}',
            'csim_design -argv "{argv}"',
            'exit',
        ]
    )

    return "\n".join(lines).format(
        top_name=top_name, tb_files=tb_files, part_name=part_name, t_clk=t_clk, argv=argv
    )

def make_build_dir(work_dir: str) -> None:
    """Create the working directory for the simulation."""
    os.makedirs(work_dir, exist_ok=True)

class ComputeFifoDepth(Transformation):
    """Compute the FIFO depth for each node in the model."""
    
    def __init__(self, work_root: str = "/tmp"):
        """
        Initializes the ComputeFifoDepth transformation.
        Args:
            work_root (str): The root directory of the project.
        """
        super().__init__()
        self.work_root = f"{work_root}/depth-sim"

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Compute the FIFO depth for each node in the model."""
        graph = model.graph

        # Create the working directory for the simulation.
        make_build_dir(self.work_root)

        with open(os.path.join(self.work_root, "fifo_depth.cpp"), "w") as f:
            f.write(generate_hls_code(model, self.work_root))

        # Write input files for the simulation, filled with zeros.
        input_files = []
        for produce in model.get_nodes_by_op_type("ProduceStream"):
            input_shape = model.get_tensor_shape(produce.input[0])
            data = np.zeros(input_shape, dtype=np.float32)
            np.save(os.path.join(self.work_root, f"{produce.name}.npy"), data)
            input_files.append(os.path.join(self.work_root, f"{produce.name}.npy"))
        
        # Write the driver code.
        with open(os.path.join(self.work_root, "testbench.cpp"), "w") as f:
            f.write(generate_hls_driver(model))

        # Generate the TCL script for the HLS project.
        tcl_script = generate_tcl_script(
            top_name=model.get_metadata_prop("top_name"),
            part_name=model.get_metadata_prop("part_name"),
            frequency=model.get_metadata_prop("frequency"),
            hls_version=model.get_metadata_prop("hls_version"),
            input_files=input_files,
        )
        with open(os.path.join(self.work_root, "setup.tcl"), "w") as f:
            f.write(tcl_script)

        # run the simulation
        subprocess.run(
            ["vitis_hls", "-f", f"{self.work_root}/setup.tcl"],
            cwd=self.work_root,
            check=True
        )

        # Read the fifo depth from the generated json file.
        fifo_depth_file = os.path.join(self.work_root, "fifo_depth.json")
        if not os.path.exists(fifo_depth_file):
            raise FileNotFoundError(f"FIFO depth file not found: {fifo_depth_file}")
        
        fifo_depth_data = {}
        with open(fifo_depth_file, "r") as f:
            fifo_depth_data = json.load(f)
        
        fifo_depths = fifo_depth_data.get("fifo_depth", {})
        if not fifo_depths:
            raise ValueError("No FIFO depth data found in the generated file.")
        
        # Store the FIFO depth in the model metadata.
        for stream_name, depths_array in fifo_depths.items():
            depths = TensorFifoDepth(depths_array)

            # Remove the "stream" suffix from the stream name.
            stream_name = stream_name.replace("_stream", "")

            # Set the custom tensor FIFO depth in the model.
            set_custom_tensor_fifo_depth(model, stream_name, depths)

        return model, False