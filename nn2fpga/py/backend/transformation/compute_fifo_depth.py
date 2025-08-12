from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from backend.core.fifo_depth import (
    set_custom_tensor_fifo_depth,
    TensorFifoDepth
)
from backend.util.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
from backend.util.board_util import board_part_names
import os
import json
import subprocess

def generate_hls_code(model: ModelWrapper, work_root: str) -> str:
    """ Generate the HLS code to execute the model in fifo-depth mode. """

    # Retrieve model II
    model_II = int(model.get_metadata_prop("model_II"))

    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")

    # Include files from the nn2FPGA library
    nn2fpga_include_dir = "/workspace/NN2FPGA/nn2fpga/library/include"
    if os.path.isdir(nn2fpga_include_dir):
        for fname in os.listdir(nn2fpga_include_dir):
            if fname.endswith(".hpp"):
                cwr.include(fname)

    cwr.include("utils/CSDFG_utils.hpp")
    cwr.include("<fstream>")
    cwr.include("<unordered_set>")

    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")

    for const_input_name in {init.name for init in model.graph.initializer if "const_" in init.name}:
        var = cpp_variable(f"{const_input_name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_code(f"{var.generate_declaration()};")

    stream_vars = []
    stream_count = 0
    for node in model.graph.node:

        # Declare the output streams.
        for stream in getCustomOp(node).get_output_stream_cpp(model):
            function.add_code(f"{stream.generate_declaration()};")
            
            # Do not consider ConsumeStream nodes for streams size calculation.
            if node.op_type != "ConsumeStream":
                stream_vars.append(stream)
                stream_count += stream.array[0] 

        # Declare the variables used in the node
        for var in getCustomOp(node).get_variable_cpp(model):
            function.add_code(f"{var.generate_declaration()};")

        # Generate the object declaration for the custom operation
        if node.op_type != "ProduceStream":
            function.add_code(getCustomOp(node).get_object_cpp(model).generate_declaration())
        else:
            # For ProduceStream, we need to pass the model II for the FixedThroughputProducer
            function.add_code(
                getCustomOp(node).get_object_cpp(model, model_II).generate_declaration()
            )

    # Declare the array of streams sizes.
    stream_sizes = cpp_variable("stream_max_size", primitive="size_t", value=[2] * stream_count) 
    function.add_code(stream_sizes.generate_initialization())

    # Declare the clock cycle counter.
    clock_cycle = cpp_variable("clock_cycle", primitive="size_t", value=0)
    function.add_code(clock_cycle.generate_initialization())

    # Declare the CSDFGState and CSDFGStateHasher for the visited states.
    function.add_code("std::unordered_set<CSDFGState, CSDFGStateHasher> visited_states;")
    function.add_code("CSDFGState current_state;")
    
    # Write the while loop to process the data until all the processors are waiting.
    function.add_code("while (true) {")
    function.codewriter.indent()
    function.add_code("std::vector<ActorStatus> actor_statuses;")
    function.add_code("std::vector<size_t> channel_quantities;")
    function.add_code("ActorStatus actor_status;")

    # Execute a step for each node in the model in reverse order.
    # It must be done in reverse order to ensure that nodes cannot immediately consume the data produced by the previous node.
    for node in reversed(model.graph.node):
        function.add_code(f"actor_status = {getCustomOp(node).generate_step_call()};")
        function.add_code("actor_statuses.push_back(actor_status);")

    # Update the fifo max size for each stream.
    iter = 0
    for stream in stream_vars:
        for _ in range(stream.array[0]):
            function.add_code(f"stream_max_size[{iter}] = std::max<size_t>({stream.name}[0].size(), stream_max_size[{iter}]);")
            function.add_code(f"channel_quantities.push_back({stream.name}[0].size());")
            iter += 1

    function.add_code("current_state = CSDFGState(actor_statuses, channel_quantities);")
    function.add_code("clock_cycle++;")
    function.add_code("if (visited_states.find(current_state) != visited_states.end()) {")
    function.codewriter.indent()
    function.add_code("break;")
    function.codewriter.dedent()
    function.add_code("}")
    function.add_code("visited_states.insert(current_state);")
    function.codewriter.dedent()
    function.add_code("};")

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

    # Accelerator kernel function definition
    kernel_function = cpp_function(
        name=model.get_metadata_prop("top_name"),
        return_type="void",
        qualifiers=["extern"],
    )

    # Add the function prototype, which will be called from the main function.
    cwr.add_function_prototype(kernel_function)
    
    # Main testbench function definition
    main_function = cpp_function(
        name="main",
        return_type="int",
        arguments=[cpp_variable("argc", "int"), cpp_variable("argv", "char**")],
    )

    # Add the kernel function call
    main_function.add_code(f"{kernel_function.generate_call()};")

    main_function.add_code("return 0;")
    cwr.add_function_definition(main_function)
    return cwr.code

def generate_tcl_script(top_name, part_name, frequency, hls_version):
    """Dump a TCL script to set up the HLS project and run the simulation."""

    t_clk = f"{1e3 / int(frequency):.2f}ns" # Convert frequency in MHz to clock period in ns
    lines = list()
    lines.append("# Auto-generated TCL script for HLS project setup")
    lines.append("# Generated by nn2FPGA simulation flow.")
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
            'add_files -tb testbench.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"',
            'set_top "{top_name}"',
            'set_part {part_name}',
            'create_clock -period {t_clk}',
            'csim_design',
            'exit',
        ]
    )

    return "\n".join(lines).format(top_name=top_name, part_name=part_name, t_clk=t_clk)

def make_build_dir(work_dir: str) -> None:
    """Create the working directory for the simulation."""
    os.makedirs(work_dir, exist_ok=True)

class ComputeFifoDepth(Transformation):
    """Compute the FIFO depth for each node in the model."""
    
    def __init__(self, work_root: str = "/tmp", erase: bool = True):
        """
        Initializes the ComputeFifoDepth transformation.
        Args:
            work_root (str): The root directory of the project.
            erase (bool): If True, the HLS project directory will be erased after the simulation.
        """
        super().__init__()
        self.work_root = f"{work_root}/depth-sim"
        self.erase = erase

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Compute the FIFO depth for each node in the model."""
        graph = model.graph

        # Create the working directory for the simulation.
        make_build_dir(self.work_root)

        with open(os.path.join(self.work_root, "fifo_depth.cpp"), "w") as f:
            f.write(generate_hls_code(model, self.work_root))

        # Write the driver code.
        with open(os.path.join(self.work_root, "testbench.cpp"), "w") as f:
            f.write(generate_hls_driver(model))

        # Generate the TCL script for the HLS project.
        part_name, _ = board_part_names(
            board=model.get_metadata_prop("board_name"),
        )
        tcl_script = generate_tcl_script(
            top_name=model.get_metadata_prop("top_name"),
            part_name=part_name,
            frequency=model.get_metadata_prop("frequency"),
            hls_version=model.get_metadata_prop("hls_version"),
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
        
        # Optionally erase the working directory.
        if self.erase:
            subprocess.run(["rm", "-rf", self.work_root], check=True)

        return model, False
