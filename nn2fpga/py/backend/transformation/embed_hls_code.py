from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from backend.util.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
from backend.core.acceleratorpackage import AcceleratorPackage
from onnx import NodeProto
import base64
import os
import numpy as np

def generate_hls_code(model: ModelWrapper) -> str:

    """Generate HLS code for the given model.
    Args:
        model (ModelWrapper): The model to generate HLS code for.
    Returns:
        str: The generated HLS code as a string.
    """
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
    
    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")
    function.add_code("#pragma HLS TOP")
    function.add_code("#pragma HLS DATAFLOW disable_start_propagation")
    function.add_code("#pragma HLS INTERFACE ap_ctrl_none port=return")

    for produce in model.get_nodes_by_op_type("ProduceStream"):
        for stream in getCustomOp(produce).get_input_stream_cpp(model):
            stream.primitive = stream.primitive + "&"
            function.add_argument(stream)
            for pragma in stream.pragma:
                function.add_code(pragma)

    for const_input_name in {init.name for init in model.graph.initializer if "const_" in init.name}:
        var = cpp_variable(f"{const_input_name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_argument(var)
        function.add_code(f"#pragma HLS INTERFACE axis port={const_input_name}_stream")

    for consume in model.get_nodes_by_op_type("ConsumeStream"):
        for stream in getCustomOp(consume).get_output_stream_cpp(model):
            stream.primitive = stream.primitive + "&"
            function.add_argument(stream)
            for pragma in stream.pragma:
                function.add_code(pragma)

    for node in model.graph.node:

        # Declare the output streams, not for ConsumeStream nodes which are arguments to the top function
        if node.op_type != "ConsumeStream":
            for stream in getCustomOp(node).get_output_stream_cpp(model):
                function.add_code(f"{stream.generate_declaration()};")
                for pragma in stream.pragma:
                    function.add_code(pragma)

        # Declare the variables used in the node
        for var in getCustomOp(node).get_variable_cpp(model):
            function.add_code(f"{var.generate_declaration()};")

        # Generate the object declaration for the custom operation
        function.add_code(getCustomOp(node).get_object_cpp(model).generate_declaration())

        # Generate the run call for the custom operation
        function.add_code(f"{getCustomOp(node).generate_run_call()};")

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
    cwr.include("utils/testbench_utils.hpp")

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
        file_name = "file_" + str(file_arg_idx)
        file_map[consume.name] = file_name
        main_function.add_code(f"std::string {file_name} = argv[{file_arg_idx}];")
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

    # Add write to file calls for output streams
    for consume in model.get_nodes_by_op_type("ConsumeStream"):
        main_function.add_code(f"{getCustomOp(consume).generate_call_write_output_to_file(model, file_map[consume.name])};")

    main_function.add_code("return 0;")
    cwr.add_function_definition(main_function)
    return cwr.code

def encode_array(arr: np.ndarray):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii")
    }

def generate_constant_input_values(model: ModelWrapper, partition_node: NodeProto) -> dict:
    """
    Generate a dictionary of input values for the costant inputs of the model.
    Args:
        model (ModelWrapper): The model to generate input values for.
    Returns:
        dict: A dictionary mapping input names to their constant values.
    """
    constant_inputs = {}
    init_dict = {init.name: init for init in model.graph.initializer}

    for tensor_name in init_dict:
        if "const_" in tensor_name:
            tensor = model.get_initializer(tensor_name)
            if tensor is not None:
                constant_inputs[tensor_name] = encode_array(tensor)
            else:
                raise ValueError(f"Initializer '{tensor_name}' not found in model.")

    return constant_inputs   

class EmbedHLSCode(Transformation):
    """
    Class to handle the conversion of ONNX models to HLS (High-Level Synthesis) format.
    """

    def __init__(self, nn2fpga_model: ModelWrapper, work_root: str = "/tmp", erase: bool = True):
        """
        Initializes the OnnxToHLS transformation.
        Args:
            work_root (str): The root directory of the project.
            nn2fpga_model (ModelWrapper): The model ready to be converted to HLS.
            erase (bool): If True, the starting onnx models will be erased after the transformation.
        """
        super().__init__()
        self.work_root = work_root
        self.nn2fpga_model = nn2fpga_model
        self.erase = erase

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        if not partition_nodes:
            raise ValueError(f"Partition nodes not found in model.")
        
        # We are sure that there is only one nn2FPGA partition node in the model.
        # as this is checked in the supported partition transformation.
        partition_node = partition_nodes[0]

        ap = AcceleratorPackage.from_json(
            getCustomOp(partition_node).get_nodeattr("accelerator_package")
        )

        # Update the accelerator package with the HLS code and driver
        ap.work_dir = self.work_root
        ap.hls_code_b64 = base64.b64encode(generate_hls_code(self.nn2fpga_model).encode()).decode("ascii")
        ap.hls_driver_b64 = base64.b64encode(generate_hls_driver(self.nn2fpga_model).encode()).decode("ascii")
        ap.constant_inputs = generate_constant_input_values(self.nn2fpga_model, partition_node)

        getCustomOp(partition_node).set_nodeattr(
            "accelerator_package", ap.to_json()
        )

        if self.erase:
            # Erase the original model file if it exists
            if os.path.exists("partition_FPGA.onnx"):
                os.remove("partition_FPGA.onnx")
            
            if os.path.exists("wrapper_model.onnx"):
                os.remove("wrapper_model.onnx")

        return (model, False)
