from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.util.par_utils import get_par_attributes
from backend.util.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
from onnx import NodeProto
import base64
import os
import json
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

def generate_input_map(model: ModelWrapper, partition_node: NodeProto) -> dict:
    """
    Generate a mapping of input names from the parent model to the FPGA model.
    Args:
        model (ModelWrapper): The FPGA model to generate the input map for.
        parent_model (ModelWrapper): The parent model containing the original names.
    Returns:
        dict: A dictionary mapping input names from the parent model to the FPGA model.
    """
    
    input_map = {}
    for i, old_iname in enumerate(partition_node.input):
        new_iname = model.graph.input[i].name
        input_map[old_iname] = new_iname

    return input_map

def generate_output_map(model: ModelWrapper, partition_node: NodeProto) -> dict:
    """
    Generate a mapping of output names from the parent model to the FPGA model.
    Args:
        model (ModelWrapper): The FPGA model to generate the output map for.
        parent_model (ModelWrapper): The parent model containing the original names.
    Returns:
        dict: A dictionary mapping output names from the parent model to the FPGA model.
    """
    
    output_map = {}
    for i, old_oname in enumerate(partition_node.output):
        new_oname = model.graph.output[i].name
        output_map[old_oname] = new_oname
    
    return output_map

def generate_blob(model: ModelWrapper, partition_node: NodeProto, work_dir: str) -> str:
    """
    Generate a base64-encoded blob from the model's code.
    Args:
        model (ModelWrapper): The model to encode.
    Returns:
        str: Base64-encoded string of the model's code.
    """
    blob = {
        "hls_code_b64": base64.b64encode(generate_hls_code(model).encode()).decode("ascii"),
        "hls_driver_b64": base64.b64encode(generate_hls_driver(model).encode()).decode("ascii"),
        "bitstream_b64": "",
        "input_map": generate_input_map(model, partition_node),
        "output_map": generate_output_map(model, partition_node),
        "constant_inputs": generate_constant_input_values(model, partition_node),
        "work_dir": work_dir,
        "part_name": model.get_metadata_prop("part_name"),
        "board_name": model.get_metadata_prop("board_name"),
        "top_name": model.get_metadata_prop("top_name"),
        "frequency": model.get_metadata_prop("frequency"),
        "hls_version": model.get_metadata_prop("hls_version"),
    }

    return json.dumps(blob)

class OnnxToHLS(Transformation):
    """
    Class to handle the conversion of ONNX models to HLS (High-Level Synthesis) format.
    """

    def __init__(self, parent_model: ModelWrapper, work_root: str = "/tmp"):
        """
        Initializes the OnnxToHLS transformation.
        Args:
            work_root (str): The root directory of the project.
        """
        super().__init__()
        self.work_root = work_root
        self.parent_model = parent_model

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        partition_node_name = model.get_metadata_prop("partition node")
        if partition_node_name is None:
            raise ValueError("Partition node name not found in model metadata.")
        
        partition_node = self.parent_model.get_node_from_name(partition_node_name)
        if partition_node is None:
            raise ValueError(f"Partition node '{partition_node_name}' not found in model.")

        getCustomOp(partition_node).set_nodeattr(
            "blob",
            generate_blob(model, partition_node, self.work_root)
        )

        return (model, False)
