from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from onnx import NodeProto
from csnake import CodeWriter, Variable, Function
import base64
import os
import json
import numpy as np

from csnake import Function

class CppFunction(Function):
    def __init__(
        self,
        name: str,
        return_type: str = "void",
        qualifiers=None,
        arguments=None,
        templates=None,  # List[str], e.g. ["typename T", "int N"]
    ) -> None:
        super().__init__(name, return_type, qualifiers, arguments)
        self.templates = templates or []

    def _generate_template_prefix(self) -> str:
        if not self.templates:
            return ""
        return f"template <{', '.join(self.templates)}>"

    def generate_prototype(self, extern: bool = False) -> str:
        # Add template line before the standard prototype
        base = super().generate_prototype(extern)
        # return self._generate_template_prefix() + base
        return base

    def generate_definition(self, indent="    "):
        # Add template line before the function definition
        writer = super().generate_definition(indent)
        if self.templates:
            tmpl_line = self._generate_template_prefix().strip()
            writer.lines.insert(0, tmpl_line)  # Insert before prototype
        return writer

    @property
    def prototype(self) -> str:
        return self.generate_prototype() + ";"

    @property
    def definition(self) -> str:
        return self.generate_definition().code

def generate_hls_code(model: ModelWrapper) -> str:

    """Generate HLS code for the given model.
    Args:
        model (ModelWrapper): The model to generate HLS code for.
    Returns:
        str: The generated HLS code as a string.
    """
    cwr = CodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")

    # Top function definition
    function = CppFunction("top", "void")
    function.add_code("#pragma HLS TOP")
    function.add_code("#pragma HLS DATAFLOW")
    function.add_code("#pragma HLS INTERFACE ap_ctrl_none port=return")
    for input_name in model.graph.input:
        var = Variable(f"{input_name.name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_argument(var)
        function.add_code(f"#pragma HLS INTERFACE axis port={input_name.name}")

    for const_input_name in {init.name for init in model.graph.initializer if "const_" in init.name}:
        var = Variable(f"{const_input_name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_argument(var)
        function.add_code(f"#pragma HLS INTERFACE axis port={const_input_name}")

    for output_name in model.graph.output:
        var = Variable(f"{output_name.name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_argument(var)
        function.add_code(f"#pragma HLS INTERFACE axis port={output_name.name}")
    
    function.add_code("for (int i = 0; i < 10; i++) {")
    function.add_code("     // Example processing loop")
    function.add_code("     // Replace with actual processing logic")

    function.add_code("     ap_uint<8> data = 1;")
    for input_name in model.graph.input:
        function.add_code(f"    ap_uint<8> {input_name.name}_data;")
        function.add_code(f"    {input_name.name}_data = {input_name.name}_stream.read();")
        function.add_code(f"    data += {input_name.name}_data;")
    
    for const_input_name in {init.name for init in model.graph.initializer if "const_" in init.name}:
        function.add_code(f"    ap_uint<8> {const_input_name}_data;")
        function.add_code(f"    {const_input_name}_data = {const_input_name}_stream.read();")
        function.add_code(f"    data += {const_input_name}_data;")

    for output_name in model.graph.output:
        function.add_code(f"    {output_name.name}_stream.write(data);")
    function.add_code("}")

    cwr.add_function_definition(function)
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
        "bitstream_b64": "",
        "input_map": generate_input_map(model, partition_node),
        "output_map": generate_output_map(model, partition_node),
        "constant_inputs": generate_constant_input_values(model, partition_node),
        "work_dir": work_dir,
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
