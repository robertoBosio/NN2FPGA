from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    NewCodeWriter,
    get_struct_type,
    get_stream_type,
    get_quant_type,
)
from backend.util.par_utils import get_par_attributes
import math

class ProduceStream(CustomOp):
    """ Node producing a streaming tensor starting from an axi lite interface. """

    def get_nodeattr_types(self):
        return {
            "normalize": ("i", False, 0),  # 0: no normalization, 1: normalize the input tensor
            "axi_bitwidth": ("i", False, 128),  # Bitwidth of the AXI interface
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node(
            "Identity",
            [node.input[0]],
            [node.output[0]],
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        context[out_name] = inp

    def verify_node(self):
        pass

    def __get_data_per_word(self, model: ModelWrapper) -> int:
        """
        Returns the number of data elements that can be stored in a single word.
        This is calculated as the maximum number of pixels that can be stored in a single AXI word,
        as long as all the channels of it are fitting in the AXI word.
        """
        axi_bitwidth = self.get_nodeattr("axi_bitwidth")
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        par_attribute = get_par_attributes(self.onnx_node)
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        fitting_data = int(math.floor(axi_bitwidth / (output_quant.bitwidth * par_attribute["in_ch_par"] * par_attribute["in_w_par"])))
        return fitting_data * par_attribute["in_ch_par"] * par_attribute["in_w_par"]
    
    def generate_output_stream_declaration(self, model) -> list[str]:
        """ Generate the output stream declaration for the ProduceStream node. """

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        par_attribute = get_par_attributes(self.onnx_node)

        var = cpp_variable(
            self.get_stream_name(self.onnx_node.output[0]),
            f"{get_stream_type(output_quant, par_attribute['out_ch_par'])}",
            array=[par_attribute["out_w_par"]],
        )
        return [var.generate_declaration()]
    
    def generate_variable_declaration(self, model) -> list[str]:
        return ""
    
    def generate_object_declaration(self, model) -> str:
        """ Generates the C++ code necessary to declare the ProduceStream object. """

        input_quant = output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")

        ProduceStream = cpp_object(
            "ProduceStream",
            f"{self.onnx_node.name}",
            [
                (f"ap_axiu<{self.get_nodeattr('axi_bitwidth')}, 0, 0, 0>", "TInputStruct"),
                (f"{get_quant_type(input_quant)}", "TInput"),
                (
                    f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}",
                    "TOutputStruct",
                ),
                (f"{get_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_quant_type(output_quant)}>",
                    "Quantizer",
                ),
                (self.__get_data_per_word(model), "DATA_PER_WORD"),
                (output_quant.bitwidth, "BITS_PER_DATA"),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "OUT_CH"),
                (par_attribute["out_w_par"], "OUT_W_PAR"),
                (par_attribute["out_ch_par"], "OUT_CH_PAR"),
            ],
        )

        return ProduceStream.generate_declaration()

    def generate_run_call(self):
        """ Generates the C++ code necessary to run the ProduceStream node. """

        # Generate the call to the ProduceStream run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"input_data_stream",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"output_data_stream",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return run.generate_call(
            [],
            self.get_stream_name(self.onnx_node.input[0]),
            self.get_stream_name(self.onnx_node.output[0]),
        )

    def append_top_inputs(self, function: cpp_function) -> cpp_function:
        """
        Append the input streams to the top function.
        """

        var = cpp_variable(self.get_stream_name(self.onnx_node.input[0]), f"hls::stream<ap_axiu<{self.get_nodeattr('axi_bitwidth')}, 0, 0, 0>>&")
        function.add_argument(var)
        function.add_code(f"#pragma HLS INTERFACE axis port={self.get_stream_name(self.onnx_node.input[0])}")

        return function

    def get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def get_file_name(self, name: str) -> str:
        """
        Returns the name of the file for the tensor.
        """
        return f"{name}_file"

    def append_call_read_input_from_file(self, model, function: cpp_function) -> cpp_function:
        """
        Generate the code to read the input from a npy file, quantize it and convert it to an HLS stream.
        """
        input_bitwidth = self.get_nodeattr("axi_bitwidth")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        npy_read = cpp_function(
            name=f"npy_to_hls_stream",
            return_type="void",
            templates=["typename TAxi", "typename TData"],
            arguments=[
                cpp_variable("input_path", "std::string"),
                cpp_variable(f"stream", f"hls::stream<TAxi>&"),
                cpp_variable("data_per_word", "int"),
                cpp_variable("bits_per_data", "int"),
                cpp_variable("scale", "float"),
                cpp_variable("zeropt", "int"),
            ],
        )

        function.add_code(
            f"""{
            npy_read.generate_call(
                [
                    f'ap_axiu<{input_bitwidth}, 0, 0, 0>',
                    get_quant_type(output_quant),
                ],
                self.get_file_name(self.onnx_node.input[0]),
                self.get_stream_name(self.onnx_node.input[0]),
                self.__get_data_per_word(model),
                output_quant.bitwidth,
                output_quant.scale,
                output_quant.zeropt,
            )};"""
        )

        return function

    def append_variable_declaration(self, function: cpp_function) -> cpp_function:
        """
        Append the variable declaration for the input stream.
        """
        input_bitwidth = self.get_nodeattr("axi_bitwidth")

        stream_var = cpp_variable(
            self.get_stream_name(self.onnx_node.input[0]),
            f"hls::stream<ap_axiu<{input_bitwidth}, 0, 0, 0>>",
        )

        function.add_code(f"{stream_var.generate_declaration()};")
        return function
