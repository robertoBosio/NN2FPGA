from onnx import helper
from qonnx.core.datatype import DataType
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

    def generate_run_call(self, model):
        """ Generates the C++ code necessary to run the ProduceStream node. """

        cwr = NewCodeWriter()

        # The input has to be an AXI Lite interface, the bitwidth is defined by the board used.
        input_bitwidth = self.get_nodeattr("axi_bitwidth")

        # The output quant is the same as the input quant, since the ProduceStream node
        # does not change the data type of the input tensor.
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])

        # Declare the outputs.
        var = cpp_variable(
            f"{self.onnx_node.output[0]}_stream",
            f"{get_stream_type(output_quant, par_attribute['out_ch_par'])}",
            array=[par_attribute["out_w_par"]],
        )
        cwr.add_variable_declaration(var)

        # Create the ProduceStream object.
        ProduceStream = cpp_object(
            "ProduceStream",
            f"{self.onnx_node.name}",
            [
                (f"ap_axiu<{input_bitwidth}, 0, 0, 0>", "TInputStruct"),
                (f"ap_uint<{input_bitwidth}>", "TInput"),
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

        cwr.add_lines(ProduceStream.generate_declaration())

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

        cwr.add_function_call(run,
            f"{self.onnx_node.input[0]}_stream",
            f"{self.onnx_node.output[0]}_stream")
        return cwr.code
    
    def append_top_inputs(self, function: cpp_function) -> cpp_function:
        """
        Append the input streams to the top function.
        """
        
        for input_name in self.onnx_node.input:
            var = cpp_variable(f"{input_name}_stream", f"hls::stream<ap_axiu<{self.get_nodeattr('axi_bitwidth')}, 0, 0, 0>>&")
            function.add_argument(var)
            function.add_code(f"#pragma HLS INTERFACE axis port={input_name}_stream")
        
        return function
