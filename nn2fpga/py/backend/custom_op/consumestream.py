from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.fifo_depth import get_custom_tensor_fifo_depth
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    NewCodeWriter,
    get_cpp_quant_type,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from backend.util.par_utils import get_par_attributes
import math

class ConsumeStream(CustomOp):
    """ Node consuming a streaming tensor to an axi lite interface. """

    def get_nodeattr_types(self):
        return {
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

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

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

        fitting_data = int(math.floor(axi_bitwidth / (output_quant.bitwidth * par_attribute["out_ch_par"] * par_attribute["out_w_par"])))
        return fitting_data * par_attribute["out_ch_par"] * par_attribute["out_w_par"]

    def get_output_stream_cpp(self, model) -> list[cpp_variable]:
        """Get the output stream cpp variables for the ConsumeStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[cpp_variable]: A list of cpp_variable objects representing the output stream variables.
        """
        var = cpp_variable(
            self.__get_stream_name(self.onnx_node.output[0]),
            f"hls::stream<ap_axiu<{self.get_nodeattr('axi_bitwidth')}, 0, 0, 0>>",
            pragma=[
                f"#pragma HLS INTERFACE axis port={self.__get_stream_name(self.onnx_node.output[0])}"
            ],
        )

        return [var]

    def get_input_stream_cpp(self, model) -> list[cpp_variable]:
        """Get the input stream cpp variables for the ConsumeStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[cpp_variable]: A list of cpp_variable objects representing the input stream variables.
        """

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )

        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve the FIFO depth for the input tensor.
        fifo_depth = get_custom_tensor_fifo_depth(model, self.onnx_node.input[0])
        pragma_list = []
        if fifo_depth is not None:
            for i, depth in enumerate(fifo_depth.depths):
                pragma_list.append(
                    f"#pragma HLS STREAM variable={self.__get_stream_name(self.onnx_node.input[0])}[{i}] depth={depth}"
                )

        var = cpp_variable(
            self.__get_stream_name(self.onnx_node.input[0]),
            f"{get_stream_type(input_quant, par_attribute['in_ch_par'])}",
            array=[par_attribute["in_w_par"]],
            pragma=pragma_list,
        )

        return [var]

    def get_variable_cpp(self, model) -> list[cpp_variable]:
        """ Get the internal cpp variables of the ProduceStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[cpp_variable]: A list of cpp_variable objects representing the internal variables.
        """
        return []

    def get_object_cpp(self, model) -> cpp_object:
        """ Generates the cpp ConsumeStream object. 
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: The ConsumeStream as cpp_object.
        """
        # The output has to be an AXI Lite interface, the bitwidth is defined by the board used.
        output_bitwidth = self.get_nodeattr("axi_bitwidth")

        # The output quant is the same as the input quant, since the ConsumeStream node
        # does not change the data type of the input tensor.
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.input[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])

        # Create the ConsumeStream object.
        ConsumeStream = cpp_object(
            "ConsumeStream",
            f"{self.onnx_node.name}",
            [
                (
                    f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}",
                    "TInputStruct",
                ),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"ap_axiu<{output_bitwidth}, 0, 0, 0>", "TOutputStruct"),
                (f"ap_uint<{output_bitwidth}>", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(input_quant)}>",
                    "Quantizer",
                ),
                (self.__get_data_per_word(model), "DATA_PER_WORD"),
                (input_quant.bitwidth, "BITS_PER_DATA"),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "IN_CH"),
                (par_attribute["in_w_par"], "IN_W_PAR"),
                (par_attribute["in_ch_par"], "IN_CH_PAR"),
            ],
        )

        return ConsumeStream

    def generate_run_call(self) -> str:
        """ Generates the C++ code necessary to run the ConsumeStream node. """

        # Generate the call to the ConsumeStream run method.
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
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def generate_step_call(self) -> str:
        """ Generates the C++ code necessary to step the ConsumeStream node. """

        # Generate the call to the ConsumeStream step method.
        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
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

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def generate_call_write_output_to_file(self, model, file_name: str) -> str:
        """
        Generate the code to read the input from a npy file, quantize it and convert it to an HLS stream.
        """
        output_bitwidth = self.get_nodeattr("axi_bitwidth")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model."
            )

        npy_write = cpp_function(
            name=f"hls_stream_to_npy",
            return_type="void",
            templates=["typename TAxi", "typename TData", "typename TDataNumpy"],
            arguments=[
                cpp_variable("input_path", "std::string"),
                cpp_variable(f"stream", f"hls::stream<TAxi>&"),
                cpp_variable("data_per_word", "int"),
                cpp_variable("bits_per_data", "int"),
                cpp_variable("shape", "const std::vector<size_t>&"),
            ],
        )

        return npy_write.generate_call(
            [
                f"ap_axiu<{output_bitwidth}, 0, 0, 0>",
                get_hls_quant_type(output_quant),
                get_cpp_quant_type(output_quant),
            ],
            file_name,
            self.__get_stream_name(self.onnx_node.output[0]),
            self.__get_data_per_word(model),
            output_quant.bitwidth,
            f"{{{output_shape[0]}, {output_shape[2]}, {output_shape[3]}, {output_shape[1]}}}",
        )
