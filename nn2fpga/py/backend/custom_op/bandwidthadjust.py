from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.fifo_depth import get_custom_tensor_fifo_depth
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from backend.util.par_utils import get_par_attributes

class BandwidthAdjust(CustomOp):
    """ Node adjusting a streaming tensor to match the bandwidth requirements."""

    def get_nodeattr_types(self):
        return {}

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

    def get_output_stream_cpp(self, model) -> list[cpp_variable]:
        """Get the output stream cpp variables for the BandwidthAdjust node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[cpp_variable]: A list of cpp_variable objects representing the output stream variables.
        """

        # Retrieve the quantization information for the output tensor.
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        # Retrieve the parallelization attributes for the output tensor.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve the FIFO depth for the output tensor.
        fifo_depth = get_custom_tensor_fifo_depth(model, self.onnx_node.output[0])
        pragma_list = []
        if fifo_depth is not None:
            for i, depth in enumerate(fifo_depth.depths):
                pragma_list.append(
                    f"#pragma HLS STREAM variable={self.__get_stream_name(self.onnx_node.output[0])}[{i}] depth={depth}"
                )

        # Create the cpp_variable for the output stream.
        var = cpp_variable(
            name=f"{self.__get_stream_name(self.onnx_node.output[0])}",
            primitive=f"{get_stream_type(output_quant, par_attribute['out_ch_par'])}",
            array=[par_attribute["out_w_par"]],
            pragma=pragma_list,
        )
        return [var]

    def get_input_stream_cpp(self, model) -> list[cpp_variable]:
        """Get the input stream cpp variables for the BandwidthAdjust node.
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

        fifo_depth = get_custom_tensor_fifo_depth(model, self.onnx_node.output[0])
        pragma_list = []
        if fifo_depth is not None:
            for i, depth in enumerate(fifo_depth.depths):
                pragma_list.append(
                    f"#pragma HLS STREAM variable={self.__get_stream_name(self.onnx_node.output[0])}[{i}] depth={depth}"
                )

        var = cpp_variable(
            name=f"{self.__get_stream_name(self.onnx_node.input[0])}",
            primitive=f"{get_stream_type(input_quant, par_attribute['in_ch_par'])}",
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

    def get_object_cpp(self, model, name) -> cpp_object:
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")

        # Create the BandwidthAdjust object.
        BandwidthAdjust = cpp_object(
            name,
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}", "TInputStruct"),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}", "TOutputStruct"),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(output_quant)}>",
                    "Quantizer",
                ),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "IN_CH"),
                (par_attribute["in_w_par"], "IN_W_PAR"),
                (par_attribute["out_w_par"], "OUT_W_PAR"),
                (par_attribute["in_ch_par"], "IN_CH_PAR"),
                (par_attribute["out_ch_par"], "OUT_CH_PAR"),
            ],
        )
        return BandwidthAdjust

    def generate_run_call(self) -> str:
        """ Generates the C++ code necessary to run the BandwidthAdjust node. """

        # Generate the call to the BandwidthAdjust run method.
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
        """ Generates the C++ code necessary to run the BandwidthAdjust node in step mode. """

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


class BandwidthAdjustIncreaseStreams(BandwidthAdjust):
    """ Node increasing the number of streams in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()
    
    def get_object_cpp(self, model):
        return super().get_object_cpp(model, "BandwidthAdjustIncreaseStreams")

class BandwidthAdjustDecreaseStreams(BandwidthAdjust):
    """ Node decreasing the number of streams in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()
    
    def get_object_cpp(self, model):
        return super().get_object_cpp(model, "BandwidthAdjustDecreaseStreams")

class BandwidthAdjustIncreaseChannels(BandwidthAdjust):
    """ Node increasing the number of channels in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def get_object_cpp(self, model):
        return super().get_object_cpp(model, "BandwidthAdjustIncreaseChannels")

class BandwidthAdjustDecreaseChannels(BandwidthAdjust):
    """ Node decreasing the number of channels in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def get_object_cpp(self, model):
        return super().get_object_cpp(model, "BandwidthAdjustDecreaseChannels")
