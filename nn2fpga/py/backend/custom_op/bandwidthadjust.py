from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
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

    def generate_run_call(self, model, name: str) -> str:
        cwr = NewCodeWriter()

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
        
        # Declare the outputs.
        var = cpp_variable(
            f"{self.onnx_node.output[0]}_stream",
            f"{get_stream_type(output_quant, par_attribute['out_ch_par'])}",
            array=[par_attribute["out_w_par"]],
        )
        cwr.add_variable_declaration(var)

        # Create the BandwidthAdjust object.
        BandwidthAdjust = cpp_object(
            name,
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}", "TInputStruct"),
                (f"{get_quant_type(input_quant)}", "TInput"),
                (f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}", "TOutputStruct"),
                (f"{get_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_quant_type(output_quant)}>",
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

        cwr.add_lines(BandwidthAdjust.generate_declaration())

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

        cwr.add_function_call(
            run,
            f"{self.onnx_node.input[0]}_stream",
            f"{self.onnx_node.output[0]}_stream",
        )

        return cwr.code

class BandwidthAdjustIncreaseStreams(BandwidthAdjust):
    """ Node increasing the number of streams in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()
    
    def generate_run_call(self, model):
        return super().generate_run_call(model,"BandwidthAdjustIncreaseStreams")

class BandwidthAdjustDecreaseStreams(BandwidthAdjust):
    """ Node decreasing the number of streams in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()
    
    def generate_run_call(self, model):
        return super().generate_run_call(model,"BandwidthAdjustDecreaseStreams")
    
class BandwidthAdjustIncreaseChannels(BandwidthAdjust):
    """ Node increasing the number of channels in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def generate_run_call(self, model):
        return super().generate_run_call(model,"BandwidthAdjustIncreaseChannels")
    
class BandwidthAdjustDecreaseChannels(BandwidthAdjust):
    """ Node decreasing the number of channels in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def generate_run_call(self, model):
        return super().generate_run_call(model,"BandwidthAdjustDecreaseChannels")