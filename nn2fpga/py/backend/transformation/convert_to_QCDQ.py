from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from onnx import TensorProto, helper, OperatorSetIdProto
import onnx.shape_inference as si
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import numpy as np
from backend.transformation.add_streaming_params import quant_array
from backend.core.acceleratorpackage import AcceleratorPackage
from backend.core.tensor_quant import TensorQuant

def get_tensorproto_dtype(bitwidth, signed):
    """Get the TensorProto data type based on bitwidth and signedness."""
    if bitwidth <= 8:
        if signed:
            return TensorProto.INT8
        else:
            return TensorProto.UINT8
    elif bitwidth <= 16:
        if signed:
            return TensorProto.INT16
        else:
            return TensorProto.UINT16
    elif bitwidth <= 32:
        if signed:
            return TensorProto.INT32
        else:
            return TensorProto.UINT32
    else:
        raise ValueError("Unsupported bitwidth for quantization.")

def toNHWC(tensor_shape):
    """Convert a tensor shape from NCHW to NHWC format."""
    NHWC_shape = [tensor_shape[0]]  # Batch size
    NHWC_shape.extend(tensor_shape[2:])  # Height and Width
    NHWC_shape.append(tensor_shape[1])  # Channels
    return NHWC_shape

def toNCHW(tensor_shape):
    """Convert a tensor shape from NHWC to NCHW format."""
    NCHW_shape = [tensor_shape[0]]  # Batch size
    NCHW_shape.append(tensor_shape[-1])  # Channels
    NCHW_shape.extend(tensor_shape[1:-1])  # Height and Width
    return NCHW_shape

def constant_quant_pattern(
    qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    return qonnx_op.Quant(
        x,
        scale,
        zero_point,
        bitwidth,
        signed=signed,
        narrow=narrow,
        _allow_other_attributes=True,
        _domain="qonnx.custom_op.general",
    )

def is_quant_with_constant_input(
    context, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode, **_
):
    # Ensure all required inputs are constants
    if not all([i.const_value is not None for i in [x, scale, zero_point, bitwidth]]):
        return False

    # Check bitwidth is a scalar and a reasonable value
    bitwidth_val = bitwidth.const_value
    if len(bitwidth_val.shape) != 0:
        return False
    
    bitwidth_scalar = bitwidth_val.numpy().squeeze()
    # It's better to check for a valid range, not just a minimum
    if not (1 <= bitwidth_scalar <= 32): # Assuming a range from 1 to 32 bits
        return False

    # Check that scale is a scalar or 1D tensor
    scale_val = scale.const_value
    if len(scale_val.shape) > 1:
        return False

    # Check that zero_point is a scalar or 1D tensor
    zero_point_val = zero_point.const_value
    if len(zero_point_val.shape) > 1:
        return False

    return True

def quant_constant_to_dequant(
    op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    x_np = x.const_value.numpy()
    scale_np = scale.const_value.numpy().squeeze()
    zero_point_np = zero_point.const_value.numpy().squeeze()
    bitwidth_np = bitwidth.const_value.numpy().squeeze()
    signed_val = signed.value
    narrow_val = narrow.value

    rounding_mode_val = rounding_mode.value if rounding_mode else "ROUND"

    c_x = quant_array(
        x_np,
        scale_np,
        zero_point_np,
        bitwidth_np,
        signed=signed_val,
        narrow=narrow_val,
        rounding_mode=rounding_mode_val,
    )

    data_type = get_tensorproto_dtype(bitwidth_np, signed_val)

    # Create the new quantized constant
    quantized_tensor = helper.make_tensor(
        name=f"quantized_{x.name}", # Use a unique name to avoid conflicts
        data_type=data_type,
        dims=c_x.shape,
        vals=c_x.flatten().tolist(),
    )

    # Create the new Constant node
    quantized_const_node = op.Constant(value=quantized_tensor)

    # The scale and zero_point are the same as in the original Quant node
    # so we just reuse the original ValueInfo objects

    # Create the DequantizeLinear node
    return op.DequantizeLinear(quantized_const_node, scale, zero_point)

def create_const_initializer(model, value, dtype):
    init_name = model.make_new_valueinfo_name()
    model.set_initializer(
        init_name,
        np.array(value, dtype=dtype),
    )
    return init_name

class ConvertToQCDQ(Transformation):
    """Convert the model to use QCDQ nodes for quantization."""

    def __init__(self):
        self._rewrite_rule_set = pattern.RewriteRuleSet(
            [
                pattern.RewriteRule(
                    constant_quant_pattern,
                    quant_constant_to_dequant,
                    is_quant_with_constant_input,
                )
            ],
            commute=True,
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to convert quantization nodes to QCDQ."""

        # Convert the model to QCDQ format
        model = ir.from_proto(model.model)
        model = rewrite(model, pattern_rewrite_rules=self._rewrite_rule_set)
        model = ir.to_proto(model)
        model = ModelWrapper(model)
        model = model.transform(QuantToQCDQ())

        # Add transpose and QuantizeLinear/DequantizeLinear nodes around the nn2fpgaPartition node
        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        partition_node = partition_nodes[0] if partition_nodes else None

        if partition_node:
            ap = AcceleratorPackage.from_json(
                getCustomOp(partition_node).get_nodeattr("accelerator_package")
            )

            new_inputs_map = {}
            for i, inp in enumerate(partition_node.input):

                inp_shape = model.get_tensor_shape(inp)
                if inp_shape is None:
                    continue  # Skip if input shape is not available
            
                inp_shape_nhwc = toNHWC(inp_shape)

                # If the input is already in NHWC format, skip the transpose
                quant_input_name = inp
                if inp_shape != inp_shape_nhwc:
                    quant_input_name = f"{inp}_transposed"
                    
                    perm = list(range(len(inp_shape_nhwc)))
                    perm = toNHWC(perm)  # Convert to NHWC permutation
                    transpose_before = helper.make_node(
                        "Transpose",
                        name=f"{inp}_transpose",
                        inputs=[inp],
                        outputs=[f"{inp}_transposed"],
                        perm=perm,
                    )
                    model.set_tensor_shape(
                        f"{inp}_transposed",
                        inp_shape_nhwc,
                    )
                    model.graph.node.append(transpose_before)

                input_tensor_quant = TensorQuant.from_canonical_name(ap.input_map[inp]["quant"])
                scale_init_name = create_const_initializer(
                    model,
                    input_tensor_quant.scale,
                    np.float32,
                )
                zeropt_init_name = create_const_initializer(
                    model,
                    input_tensor_quant.zeropt,
                    input_tensor_quant.get_numpy_dtype()
                )

                quantize_node = helper.make_node(
                    "QuantizeLinear",
                    inputs=[quant_input_name, scale_init_name, zeropt_init_name],
                    outputs=[f"{inp}_quantized"],
                    name=f"{inp}_quantize",
                    axis=1,  # Channel axis for NHWC
                )
                model.set_tensor_shape(
                    f"{inp}_quantized",
                    inp_shape_nhwc,
                    dtype=input_tensor_quant.get_tensorproto_dtype(),
                )

                model.graph.node.append(quantize_node)
                new_inputs_map[inp] = (i, f"{inp}_quantized")

            # Replace the inputs with the transposed versions
            for old_name, (index, new_name) in new_inputs_map.items():
                partition_node.input[index] = new_name

                # Remove the old item from the input map and add the new one
                ap.input_map[new_name] = ap.input_map.pop(old_name)
                old_shape = ap.input_map[new_name]["shape"]
                ap.input_map[new_name]["shape"] = toNHWC(old_shape)


            new_outputs_map = {}
            for i, out in enumerate(partition_node.output):

                out_shape = model.get_tensor_shape(out)
                if out_shape is None:
                    continue

                # Compute the shape in output to the nn2fpgaPartition node which is channel last format
                out_shape_nhwc = toNHWC(out_shape)

                # If the shapes in channel last and channel first formats are the same, skip
                # the transpose node and assign the output directly to the dequantize node
                dequant_output_name = f"{out}_dequantized"
                if out_shape == out_shape_nhwc:
                    dequant_output_name = out

                output_tensor_quant = TensorQuant.from_canonical_name(ap.output_map[out]["quant"])
                scale_init_name = create_const_initializer(
                    model,
                    output_tensor_quant.scale,
                    np.float32,
                )
                zeropt_init_name = create_const_initializer(
                    model,
                    output_tensor_quant.zeropt,
                    output_tensor_quant.get_numpy_dtype()
                )

                dequantize_node = helper.make_node(
                    "DequantizeLinear",
                    inputs=[f"{out}_quantized", scale_init_name, zeropt_init_name],
                    outputs=[dequant_output_name],
                    name=f"{out}_dequantize",
                    axis=1,  # Channel axis for NHWC
                )

                model.set_tensor_shape(
                    f"{out}_quantized",
                    out_shape_nhwc,
                    dtype=output_tensor_quant.get_tensorproto_dtype(),
                )

                model.graph.node.append(dequantize_node)
                if out_shape != out_shape_nhwc:
                    # Add a Transpose node after the partition node
                    perm = list(range(len(out_shape_nhwc)))
                    perm = toNCHW(perm)  # Convert to NCHW permutation
                    # Create a Transpose node to convert from NHWC to NCHW
                    # This is needed because the output of the partition node is in NHWC format
                    # but the rest of the model expects NCHW format

                    transpose_after = helper.make_node(
                        "Transpose",
                        name=f"{out}_transpose",
                        inputs=[f"{out}_dequantized"],
                        outputs=[out],
                        perm=perm,  # NHWC to NCHW
                    )
                    model.set_tensor_shape(
                        f"{out}_dequantized",
                        out_shape_nhwc,
                    )

                    model.graph.node.append(transpose_after)
                new_outputs_map[out] = (i, f"{out}_quantized")

            # Replace the outputs with the transposed versions
            for old_name, (index, new_name) in new_outputs_map.items():
                # Update the partition node output
                partition_node.output[index] = new_name

                # Remove the old item from the output map and add the new one
                ap.output_map[new_name] = ap.output_map.pop(old_name)
                old_shape = ap.output_map[new_name]["shape"]
                ap.output_map[new_name]["shape"] = toNHWC(old_shape)

            # Set the updated accelerator package back to the partition node
            getCustomOp(partition_node).set_nodeattr(
                "accelerator_package", ap.to_json()
            )

        return model, False
