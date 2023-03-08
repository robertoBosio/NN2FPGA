import os
import sys

def io_shape(io):
    input_shapes = [
        d.dim_value for d in io.type.tensor_type.shape.dim
    ]


import onnx
from typing import Iterable


def print_tensor_data(initializer: onnx.TensorProto) -> None:

    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        print(initializer.float_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT32:
        print(initializer.int32_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT64:
        print(initializer.int64_data)
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        print(initializer.double_data)
    elif initializer.data_type == onnx.TensorProto.DataType.UINT64:
        print(initializer.uint64_data)
    else:
        raise NotImplementedError

    return


def dims_prod(dims: Iterable) -> int:

    prod = 1
    for dim in dims:
        prod *= dim

    return prod


def output_shape_quant() :

    model = onnx.load('./onnx/Brevonnx_resnet_final.onnx')
    onnx.checker.check_model(model)

    graph_def = model.graph

    initializers = graph_def.initializer
    print("Hellooooo")
    # Modify initializer
    for initializer in initializers:
        # Data type:
        # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
        print("Tensor information:")
        print(
            f"Tensor Name: {initializer.name}, Data Type: {initializer.data_type}, Shape: {initializer.dims}"
        )
        print("Tensor value before modification:")
        print_tensor_data(initializer)
        # Replace the value with new value.
        #if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
            #for i in range(dims_prod(initializer.dims)):
                #initializer.float_data[i] = 2
        print("Tensor value after modification:")
        print_tensor_data(initializer)
        # If we want to change the data type and dims, we need to create new tensors from scratch.
        # onnx.helper.make_tensor

    # Modify nodes
    nodes = graph_def.node
    for node in nodes:
        print(node.name)
        print(node.op_type)
        print(node.input)
        print(node.output)
        # Modify batchnorm attributes.
        if node.op_type == "Quant":
            print("Attributes before adding:")
            for attribute in node.attribute:
                print(attribute)
            # Add epislon for the BN nodes.
            epsilon_attribute = onnx.helper.make_attribute("output_shape", node.input[0])
            node.attribute.extend([epsilon_attribute])
            # node.attribute.pop() # Pop an attribute if necessary.
            print("Attributes after adding:")
            for attribute in node.attribute:
                print(attribute)

    inputs = graph_def.input
    for graph_input in inputs:
        input_shape = []
        for d in graph_input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        print(
            f"Input Name: {graph_input.name}, Input Data Type: {graph_input.type.tensor_type.elem_type}, Input Shape: {input_shape}"
        )

    outputs = graph_def.output
    for graph_output in outputs:
        output_shape = []
        for d in graph_output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                output_shape.append(None)
            else:
                output_shape.append(d.dim_value)
        print(
            f"Output Name: {graph_output.name}, Output Data Type: {graph_output.type.tensor_type.elem_type}, Output Shape: {output_shape}"
        )

    # To modify inputs and outputs, we would rather create new inputs and outputs.
    # Using onnx.helper.make_tensor_value_info and onnx.helper.make_model

    onnx.checker.check_model(model)
    onnx.save(model, 'onnx/convnets_modified.onnx')


if __name__ == "__main__":

    main() 
