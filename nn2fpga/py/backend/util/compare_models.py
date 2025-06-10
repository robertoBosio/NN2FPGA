import numpy as np
import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

def generate_random_input(model: ModelWrapper) -> dict:
    """
    Generate random input data for the ONNX model based on its input shapes and data types.
    Args:
        model (ModelWrapper): The ONNX model wrapped in QONNX ModelWrapper.
    Returns:
        dict: A dictionary where keys are input names and values are numpy arrays of random data.
    """
    input_dict = {}
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        dtype = model.get_tensor_datatype(inp.name)
        np_dtype = {
            DataType["FLOAT32"]: np.float32,
            DataType["UINT8"]: np.uint8,
            DataType["INT8"]: np.int8,
            DataType["INT32"]: np.int32,
        }.get(dtype, np.float32)
        input_dict[inp.name] = np.random.randn(*shape).astype(np_dtype)
    return input_dict

def get_output_names(model: ModelWrapper) -> list[str]:
    """
    Get the names of the outputs of the ONNX model.
    Args:
        model (ModelWrapper): The ONNX model wrapped in QONNX ModelWrapper.
    Returns:
        list[str]: A list of output names.
    """
    return [out.name for out in model.graph.output]

def test_transformation_equivalence(model_pre: ModelWrapper, model_post: ModelWrapper):
    """
    Test if the outputs of two ONNX models are equivalent given the same random input.
    Args:
        model_pre (ModelWrapper): The original ONNX model before transformation.
        model_post (ModelWrapper): The transformed ONNX model after transformation.
    """
    input_dict = generate_random_input(model_pre)
    output_names = get_output_names(model_pre)

    out_expected = oxe.execute_onnx(model_pre, input_dict)
    out_produced = oxe.execute_onnx(model_post, input_dict)

    for name in output_names:
        flattened_expected = out_expected[name].flatten()
        flattened_produced = out_produced[name].flatten()
        assert name in out_expected and name in out_produced, f"Missing output: {name}"
        assert flattened_expected.shape == flattened_produced.shape, f"Shape mismatch for: {name}"
        assert np.allclose(flattened_expected, flattened_produced, rtol=1e-05, atol=1e-08), f"Output mismatch for: {name}"
