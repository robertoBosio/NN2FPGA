import os
import sys
import backend.transformation as transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.compare_models import test_transformation_equivalence
from backend.analysis.check_quantization import check_quantization
import logging

def nn2fpga_compile(
    onnx_model,
    board="ULTRA96v2",
    frequency=200,
    hls_version="2024.2",
    silvia_packing=False,
    prj_root="/tmp",
    top_name="top",
):

    """Compile an ONNX model for FPGA using nn2FPGA flow.
    Args:
        onnx_model (str or ModelWrapper): Path to the ONNX model.
        board (str): Target FPGA board name.
        part (str): Target FPGA part name.
        frequency (str): Target clock frequency in MHz (without the 'MHz' suffix).
        hls_version (str): Version of HLS to use.
        silvia_packing (bool): Whether to use Silvia packing for resource allocation.
        prj_root (str): Root directory for the project.
        top_name (str): Name of the top-level module in the HLS project.
    Returns:
        None
    """

    # Change the working directory to the project root.
    os.chdir(prj_root)
    logging.basicConfig(level=logging.INFO)

    original_model = ModelWrapper(onnx_model)
    generate_report_file = f"{prj_root}/generate_{top_name}_{board}.rpt"

    # If the file generate_report_file exists, delete it
    if os.path.exists(generate_report_file):
        os.remove(generate_report_file)

    # Save the model before any transformations.
    model = original_model

    # Save target board name in metadata properties.
    model.set_metadata_prop("board_name", board)
    model.set_metadata_prop("top_name", top_name)
    model.set_metadata_prop("frequency", frequency)
    model.set_metadata_prop("hls_version", hls_version)
    model.set_metadata_prop("axilite_address", str(0xA0000000))
    model.set_metadata_prop("axilite_size", str(0x10000))

    # Clean up the model.
    model.cleanup()
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Propagate quantization through quantization invariant nodes.
    model = model.transform(transformation.PropagateQuant())

    # Extract implementable partition.
    nn2fpga_model = model.transform(transformation.SupportedPartition(prj_root))

    # Insert custom nodes.
    nn2fpga_model = nn2fpga_model.transform(transformation.FullyConnectedToConv())
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertProduceStream(nn2fpga_root=prj_root))
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertConsumeStream(nn2fpga_root=prj_root))
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertTensorDuplicator())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())

    # Handle quantization.
    nn2fpga_model = nn2fpga_model.transform(transformation.PropagateQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveRedundantQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())
    nn2fpga_model = nn2fpga_model.transform(GiveReadableTensorNames())
    nn2fpga_model = nn2fpga_model.transform(transformation.LowerToNN2FPGALayers())

    # Start of the backend.
    # Fold quantization into tensor datatype.
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldAsymmetricActQuant())

    # Balance resource allocation per layer.
    nn2fpga_model = nn2fpga_model.transform(
        transformation.BalanceComputation(silvia_packing=silvia_packing, nn2fpga_root=prj_root)
    )
    nn2fpga_model = nn2fpga_model.transform(transformation.AdjustStreamingCommunication())
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertStreamingLineBuffer())
    nn2fpga_model = nn2fpga_model.transform(transformation.InferQuant())

    # Handle weights streaming.
    nn2fpga_model = nn2fpga_model.transform(transformation.AddStreamingParams(nn2fpga_root=prj_root))
    nn2fpga_model = nn2fpga_model.transform(transformation.ComputeFifoDepth(work_root=prj_root))
    model = ModelWrapper("wrapper_model.onnx")
    model = model.transform(
        transformation.EmbedHLSCode(
            nn2fpga_model=nn2fpga_model, work_root=prj_root
        )
    )

    model = model.transform(transformation.GenerateBitstream(work_dir=prj_root))
    model = model.transform(transformation.GenerateDriver(work_dir=prj_root))

    # Simulate the model to check if it works.
    # test_transformation_equivalence(original_model, model)
    
