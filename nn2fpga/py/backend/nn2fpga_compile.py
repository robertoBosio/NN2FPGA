import os
import sys
import time
import threading
import numpy as np
import backend.transformation as transformation
import qonnx.util.basic as util
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.compare_models import test_transformation_equivalence
from backend.analysis.check_quantization import check_quantization
from onnx import numpy_helper
from onnx import helper, OperatorSetIdProto

class StatusThread(threading.Thread):
    def __init__(self, job, stdout):
        super(StatusThread, self).__init__()
        self.daemon = True
        self.job = job
        self.start_time = time.time()
        self.end_time = 0
        self.stdout = stdout
        self.stop_event = threading.Event()
        self.loading = ["-", "\\", "|", "/"]

    def run(self):
        index = 0
        while not self.stop_event.is_set():
            print(self.job + "\t" + self.loading[index], end='\r', flush=True, file=self.stdout)
            index = (index + 1) % 4
            time.sleep(0.2)
        print(f"{self.job}. Done in {self.end_time - self.start_time:.2f}s", end='\n', flush=True, file=self.stdout)

    def stop(self):
        self.end_time = time.time()
        self.stop_event.set()


def nn2fpga_compile(
    onnx_model,
    board="ULTRA96v2",
    part="xczu3eg-sbva484-1-e",
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

    original_model = ModelWrapper(onnx_model)
    generate_report_file = f"{prj_root}/generate_{top_name}_{board}.rpt"
    generate_log_file = f"{prj_root}/generate_{top_name}_{board}.log"

    # If the file generate_report_file exists, delete it
    if os.path.exists(generate_report_file):
        os.remove(generate_report_file)

    # If the file generate_report_file exists, delete it
    if os.path.exists(generate_log_file):
        os.remove(generate_log_file)
    print(f"\nCurrent log file: {generate_log_file}\n")

    # Import nn2FPGA custom operators.
    model = original_model
    model.model.opset_import.append(
        OperatorSetIdProto(domain="backend.custom_op", version=1)
    )

    # Save target board name in metadata properties.
    model.set_metadata_prop("board_name", board)
    model.set_metadata_prop("part_name", part)
    model.set_metadata_prop("top_name", top_name)
    model.set_metadata_prop("frequency", frequency)
    model.set_metadata_prop("hls_version", hls_version)

    # Clean up the model.
    model.cleanup()
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Propagate quantization through quantization invariant nodes.
    model = model.transform(transformation.PropagateQuant())
    model = model.transform(InferShapes())

    # Extract implementable partition.
    model = model.transform(transformation.SupportedPartition(prj_root))

    # Insert custom nodes.
    model = model.transform(transformation.FullyConnectedToConv())
    model = model.transform(transformation.InsertProduceStream(nn2fpga_root=prj_root))
    model = model.transform(transformation.InsertConsumeStream(nn2fpga_root=prj_root))
    model = model.transform(transformation.InsertTensorDuplicator())
    model = model.transform(transformation.CustomInferShapes())

    # Handle quantization.
    model = model.transform(transformation.PropagateQuant())
    model = model.transform(transformation.RemoveRedundantQuant())
    model = model.transform(transformation.CustomInferShapes())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(transformation.LowerToNN2FPGALayers())
    # model.save("lowered.onnx")
    # test_transformation_equivalence(fpga_model, model)

    # Start of the backend.
    # Fold quantization into tensor datatype.
    model = model.transform(transformation.FoldQuant())
    model = model.transform(transformation.FoldAsymmetricActQuant())

    # Balance resource allocation per layer.
    model = model.transform(
        transformation.BalanceComputation(silvia_packing=silvia_packing, nn2fpga_root=prj_root)
    )
    model = model.transform(transformation.AdjustStreamingCommunication())
    model = model.transform(transformation.InsertStreamingLineBuffer())
    model = model.transform(transformation.InferQuant())

    # Handle weights streaming.
    model = model.transform(transformation.AddStreamingParams(nn2fpga_root=prj_root))
    model = model.transform(transformation.ComputeFifoDepth(work_root=prj_root))
    model.save("nn2fpga_accel.onnx")
    parent_model = ModelWrapper("wrapper_model.onnx")
    model = model.transform(
        transformation.OnnxToHLS(
            parent_model=parent_model, work_root=prj_root
        )
    )
    parent_model.save("embedHLS.onnx")

    # Simulate the model to check if it works.
    # oxe.execute_onnx(parent_model, {"global_in": np.random.randn(1, 3, 224, 224).astype(np.float32)})
    test_transformation_equivalence(original_model, parent_model)

    # parent_model = parent_model.transform(QuantToQCDQ())
    # parent_model.save("qcdq_wrapper_model.onnx")

    # Save the original stdout and stderr
    original_stdout = sys.stdout
    # original_stderr = sys.stderr

    with open(generate_log_file, "w") as log_file:
        sys.stdout = log_file

    # Restore the original stdout and stderr
    sys.stdout = original_stdout
    print(f"\nCompilation successful.")
    # sys.stderr = original_stderr
