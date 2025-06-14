import sys
import time
import threading
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, GiveUniqueParameterTensors
from qonnx.core.modelwrapper import ModelWrapper

from backend.graph import *
from backend.opt import *
import backend.layers.weights as weights
import backend.balance_computations as balance_computations
import backend.main as main
import backend.sim as sim
import backend.transformation as transformation
from backend.util.compare_models import test_transformation_equivalence
import qonnx.util.basic as util
from backend.analysis.check_quantization import check_quantization

from onnx import numpy_helper
from onnx import helper, OperatorSetIdProto
import numpy as np

def print_tensor_shapes(model):
    """
    Print all known shape info from inputs, outputs, and value_info.
    Handles both fixed and symbolic dimensions.
    """
    from onnx import TensorProto

    def dim_to_str(dim):
        if dim.HasField("dim_value"):
            return str(dim.dim_value)
        elif dim.HasField("dim_param"):
            return f"{dim.dim_param}"
        else:
            return "?"

    def shape_str(tensor_type):
        dims = tensor_type.shape.dim
        return "[" + ", ".join(dim_to_str(d) for d in dims) + "]"

    def print_vi(vi):
        if not vi.type.HasField("tensor_type"):
            print(f"{vi.name}: (no tensor_type)")
            return
        tt = vi.type.tensor_type
        elem_type = TensorProto.DataType.Name(tt.elem_type)
        shape = shape_str(tt)
        print(f"{vi.name}: {elem_type} {shape}")

    print("== Inputs ==")
    for vi in model.graph.input:
        print_vi(vi)

    print("\n== Outputs ==")
    for vi in model.graph.output:
        print_vi(vi)

    print("\n== ValueInfo ==")
    for vi in model.graph.value_info:
        print_vi(vi)


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



# Expects an ONNX model
def write_network(
    model,
    file_name="network",
    off_chip_storage=False,
    board="ULTRA96v2",
    dynamic_init=False,
    silvia_packing=False,
    object_detection=False,
    anchors=[],
    prj_root="/tmp",
    transform=False,
    generate_report_file="tmp.rpt",
    generate_log_file="tmp.log"
):

    print(f"\nCurrent log file: {generate_log_file}\n")

    # Import nn2FPGA custom operators.
    model.model.opset_import.append(
        OperatorSetIdProto(domain="backend.custom_op", version=1)
    )

    # Save target board name in metadata properties.
    model.set_metadata_prop("board_name", board)

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
    model = model.transform(transformation.InsertProduceStream())
    model = model.transform(transformation.InsertConsumeStream())
    model = model.transform(transformation.InsertTensorDuplicator())

    # Handle quantization.
    model = model.transform(transformation.PropagateQuant())
    model = model.transform(transformation.RemoveRedundantQuant())
    model = model.transform(transformation.CustomInferShapes())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(transformation.FoldQuant())
    model = model.transform(transformation.FoldAsymmetricActQuant())

    model.save("frontend.onnx")

    # Balance resource allocation per layer.
    model = model.transform(
        transformation.BalanceComputation(silvia_packing=silvia_packing, nn2fpga_root=prj_root)
    )

    model.save("balance_computation.onnx")
    exit(-1)

    # Save the original stdout and stderr
    original_stdout = sys.stdout
    # original_stderr = sys.stderr

    with open(generate_log_file, "w") as log_file:
        sys.stdout = log_file

        status_thread = StatusThread("Balancing performances", original_stdout)
        status_thread.start()
        io_dict = {}
        balance_computations.ilp(
            io_dict,
            model,
            file_name,
            board,
            silvia_packing,
            generate_report_file,
            prj_root=prj_root
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Computing buffers", original_stdout)
        status_thread.start()

        io_dict = compute_buffers(
            inferred_model,
            io_dict
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Renaming", original_stdout)
        status_thread.start()

        io_dict = rename_edges(
            model,
            io_dict
        )

        io_dict = rename_nodes(
            io_dict
        )

        io_dict = duplicate_tensor(
            model,
            io_dict,
            True
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Writing model code", original_stdout)
        status_thread.start()

        parsed_write = main.write(
            io_dict,
            model,
            file_name,
            ap_ctrl_chain=off_chip_storage,
            object_detection=object_detection,
            dynamic_init=dynamic_init,
            board=board,
            off_chip_storage=off_chip_storage,
            prj_root=prj_root,
            generate_report_file=generate_report_file
        )

        status_thread.stop()
        status_thread.join()

        if (off_chip_storage):
            status_thread = StatusThread("Writing memory management code", original_stdout)
            status_thread.start()

            weights.write(
                io_dict,
                model,
                file_name,
                board,
                generate_report_file,
                prj_root=prj_root
            )

            status_thread.stop()
            status_thread.join()

        status_thread = StatusThread("Writing host code", original_stdout)
        status_thread.start()

        sim.write(
            io_dict,
            model,
            file_name,
            parsed_write,
            dynamic_init=dynamic_init,
            off_chip_storage=off_chip_storage,
            prj_root=prj_root
        )

        status_thread.stop()
        status_thread.join()

    # Restore the original stdout and stderr
    sys.stdout = original_stdout
    print(f"\nCompilation successful.")
    # sys.stderr = original_stderr
