import sys
import time
import threading
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.core.modelwrapper import ModelWrapper

from backend.graph import *
from backend.opt import *
import backend.layers.weights as weights
import backend.balance_computations as balance_computations
import backend.main as main
import backend.sim as sim
import backend.transformation as transformation
from backend.custom_op.producestream import ProduceStream
from backend.custom_op.consumestream import ConsumeStream
import qonnx.custom_op.general as general
from qonnx.custom_op.registry import getCustomOp

from onnx import numpy_helper
from onnx import helper, OperatorSetIdProto
import numpy as np

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

    model.model.opset_import.append(
        OperatorSetIdProto(domain="backend.custom_op", version=1)
    )

    # Cases in which a master axi interface is needed

    ap_ctrl_chain = off_chip_storage
    model = model.transform(transformation.CustomInferShapes())
    print(model.check_all_tensor_shapes_specified())
    model = model.transform(transformation.InsertProduceStream())
    model = model.transform(transformation.InsertConsumeStream())
    model = model.transform(transformation.CustomInferShapes())
    model = model.transform(InferDataTypes())
    print(model.check_all_tensor_shapes_specified())
    model.save("produce_stream.onnx")
    
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    # original_stderr = sys.stderr

    with open(generate_log_file, "w") as log_file:
        sys.stdout = log_file

        status_thread = StatusThread("Recovering informations from ONNX", original_stdout)
        status_thread.start()

        io_dict = graph_info(
            inferred_model,
            init_info,
            object_detection,
            anchors,
            transform=transform
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Optimizing the model graph", original_stdout)
        status_thread.start()

        io_dict = opt_step(
            inferred_model,
            io_dict,
            init_info,
            True
        )

        status_thread.stop()
        status_thread.join()
        
        status_thread = StatusThread("Balancing performances", original_stdout)
        status_thread.start()

        io_dict = balance_computations.ilp(
            io_dict,
            off_chip_storage,
            inferred_model,
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
            ap_ctrl_chain,
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
