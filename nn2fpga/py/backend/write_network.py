import os
import sys
import time
import threading
#import onnx
import qonnx
from qonnx.transformation import infer_shapes

from backend.quant import *
from backend.graph import *
from backend.opt import *
import backend.layers.weights as weights
import backend.balance_computations as balance_computations
import backend.balance_reuse as balance_reuse
import backend.main as main
import backend.sim as sim
import backend.kpn_sim as kpn_sim

from onnx import numpy_helper
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
        print(f"{self.job} done in {self.end_time - self.start_time:.2f}s", end='\n', flush=True, file=self.stdout)

    def stop(self):
        self.end_time = time.time()
        self.stop_event.set()

def func():
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect stdout and stderr to a file
    with open('output.log', 'w') as log_file:
        sys.stdout = log_file
        sys.stderr = log_file

        # Create and start the status thread
        status_thread = StatusThread()
        status_thread.start()

        # Your existing program logic here
        print("This is a sample program.")

        # Stop the status thread when the main program is done
        status_thread.stop()
        status_thread.join()

    # Restore the original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

# Expects an ONNX model
def write_network(
    model,
    file_name="network",
    off_chip_storage=False,
    board="ULTRA96v2",
    dynamic_init=False,
    uram_storage=False,
    object_detection=False,
    anchors=[],
    prj_root="/tmp",
    transform=False,
    generate_report_file="tmp.rpt",
    generate_log_file="tmp.log"
):

    print(f"Current log file: {generate_log_file}")

    # Cases in which a master axi interface is needed
    ap_ctrl_chain = off_chip_storage

    read_width = 8

    inferred_model = model.transform(infer_shapes.InferShapes())

    init_info = {}

    for info in model.graph.initializer:
        info_name = info.name.replace(".", "_")
        init_info[info_name] = info
    
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    # original_stderr = sys.stderr

    with open(generate_log_file, "w") as log_file:
        sys.stdout = log_file
        # sys.stderr = log_file

        status_thread = StatusThread("Graph_info", original_stdout)
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
        status_thread = StatusThread("Model optimizations", original_stdout)
        status_thread.start()

        io_dict = opt_steps(
            inferred_model,
            io_dict,
            init_info
        )
        
        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Weights quantization extraction", original_stdout)
        status_thread.start()

        io_dict = weights_quant(
            model,
            io_dict
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Balance performances", original_stdout)
        status_thread.start()

        io_dict = balance_computations.ilp(
            io_dict,
            off_chip_storage,
            inferred_model,
            file_name,
            board,
            generate_report_file,
            prj_root=prj_root
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Compute buffers", original_stdout)
        status_thread.start()

        io_dict = compute_buffers(
            inferred_model,
            io_dict
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Hardware quantization", original_stdout)
        status_thread.start()

        io_dict = hw_quant(
            model,
            io_dict
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Weights packeting", original_stdout)
        status_thread.start()

        io_dict = weights.weights_info(
            inferred_model,
            io_dict,
            init_info,
            off_chip_storage,
            dynamic_init,
            uram_storage
        )


        if off_chip_storage:
            io_dict = balance_reuse.ilp(
                io_dict
            )
        
        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Share reuse", original_stdout)
        status_thread.start()

        # 2 times to be sure that both weights and conv are updated
        io_dict = share_reuse(
            inferred_model,
            io_dict
        )

        io_dict = share_reuse(
            inferred_model,
            io_dict
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Renaming", original_stdout)
        status_thread.start()

        io_dict = rename_nodes(
            io_dict
        )

        io_dict = rename_edges(
            model,
            io_dict
        )
        
        # kpn_sim.simulate(model, io_dict)

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Write model code", original_stdout)
        status_thread.start()

        main.write(
            io_dict,
            model,
            file_name,
            ap_ctrl_chain,
            object_detection=object_detection,
            dynamic_init=dynamic_init,
            prj_root=prj_root,
        )

        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Write memory management code", original_stdout)
        status_thread.start()

        weights.write(
            io_dict,
            model,
            file_name,
            board,
            uram_storage,
            generate_report_file,
            prj_root=prj_root
        )
        
        status_thread.stop()
        status_thread.join()
        status_thread = StatusThread("Write host code", original_stdout)
        status_thread.start()
        
        sim.write(
            io_dict,
            model,
            file_name,
            dynamic_init=dynamic_init,
            prj_root=prj_root
        )
        
        status_thread.stop()
        status_thread.join()

    # Restore the original stdout and stderr
    sys.stdout = original_stdout
    # sys.stderr = original_stderr
