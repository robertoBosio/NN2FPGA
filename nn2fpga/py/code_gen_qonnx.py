import sys
import os
from backend.write_network import write_network
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model

def main():

    if "TOP_NAME" not in os.environ:
        print("ERROR: The name of the top entity is not defined.")
        sys.exit(1)

    top_name = str(os.environ.get("TOP_NAME"))

    allowed_boards = ["PYNQ", "ZC706", "ULTRA96v2", "KRIA", "ZCU102", "U280", "U250", "U55C"]

    if "BOARD" not in os.environ:
        print("BOARD PLATFORM NOT DEFINED")
        sys.exit(1)

    board = str(os.environ.get("BOARD"))

    if board not in allowed_boards:
        print("ALLOWED OPTIONS: ", end="")
        for name in allowed_boards:
            print(f"{name} ", end="")
        print()
        sys.exit(1)

    if "ONNX_PATH" not in os.environ:
        print("PATH TO ONNX NOT DEFINED")
        sys.exit(1)

    onnx_path = str(os.environ.get("ONNX_FULL_PATH"))

    if "PRJ_FULL_ROOT" not in os.environ:
        print("PROJECT ROOT NOT DEFINED")
        sys.exit(1)

    PRJ_ROOT = str(os.environ.get("PRJ_FULL_ROOT"))

    dynamic_init = False
    if "DYNAMIC_INIT" in os.environ:
        if int(os.environ.get("DYNAMIC_INIT")) == 1:
            dynamic_init = True

    uram_storage = False
    if "URAM_STORAGE" in os.environ:
        if int(os.environ.get("URAM_STORAGE")) == 1:
            uram_storage = True

    off_chip_storage = False
    if "OFF_CHIP_STORAGE" in os.environ:
        if int(os.environ.get("OFF_CHIP_STORAGE")) == 1:
            off_chip_storage = True

    object_detection = False
    if "OBJECT_DETECTION" in os.environ:
        if int(os.environ.get("OBJECT_DETECTION")) == 1:
            object_detection = True

    # if "PACKING" in os.environ:
    #     if int(os.environ.get("PACKING")) == 1:
    #         packing = True
    #     else:
    #         packing = False

    # onnx_path = "./onnx/Brevonnx_resnet_final_fx.onnx"
    # onnx_path = "./onnx/Brevonnx_resnet8_final_fx.onnx"
    # onnx_path = "./onnx/2layer.onnx"
    #onnx_path = "./onnx/CNV_2W2A.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    #write_network( inferred_model , off_chip_storage = False 1)
    inferred_model.set_tensor_datatype("global_in", DataType["UINT8"])
    #assert inferred_model.check_all_tensor_shapes_specified(), "There are still tensors that are not specified"
    
    if "BOARD" not in os.environ:
        print("BOARD PLATFORM NOT DEFINED")
        print("ALLOWED OPTIONS: ", end="")
        for name in allowed_boards:
            print(f"{name} ", end="")
        print()
        sys.exit(0)

    board = os.environ["BOARD"]

    if object_detection:
        # anchors = [5,5, 6,13, 10,8, 16,12, 13,27, 27,19, 42,31, 69,51, 136,110]
        anchors = [9,10, 21,18, 51,38]
        # divide anchors in group of 6 elements
        anchors = [anchors[i:i+6] for i in range(0, len(anchors), 6)]
        
    else:
        anchors = []
    
    if 'DATASET' in os.environ:
        dataset = os.environ['DATASET']
    else:
        dataset = 'cifar10'
    
    transform = bool(int(os.environ['TRANSFORM']))
    generate_report_file = f"{PRJ_ROOT}/generate_{top_name}_{board}.rpt"
    generate_log_file = f"{PRJ_ROOT}/generate_{top_name}_{board}.log"
    
    # If the file generate_report_file exists, delete it
    if os.path.exists(generate_report_file):
        os.remove(generate_report_file)
    
    # If the file generate_report_file exists, delete it
    if os.path.exists(generate_log_file):
        os.remove(generate_log_file)

    write_network(
        inferred_model,
        file_name = top_name,
        off_chip_storage=off_chip_storage,
        board=board, 
        dynamic_init=dynamic_init,
        uram_storage=uram_storage,
        object_detection=object_detection,
        anchors=anchors,
        prj_root=PRJ_ROOT,
        generate_report_file=generate_report_file,
        generate_log_file=generate_log_file,
        transform=transform
    )

if __name__ == '__main__':
    main()