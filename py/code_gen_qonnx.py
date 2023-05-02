import torch
import onnx
import sys
import os
from backend.write_network import write_network
import network_graph
# import models.resnet20 as resnet20
import models.resnet_brevitas_int as resnet20
import torchvision
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model
def main():


    if "BOARD" not in os.environ:
        print("BOARD PLATFORM NOT DEFINED")
        print("ALLOWED OPTIONS: ULTRA96v2, KRIA")
        sys.exit(0)

    board = os.environ["BOARD"]

    if "ONNX_PATH" not in os.environ:
        print("PATH TO ONNX NOT DEFINED")
        sys.exit(0)

    onnx_path = os.environ["ONNX_PATH"]

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
        print("ALLOWED OPTIONS: ULTRA96v2, KRIA")
        sys.exit(0)

    board = os.environ["BOARD"]

    write_network(
        inferred_model,
        off_chip_storage=False,
        board=board
    )

if __name__ == '__main__':
    main()
