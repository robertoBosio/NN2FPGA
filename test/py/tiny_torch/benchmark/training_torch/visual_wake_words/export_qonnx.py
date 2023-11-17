# Exporting the model to ONNX format starting from the checkpoint
# and the brevitas model

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
import onnx
from brevitas.export import export_onnx_qcdq
from vww_torch import MobileNetV1
from qonnx.util.convert import convert
from brevitas.export.onnx.qonnx.manager import QONNXManager

IMAGE_SIZE = 128
def export_onnx(checkpoint_path, onnx_path, qonnx_path):
    # Load the model
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    model = MobileNetV1(num_filters=3, num_classes=2)
    model.load_state_dict(checkpoint)
    model.eval()

    # Export the model to ONNX
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    # export_onnx_qcdq(model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])
    exported_model = export_onnx_qcdq(model, args=dummy_input, export_path=onnx_path, opset_version=11)

    QONNXManager.export(model, input_shape=(1, 3, 32, 32), export_path=qonnx_path)

    # Check the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    checkpoint_path = 'trained_models/vww_96.pt'
    onnx_path = 'onnx/model.onnx'
    qonnx_path = 'onnx/qonnx_model.onnx'
    export_onnx(checkpoint_path, onnx_path, qonnx_path)
