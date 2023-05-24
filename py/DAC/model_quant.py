import os
import torch
import torch.nn as nn
from layers_quant import Conv, Concat, Detect
from brevitas.nn import QuantMaxPool2d


class Yolo(nn.Module):
    def __init__(self, nc, anchors):
        super(Yolo, self).__init__()
        
        # Define the backbone layers
        self.conv1 = Conv(3, 16, 3, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = Conv(16, 32, 3, 1)
        self.maxpool2 = QuantMaxPool2d(kernel_size=2, stride=2) 
        self.conv3 = Conv(32, 64, 3, 1)
        self.maxpool3 = QuantMaxPool2d(kernel_size=2, stride=2)
        self.conv4 = Conv(64, 128, 3, 1)
        self.maxpool4 = QuantMaxPool2d(kernel_size=2, stride=2)
        self.conv5 = Conv(128, 256, 3, 1)
        self.maxpool5 = QuantMaxPool2d(kernel_size=2, stride=2)
        self.conv6 = Conv(256, 512, 3, 1)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool6 = QuantMaxPool2d(kernel_size=2, stride=1)
        
        # Define the detection layers
        self.conv7 = Conv(512, 1024, 3, 1)
        self.conv8 = Conv(1024, 256, 1, 1)
        self.conv9 = Conv(256, 512, 3, 1)
        self.conv10 = Conv(256, 128, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat(1)
        self.conv11 = Conv(384, 256, 3, 1)
        self.detect = Detect(nc, anchors, [256, 512])
        
    
    def forward(self, x):
        # Backbone
        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool2(x3)
        x5 = self.conv3(x4)
        x6 = self.maxpool3(x5)
        x7 = self.conv4(x6)
        x8 = self.maxpool4(x7)
        x9 = self.conv5(x8)
        x10 = self.maxpool5(x9)
        x11 = self.conv6(x10)
        x12 = self.zeropad(x11)
        x13 = self.maxpool6(x12)
        
        # Detection
        x14 = self.conv7(x13)
        x15 = self.conv8(x14)
        x16 = self.conv9(x15)
        x17 = self.conv10(x15)
        x18 = self.upsample(x17)
        x19 = self.concat(x18, x9)
        x20 = self.conv11(x19)
        y = self.detect([x20, x16])
        
        
        
        return y
 
def yolo():
    return Yolo(nc=7, anchors =[[10,14,23,27,37,58],[81,82,135,169,334,319]])
# Create the model
model = yolo()
# Print the summary
from utils.convbn_merge import *
fuse_layers(model)
replace_layers(model,torch.nn.BatchNorm2d ,torch.nn.Identity())
example_path = './onnx/'
path = example_path + 'Brevonnx_yolov3qonnx.onnx'
os.makedirs(example_path, exist_ok=True)
from brevitas.export.onnx.qonnx.manager import QONNXManager
QONNXManager.export(model, input_shape=(1, 3, 640, 640), export_path='onnx/Brevonnx_yolov3quant.onnx')
# Export to onnx format
# dummy_input = torch.randn(1, 3, 640, 640)
# torch.onnx.export(model, dummy_input, "yolov3-tiny.onnx", verbose=True)


