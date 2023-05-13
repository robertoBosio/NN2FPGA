import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from .common import CommonIntActQuant, CommonUintActQuant
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat 

def conv3x3(in_planes, out_planes, stride=1, weight_bits=8):
    return QuantConv2d(in_planes,
                       out_planes,
                       kernel_size=(3,3), 
                       stride=stride,
                       padding=1,
                       bias=None,
                       weight_quant = Int8WeightPerTensorFixedPoint,
                       bias_quant = Int16Bias,
                       input_quant = Int8ActPerTensorFixedPoint,
                       output_quant = Int8ActPerTensorFixedPoint
                       )
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, weight_bits=8):

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, weight_bits=weight_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = QuantReLU(quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST,
            act_quant=CommonUintActQuant,
            bit_width=8)
        self.conv2 = conv3x3(planes, planes, weight_bits=weight_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, weight_bits=8):
        super(ResNet, self).__init__()
        self.weight_bits = weight_bits
        self.inplanes = 16
        self.conv1 = QuantConv2d(3, 16, kernel_size=(3, 3),
                     weight_bit_width = weight_bits,
                     stride=1, 
                     padding=1, 
                     bias=False, 
                     weight_quant= Int8WeightPerTensorFixedPoint,
                     bias_quant=Int16Bias,
                     input_quant = Int8ActPerTensorFixedPoint,
                     output_quant = Int8ActPerTensorFixedPoint,
                     )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = QuantReLU(quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        #self.avgpool = QuantAvgPool2d(kernel_size=8, stride=1, bit_width = weight_bits)
        self.avgpool = QuantMaxPool2d(kernel_size=8, stride=1)
        self.fc = QuantConv2d(64 * block.expansion, num_classes,
                kernel_size=(1, 1), bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                input_quant = Int8ActPerTensorFixedPoint,
                output_quant = Int8ActPerTensorFixedPoint,
                bias_quant=Int16Bias,
                )


        self.bn2 = nn.BatchNorm2d(num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(
                QuantConv2d(self.inplanes, planes * block.expansion,
                    kernel_size=(1, 1),weight_bit_width = self.weight_bits, 
                    stride=stride, bias=None,
                    weight_quant = Int8WeightPerTensorFixedPoint,
                    input_quant = Int8ActPerTensorFixedPoint,
                    output_quant=Int8ActPerTensorFixedPoint,
                    bias_quant=Int16Bias,
                    ),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample, weight_bits=self.weight_bits))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, weight_bits=self.weight_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        #x = self.bn2(x)
        #return x.view(x.size(0), -1)
        return x
    

def resnet8(num_classes=10, weight_bits=8, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1], num_classes=num_classes,
                    weight_bits=weight_bits)