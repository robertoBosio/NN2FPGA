'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from .common import CommonIntActQuant, CommonUintActQuant
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat 

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                #output_quant = Int8ActPerTensorFixedPoint,
                weight_bit_width = 4,
                input_bit_width = 4
                                 )
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                #output_quant = Int8ActPerTensorFixedPoint,
                weight_bit_width = 4,
                input_bit_width = 4
                                 )
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = QuantConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                output_quant = Int8ActPerTensorFixedPoint,
                weight_bit_width = 4,
                input_bit_width = 4,
                output_bit_width = 8
                                 )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                #nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                QuantConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,
                    weight_quant = Int8WeightPerTensorFixedPoint,
                    bias_quant = Int16Bias,
                    input_quant = Int8ActPerTensorFixedPoint,
                    output_quant = Int8ActPerTensorFixedPoint,
                    weight_bit_width = 4,
                    input_bit_width = 4,
                    output_bit_width = 4
                                    ),
                nn.BatchNorm2d(out_planes),
            )
        self.relu = QuantReLU(quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST,
            act_quant=CommonUintActQuant,
            bit_width=4)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        out = self.relu(out)
        # out = F.relu(self.bn1(out))

        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))
        # out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = QuantConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                #output_quant = Int8ActPerTensorFixedPoint,
                weight_bit_width = 4,
                input_bit_width = 4
                                 )
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        #self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = QuantConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                #output_quant = Int8ActPerTensorFixedPoint,
                weight_bit_width = 4,
                input_bit_width = 4
                                 )
        self.bn2 = nn.BatchNorm2d(1280)
        #self.linear = nn.Linear(1280, num_classes)
        # self.linear = nn.Conv2d(1280, num_classes,
        #         kernel_size=(1, 1), bias=None,
        #         )
        self.linear = QuantConv2d(1280, num_classes,
                kernel_size=(1, 1), bias=None,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                output_quant = Int8ActPerTensorFixedPoint,
                weight_bit_width = 4,
                input_bit_width = 4,
                output_bit_width = 8
                                 )
        self.relu = QuantReLU(quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST,
            act_quant=CommonUintActQuant,
            bit_width=4)
        #self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.linear(out)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layers(out)
        # out = F.relu(self.bn2(self.conv2(out)))
        # # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        # #out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

test()