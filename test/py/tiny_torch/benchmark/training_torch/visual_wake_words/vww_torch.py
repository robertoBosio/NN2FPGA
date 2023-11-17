import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit

class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, weight_bits=8, act_bits=8):
    return qnn.QuantConv2d(
        in_channels, out_channels, 
        kernel_size=(kernel_size, kernel_size),
        weight_bit_width=weight_bits,
        input_bit_width=act_bits,
        outut_bit_width=act_bits,
        stride=stride,
        padding=padding, 
        bias=bias, 
        groups=groups,
        weight_quant= Int8WeightPerTensorFixedPoint,
        bias_quant=Int16Bias,
        input_quant=Int8ActPerTensorFixedPoint,
        output_quant=Int8ActPerTensorFixedPoint,
    )

def relu(bit_width=8):
    return qnn.QuantReLU(
        quant_type=QuantType.INT,
        restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
        scaling_impl_type=ScalingImplType.CONST,
        act_quant=CommonUintActQuant,
        bit_width=bit_width
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=2, num_filters=8):
        super(MobileNetV1, self).__init__()

        # MobileNet parameters
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.features = nn.Sequential(
            conv(3, num_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            relu(),

            conv(num_filters, num_filters, kernel_size=3, stride=1, padding=1, groups=num_filters, bias=False),
            nn.BatchNorm2d(num_filters),
            relu(),
            conv(num_filters, 2 * num_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * num_filters),
            relu(),

            conv(2 * num_filters, 2 * num_filters, kernel_size=3, stride=2, padding=1, groups=2 * num_filters, bias=False),
            nn.BatchNorm2d(2 * num_filters),
            relu(),
            conv(2 * num_filters, 4 * num_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4 * num_filters),
            relu(),

            conv(4 * num_filters, 4 * num_filters, kernel_size=3, stride=1, padding=1, groups=4 * num_filters, bias=False),
            nn.BatchNorm2d(4 * num_filters),
            relu(),
            conv(4 * num_filters, 4 * num_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4 * num_filters),
            relu(),

            conv(4 * num_filters, 8 * num_filters, kernel_size=3, stride=2, padding=1, groups=4 * num_filters, bias=False),
            nn.BatchNorm2d(8 * num_filters),
            relu(),
            conv(8 * num_filters, 8 * num_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8 * num_filters),
            relu(),

            # Repeat identical depthwise separable convs (8th-12th layers)
            self._make_depthwise_block(8 * num_filters),
            self._make_depthwise_block(8 * num_filters),
            self._make_depthwise_block(8 * num_filters),
            self._make_depthwise_block(8 * num_filters),
            self._make_depthwise_block(8 * num_filters),

            conv(8 * num_filters, 16 * num_filters, kernel_size=3, stride=2, padding=1, groups=8 * num_filters, bias=False),
            nn.BatchNorm2d(16 * num_filters),
            relu(),
            conv(16 * num_filters, 16 * num_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16 * num_filters),
            relu(),

            self._make_depthwise_block(16 * num_filters),
            self._make_depthwise_block(16 * num_filters),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            conv(16 * num_filters, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_depthwise_block(self, channels):
        return nn.Sequential(
            conv(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            relu(),
            conv(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            relu()
        )