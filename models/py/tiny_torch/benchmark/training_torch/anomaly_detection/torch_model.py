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

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False, weight_bits=8, act_bits=8):
    return qnn.QuantConv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size,kernel_size),
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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, weight_bits=8, act_bits=8):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
#            nn.Linear(input_dim, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=input_dim,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),


#            nn.Linear(128, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
#            
#            nn.Linear(128, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
#            
#            nn.Linear(128, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
#            
#            nn.Linear(128, 8),
#            nn.BatchNorm2d(8),
#            nn.ReLU(),
            conv(
                in_channels=128,
                out_channels=8,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(8),
            relu(),
#            
#            nn.Linear(8, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=8,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
#            
#            nn.Linear(128, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
#            
#            nn.Linear(128, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
            conv(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
#            
#            nn.Linear(128, 128),
#            nn.BatchNorm2d(128),
#            nn.ReLU()
            conv(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            ),
            nn.BatchNorm2d(128),
            relu(),
        )

        self.decoder = nn.Sequential(
            #nn.Linear(128, input_dim)
            conv(
                in_channels=128,
                out_channels=input_dim,
                kernel_size=1,
                padding=0,
                bias=True,
                stride=1,
                weight_bits=weight_bits,
                act_bits=act_bits
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

