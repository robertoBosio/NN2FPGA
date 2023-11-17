import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from .common import CommonIntActQuant, CommonUintActQuant
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint 

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.

    This block consists of a depthwise convolution followed by a pointwise convolution.
    It is used to reduce the number of parameters and computations in a neural network.

    Attributes:
    - in_channels: int
        Number of input channels.
    - out_channels: int
        Number of output channels.
    - kernel_size: int
        Size of the convolutional kernel.
    - stride: int
        Stride value for the convolution.
    - padding: int
        Padding value for the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Constructor to instantiate the DepthwiseSeparableConv class.

        Parameters:
        - in_channels: int
            Number of input channels.
        - out_channels: int
            Number of output channels.
        - kernel_size: int
            Size of the convolutional kernel.
        - stride: int (default: 1)
            Stride value for the convolution.
        - padding: int (default: 0)
            Padding value for the convolution.
        """

        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution
        self.depthwise = QuantConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            weight_quant = Int8WeightPerTensorFixedPoint,
            bias_quant = Int16Bias,
            input_quant = Int8ActPerTensorFixedPoint,
            weight_quant_bits=8,
            input_quant_bits=8,
            output_quant_bits=8,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = QuantReLU(
            inplace=True,
            quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8
        )

        # Pointwise convolution
        self.pointwise = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_quant = Int8WeightPerTensorFixedPoint,
            bias_quant = Int16Bias,
            input_quant = Int8ActPerTensorFixedPoint,
            output_quant = Int8ActPerTensorFixedPoint,
            weight_quant_bits=8,
            input_quant_bits=8,
            output_quant_bits=8,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = QuantReLU(
            inplace=True,
            quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8
        )


    def forward(self, x):
        """
        Forward pass of the DepthwiseSeparableConv block.

        Parameters:
        - x: torch.Tensor
            Input tensor.

        Returns:
        - torch.Tensor
            Output tensor after passing through the DepthwiseSeparableConv block.
        """

        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class mobilenetresidual(nn.Module):
    """
    Depthwise Separable Convolution block.

    This block consists of a depthwise convolution followed by a pointwise convolution.
    It is used to reduce the number of parameters and computations in a neural network.

    Attributes:
    - in_channels: int
        Number of input channels.
    - out_channels: int
        Number of output channels.
    - kernel_size: int
        Size of the convolutional kernel.
    - stride: int
        Stride value for the convolution.
    - padding: int
        Padding value for the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Constructor to instantiate the DepthwiseSeparableConv class.

        Parameters:
        - in_channels: int
            Number of input channels.
        - out_channels: int
            Number of output channels.
        - kernel_size: int
            Size of the convolutional kernel.
        - stride: int (default: 1)
            Stride value for the convolution.
        - padding: int (default: 0)
            Padding value for the convolution.
        """

        super(mobilenetresidual, self).__init__()

        # Pointwise convolution
        self.pointwise1 = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_quant = Int8WeightPerTensorFixedPoint,
            bias_quant = Int16Bias,
            input_quant = Int8ActPerTensorFixedPoint,
            weight_quant_bits=8,
            input_quant_bits=8,
            output_quant_bits=8,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = QuantReLU(
            inplace=True,
            quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8
        )

        # Depthwise convolution
        self.depthwise = QuantConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            weight_quant = Int8WeightPerTensorFixedPoint,
            bias_quant = Int16Bias,
            input_quant = Int8ActPerTensorFixedPoint,
            weight_quant_bits=8,
            input_quant_bits=8,
            output_quant_bits=8,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = QuantReLU(
            inplace=True,
            quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8
        )

        self.pointwise = QuantConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_quant = Int8WeightPerTensorFixedPoint,
            bias_quant = Int16Bias,
            input_quant = Int8ActPerTensorFixedPoint,
            output_quant = Int8ActPerTensorFixedPoint,
            weight_quant_bits=8,
            input_quant_bits=8,
            output_quant_bits=8,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = QuantReLU(
            inplace=True,
            quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8
        )

    def forward(self, x):
        """
        Forward pass of the DepthwiseSeparableConv block.

        Parameters:
        - x: torch.Tensor
            Input tensor.

        Returns:
        - torch.Tensor
            Output tensor after passing through the DepthwiseSeparableConv block.
        """

        x1 = self.pointwise1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.depthwise(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.pointwise(x1)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)
        x = x + x1

        return x


class QuantizedCifar10Net(nn.Module):
    """
    Neural network quantized with Brevitas for CIFAR-10 dataset.

    This network consists of depthwise separable convolutions and is quantized using Brevitas.

    Attributes:
    - num_classes: int
        Number of output classes.
    """

    def __init__(self, num_classes: int = 10):
        """
        Constructor to instantiate the QuantizedCifar10Net class.

        Parameters:
        - num_classes: int (default: 10)
            Number of output classes.
        """

        super(QuantizedCifar10Net, self).__init__()

        self.features = nn.Sequential(
            QuantConv2d(3, 3,
                kernel_size=(1, 1),weight_bit_width = 8, 
                stride=1, bias=None,
                weight_quant = Int8WeightPerTensorFixedPoint,
                input_quant = Int8ActPerTensorFixedPoint,
                bias_quant=Int16Bias,
                output_quant_bits=8,
            ),
            DepthwiseSeparableConv(3, 32, 3, padding=1),
            mobilenetresidual(32, 32, 3, padding=1),
            QuantReLU(
                inplace=True,
                quant_type=QuantType.INT,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                scaling_impl_type=ScalingImplType.CONST, 
                act_quant=CommonUintActQuant,
                bit_width=8
            ),
            QuantMaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(32, 64, 3, padding=1),
            QuantReLU(
                inplace=True,
                quant_type=QuantType.INT,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                scaling_impl_type=ScalingImplType.CONST, 
                act_quant=CommonUintActQuant,
                bit_width=8
            ),
            QuantMaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(64, 128, 3, padding=1),
            QuantReLU(
                inplace=True,
                quant_type=QuantType.INT,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                scaling_impl_type=ScalingImplType.CONST, 
                act_quant=CommonUintActQuant,
                bit_width=8
            ),
            QuantMaxPool2d(kernel_size=8, stride=1)
        )

        self.classifier = nn.Sequential(
            QuantConv2d(128, 256,
                kernel_size=(1, 1),weight_bit_width = 8, 
                stride=1, bias=None,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                weight_quant_bits=8,
                input_quant_bits=8,
                output_quant_bits=8,
            ),
            QuantReLU(
                inplace=True,
                quant_type=QuantType.INT,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                scaling_impl_type=ScalingImplType.CONST, 
                act_quant=CommonUintActQuant,
                bit_width=8
            ),
            QuantConv2d(256, 10,
                kernel_size=(1, 1),weight_bit_width = 8, 
                stride=1, bias=None,
                weight_quant = Int8WeightPerTensorFixedPoint,
                bias_quant = Int16Bias,
                input_quant = Int8ActPerTensorFixedPoint,
                output_quant = Int8ActPerTensorFixedPoint,
                weight_quant_bits=8,
                input_quant_bits=8,
                output_quant_bits=8,
            ),
        )

    def forward(self, x):
        """
        Forward pass of the QuantizedCifar10Net network.

        Parameters:
        - x: torch.Tensor
            Input tensor.

        Returns:
        - torch.Tensor
            Output tensor after passing through the network.
        """

        x = self.features(x)
        x = self.classifier(x)
        return x