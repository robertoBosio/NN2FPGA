import torch.nn as nn
from torchsummary import summary
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from .common import CommonIntActQuant, CommonUintActQuant
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint 


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                QuantConv2d(inp,
                       oup,
                       kernel_size=(3,3), 
                       stride=stride,
                       padding=1,
                       bias=None,
                       weight_quant = Int8WeightPerTensorFixedPoint,
                       bias_quant = Int16Bias,
                       input_quant = Int8ActPerTensorFixedPoint,
                       #output_quant = Int8ActPerTensorFixedPoint
                       ),
                nn.BatchNorm2d(oup),
                QuantReLU(quant_type=QuantType.INT,
                          restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                          scaling_impl_type=ScalingImplType.CONST,
                          act_quant=CommonUintActQuant,
                          bit_width=8)               
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                QuantConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False,
                       weight_quant = Int8WeightPerTensorFixedPoint,
                       bias_quant = Int16Bias,
                       input_quant = Int8ActPerTensorFixedPoint,
                       #output_quant = Int8ActPerTensorFixedPoint
                ),
                nn.BatchNorm2d(inp),
                QuantReLU(quant_type=QuantType.INT,
                          restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                          scaling_impl_type=ScalingImplType.CONST,
                          act_quant=CommonUintActQuant,
                          bit_width=8),               
                
                

                # pw
                QuantConv2d(inp, oup, 1, 1, 0, bias=False,
                          weight_quant = Int8WeightPerTensorFixedPoint,
                          bias_quant = Int16Bias,
                          input_quant = Int8ActPerTensorFixedPoint,
                          #output_quant = Int8ActPerTensorFixedPoint
                          ),
                nn.BatchNorm2d(oup),
                QuantReLU(quant_type=QuantType.INT,
                          restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                          scaling_impl_type=ScalingImplType.CONST,
                          act_quant=CommonUintActQuant,
                          bit_width=8),  
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d((1,1))
        )
        #self.fc = nn.Linear(1024, n_classes)
        self.fc = QuantConv2d(1024, n_classes,
                kernel_size=(1, 1), bias=None,
                weight_quant = Int8WeightPerTensorFixedPoint,
                input_quant = Int8ActPerTensorFixedPoint,
                output_quant = Int8ActPerTensorFixedPoint,
                bias_quant=Int16Bias,
                )

    def forward(self, x):
        x = self.model(x)
        #print(x.shape)
        #x = x.view(-1, 1024)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x

if __name__=='__main__':
    # model check
    model = MobileNetV1(ch_in=3, n_classes=10)
    summary(model, input_size=(10,3, 32, 32), device='cpu')