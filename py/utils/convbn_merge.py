import torch
import brevitas
from brevitas.nn import QuantConv2d, QuantIdentity
from brevitas.nn.utils import merge_bn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType

def merge_conv_bn(model) :
    merge_bn(model.module.conv1,model.module.bn1)
    model.module.bn1 = torch.nn.Identity()
    
    merge_bn(model.module.layer1[0].conv1,model.module.layer1[0].bn1)
    model.module.layer1[0].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer1[0].conv2,model.module.layer1[0].bn2)
    #model.module.layer1[0].bn2 = QuantIdentity(bit_width = 8,weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
    #                   weight_quant_type=QuantType.INT,
    #                   weight_scaling_impl_type = ScalingImplType.CONST,
    #                   weight_scaling_const=1.0)
    model.module.layer1[0].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer1[1].conv1,model.module.layer1[1].bn1)
    model.module.layer1[1].bn1 = torch.nn.Identity()
    merge_bn(model.module.layer1[1].conv2,model.module.layer1[1].bn2)
    model.module.layer1[1].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer1[2].conv1,model.module.layer1[2].bn1)
    model.module.layer1[2].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer1[2].conv2,model.module.layer1[2].bn2)
    model.module.layer1[2].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer2[0].conv1,model.module.layer2[0].bn1)
    model.module.layer2[0].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer2[0].conv2,model.module.layer2[0].bn2)
    model.module.layer2[0].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer2[1].conv1,model.module.layer2[1].bn1)
    model.module.layer2[1].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer2[1].conv2,model.module.layer2[1].bn2)
    model.module.layer2[1].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer2[2].conv1,model.module.layer2[2].bn1)
    model.module.layer2[2].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer2[2].conv2,model.module.layer2[2].bn2)
    model.module.layer2[2].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer3[0].conv1,model.module.layer3[0].bn1)
    model.module.layer3[0].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer3[0].conv2,model.module.layer3[0].bn2)
    model.module.layer3[0].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer3[1].conv1,model.module.layer3[1].bn1)
    model.module.layer3[1].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer3[1].conv2,model.module.layer3[1].bn2)
    model.module.layer3[1].bn2 = torch.nn.Identity()

    merge_bn(model.module.layer3[2].conv1,model.module.layer3[2].bn1)
    model.module.layer3[2].bn1 = torch.nn.Identity()

    merge_bn(model.module.layer3[2].conv2,model.module.layer3[2].bn2)
    model.module.layer3[2].bn2 = torch.nn.Identity()


    merge_bn(model.module.layer2[0].downsample[0],model.module.layer2[0].downsample[1])
    model.module.layer2[0].downsample[1] = torch.nn.Identity()
    merge_bn(model.module.layer3[0].downsample[0],model.module.layer3[0].downsample[1])
    model.module.layer3[0].downsample[1] = torch.nn.Identity()

    #merge_bn(model.module.fc,model.module.bn2)
    #model.module.bn2 = torch.nn.Identity()
    return model


