from .resnet8 import resnet8
from tiny_torch.benchmark.training_torch.visual_wake_words.vww_torch import MobileNetV1
from .mobilenetv2 import mobilenetv2

def get_model(dataset, device, Wbits, Abits):
    if dataset == 'cifar10':
        model = resnet8(weight_bits=Wbits,act_bits=Abits).to(device)
    elif dataset == 'imagenet':
        model = mobilenetv2(pretrained=True, progress=True, Abits=Abits, Wbits=Wbits)
    elif dataset == 'vww':
        model = MobileNetV1(num_filters=8, num_classes=2).to(device)
    else:
        assert False, 'dataset unknown !'
    return model