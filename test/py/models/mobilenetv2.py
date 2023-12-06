# Mobilenetv2 model for Imagenet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model

def get_torchvision_model(model_name, pretrained=False, progress=True):
    model_fn = getattr(torchvision.models, model_name)
    if model_name == 'inception_v3' or model_name == 'googlenet':
        model = model_fn(pretrained=pretrained, transform_input=False)
    else:
        model = model_fn(pretrained=pretrained)

    return model

def mobilenetv2(pretrained=False, progress=True, Abits=8, Wbits=8):
    model = get_torchvision_model('mobilenet_v2', pretrained, progress)

    model = preprocess_for_quantize(
        model,
        equalize_iters=20,
        equalize_merge_bias=False
    )

    model = quantize_model(
        model,
        backend="layerwise",
        act_bit_width=Abits,
        weight_bit_width=Wbits,
        weight_narrow_range=True,
        bias_bit_width="int16",
        scaling_per_output_channel=False,
        act_quant_percentile=99.999,
        act_quant_type="symmetric",
        scale_factor_type="po2")

    return model