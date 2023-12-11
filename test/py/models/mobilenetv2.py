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

    # understand why when with the error it was working
    model = quantize_model(
        model,
        backend="fx",
        scale_factor_type="po2_scale",
        bias_bit_width=16,
        weight_bit_width=Wbits,
        weight_narrow_range=True,
        weight_param_method="stats",
        weight_quant_granularity="per_tensor",
        weight_quant_type="sym",
        layerwise_first_last_bit_width=8,
        act_bit_width=8,
        act_param_method="stats",
        act_quant_percentile=99.999,
        act_quant_type="sym",
        quant_format="int"
    )

    return model