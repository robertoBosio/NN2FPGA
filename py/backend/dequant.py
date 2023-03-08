from math import ceil, log2
from quant_dorefa import weight_quantize_fn
import numpy as np
import torch

def dequant(weights):
    wact = weight_quantize_fn(w_bit=8)
    wact.export = False

    # TODO: less dirty
    # weights, max_w = wact(torch.Tensor(weights))
    weights = wact(torch.Tensor(weights))
    weights = np.asarray(weights)

    weights = weights * 128

    # Rescaled to have more precision in hardware, the comma position is
    # taken into account when the output data is extracted from s_acc
    sw = 7 - ceil(log2(np.amax(np.abs(weights))))

    weights = weights * (2**sw)
    weights = np.round(weights)

    return weights, sw
