import torch
import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np

# def postprocess(out_buffer, results, accuracy, batch_size):
#     predicted = np.argmax(np.asarray(out_buffer[:]), axis=-1)
#     accuracy_batch = np.equal(predicted, results)
#     accuracy_batch = accuracy_batch.sum()
#     accuracy += accuracy_batch
#     return accuracy, accuracy_batch

import numpy as np

# Assume shape info and offsets are known
YOLOV5_HEAD_SHAPES = [
    (1, 3, 80, 80, 18),  # small scale head
    (1, 3, 40, 40, 18),  # medium scale head
    (1, 3, 20, 20, 18),  # large scale head
]

def postprocess(out_bufs, batch_size):
    outputs = []
    for i, out_buf in enumerate(out_bufs):
        output_array = np.asarray(out_buf).reshape(batch_size, -1)
        outputs.append(output_array.copy())  # Important: .copy() to detach from buffer
    return outputs