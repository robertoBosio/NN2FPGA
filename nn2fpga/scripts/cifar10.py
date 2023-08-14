import torch
import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np

def postprocess(out_buffer, results, accuracy, batch_size):
    predicted = np.argmax(np.asarray(out_buffer[:]), axis=-1)
    accuracy_batch = np.equal(predicted, results)
    accuracy_batch = accuracy_batch.sum()
    accuracy += accuracy_batch
    return accuracy, accuracy_batch
