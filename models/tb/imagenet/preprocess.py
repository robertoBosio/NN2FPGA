import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import numpy as np
import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx_and_make_model
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model
from utils.datasets import get_dataset

def process_image(n_images, onnx_path, dataset):
    
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    
    batch_size = 1
    train_dataset, eval_dataset, input_shape = get_dataset(dataset)

    counter = 0
    # remove /tmp/images_preprocessed.bin if it exists
    try:
        os.remove("/tmp/images_preprocessed.bin")
    except FileNotFoundError:
        pass

    # remove /tmp/labels_preprocessed.bin if it exists
    try:
        os.remove("/tmp/labels_preprocessed.bin")
    except FileNotFoundError:
        pass

    # remove /tmp/result_preprocessed.bin if it exists
    try:
        os.remove("/tmp/results_preprocessed.bin")
    except FileNotFoundError:
        pass

    with torch.no_grad():
        with open("/tmp/images_preprocessed.bin", "ab") as f_image, \
             open("/tmp/labels_preprocessed.bin", "ab") as f_labels, \
             open("/tmp/results_preprocessed.bin", "ab") as f_res:
            
            for images, labels in eval_dataset:
                f_image.write(np.asarray(torch.permute(images, (1, 2, 0))).flatten().astype(np.float32).tobytes())
                f_labels.write(labels.numpy().astype(np.uint32).tobytes())

                np_images = np.expand_dims(images.numpy(), axis=0)

                outputs = execute_onnx(inferred_model, {'global_in': np_images})
                outputs = outputs['global_out']
                outputs = np.squeeze(outputs)
                f_res.write(outputs.astype(np.float32).tobytes())

                counter += 1
                if counter == int(n_images):
                    break

if __name__ == "__main__":
    print(sys.argv)
    n_images = sys.argv[1]
    onnx_path = sys.argv[2]
    dataset = sys.argv[3]
    process_image(n_images, onnx_path, dataset)
