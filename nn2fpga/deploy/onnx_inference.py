import onnxruntime as ort
import numpy as np
import time
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, root, train, transform=None, sample_size=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        items = os.listdir(root + "/train/")
        sorted_items = sorted(items)
        for class_id, syn_id in enumerate(sorted_items):
            self.syn_to_class[syn_id] = class_id

        if train:
            image_path = root + "/train/"
        else:
            image_path = root + "/val/"
        items = os.listdir(image_path)
        sorted_items = sorted(items)
        for syn_id in sorted_items:
            syn_folder = os.path.join(image_path, syn_id)
            class_id = self.syn_to_class[syn_id]
            for sample in os.listdir(syn_folder):
                sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(class_id)
        
        if sample_size is not None:
            # Randomly sample a subset of the dataset and targets
            assert len(self.samples) == len(self.targets)
            indices = np.random.choice(len(self.samples), sample_size, replace=False)
            self.samples = [self.samples[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]

    def __len__(self):
            return len(self.samples)
    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            x = self.transform(x)
            return x, self.targets[idx]

def imagenet_dataloader(batch_size, sample_size=None):
    IMAGENET_DIRECTORY = '/home/datasets/Imagenet/'

    if not os.path.exists(IMAGENET_DIRECTORY):
        print("IMAGENET Dataset not present")
        exit(0)

    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    args = {
        'train': False,
        'transform': transform,
        'root': IMAGENET_DIRECTORY,
        'sample_size': sample_size
    }

    dataset = ImageNet(**args)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    return test_loader

def postprocess(out_buffer, results, accuracy, batch_size):
    predicted = np.argmax(np.asarray(out_buffer[:]), axis=-1)
    accuracy_batch = np.equal(predicted, results)
    accuracy_batch = accuracy_batch.sum()
    accuracy += accuracy_batch
    return accuracy

# Paths
MODEL_PATH = "qcdq_wrapper_model.onnx"              # your ONNX model with nn2fpgaPartition node
CUSTOM_OP_SO = os.path.abspath("libnn2fpga_customop.so")  # <--- absolute path

# Session options
so = ort.SessionOptions()
print("Loading the operator")
so.register_custom_ops_library(CUSTOM_OP_SO)

# Enable optimizations
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable profiling
so.enable_profiling = True

# Create session
print("Starting the session")
sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

# Dummy input data (adapt dtype/shape to your model)
input_name = sess.get_inputs()[0].name
input_shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]
x = np.random.rand(10, 3, 224, 224).astype(np.float32)

# Warmup
print("Warmup run...")
sess.run(None, {input_name: x})
del sess


sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

# Profile multiple runs
dataloader = imagenet_dataloader(batch_size=10, sample_size=100)  # Only 100 samples
accuracy = 0
for batch, (features, expected_results) in enumerate(dataloader):
    np_features = (np.asarray(features).astype(np.float32))
    input_data = {input_name: np_features}
    
    # Run inference
    t1 = time.time()
    actual_result = sess.run(None, input_data)
    t2 = time.time()
    accuracy = postprocess(actual_result, expected_results, accuracy, 10)
    print(f"Batch {batch} processed in {t2 - t1:.3f} seconds")


print(f"Total accuracy: {accuracy / (len(dataloader) * 10):.2f}, which means {accuracy} correct predictions out of {len(dataloader) * 10} total predictions.")
# Close the session to flush the profiling file
prof_file = sess.end_profiling()
print(f"Profiling trace written to: {prof_file}")

