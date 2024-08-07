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
#from utils.datasets import get_dataset
# import coco dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return torch.tensor(img), target

    def __len__(self):
        return len(self.ids)
    

def get_dataset(dataset):
    if dataset == 'coco':
        root = '/data/coco/images/val2017'
        annFile = '/data/coco/annotations/instances_val2017.json'
        transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
        dataset = COCODataset(root, annFile, transform)
        input_shape = (3, 416, 416)
    return dataset, dataset, input_shape

    
def process_image(n_images, onnx_path , dataset):
    
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    
    batch_size = 1
    train_dataset, eval_dataset, input_shape = get_dataset(dataset= 'coco')

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
                #f_labels.write(np.asarray(labels).astype(np.uint32).tobytes())

                np_images = np.expand_dims(images.numpy(), axis=0)

                outputs = execute_onnx(inferred_model, {'images': np_images})
                outputs = outputs['output']
                outputs = np.squeeze(outputs)
                f_res.write(outputs.astype(np.float32).tobytes())

                counter += 1
                if counter == int(n_images):
                    break

if __name__ == "__main__":
    # n_images = sys.argv[1]
    # onnx_path = sys.argv[2]
    # dataset = sys.argv[3]
    n_images = 10
    onnx_path = 'runs/train/yolov3_tiny_quant3/weights/best.qonnx.onnx'
    dataset = 'coco'
    process_image(n_images, onnx_path, dataset)