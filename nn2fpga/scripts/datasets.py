import torch
import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from coco import CocoDetection
from coco import preproc as coco_preproc

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

def cifar10_dataloader(batch_size):
    CIFAR10_DIRECTORY = '/home/datasets/cifar10'

    if not os.path.exists(CIFAR10_DIRECTORY):
        print("CIFAR10 Dataset not present")
        exit(0)

    test_data = torchvision.datasets.CIFAR10(
        CIFAR10_DIRECTORY,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.float(),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    ICH = 3
    IH = 32
    IW = 32
    OCH = 10
    OH = 1
    OW = 1

    buffer_dim = [
        ICH*IW*IH,
        OCH*OH*OW
    ]

    return test_loader, buffer_dim

def cifar10_4bit_dataloader(batch_size):
    CIFAR10_DIRECTORY = '/home/datasets/cifar10'

    if not os.path.exists(CIFAR10_DIRECTORY):
        print("CIFAR10 Dataset not present")
        exit(0)

    test_data = torchvision.datasets.CIFAR10(
        CIFAR10_DIRECTORY,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.float(),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    ICH = 3
    IH = 32
    IW = 32
    OCH = 10
    OH = 1
    OW = 1

    buffer_dim = [
        ICH*IW*IH,
        OCH*OH*OW
    ]

    return test_loader, buffer_dim

def coco_dataloader(batch_size):
    COCO_DIRECTORY = '/home/datasets/coco/'

    if not os.path.exists(COCO_DIRECTORY):
        print("COCO Dataset not present")
        exit(0)

    images_path = os.path.join(COCO_DIRECTORY, 'images')
    annotations_path = os.path.join(COCO_DIRECTORY, 'annotations')

    X_AXIS = 360
    Y_AXIS = 640

    test_data = CocoDetection(
        images_path,
        os.path.join(annotations_path, 'instances_val2017.json'),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.float(),
            lambda x: coco_preproc(x, Y_AXIS, X_AXIS),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    ICH = 3
    IH = Y_AXIS
    IW = X_AXIS
    OCH = 16
    MAX_NUM_BOX = 3000

    buffer_dim = [
        ICH*IW*IH,
        OCH*MAX_NUM_BOX
    ]

    return test_loader, buffer_dim

def vw_dataloader(batch_size):
    IMAGE_SIZE = 96
    BASE_DIR = os.path.join("/home/datasets/vw", 'vw_coco2014_96')
    vw_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_args = {
        'transform': vw_transform,
        'root': BASE_DIR
    }
    dataset = torchvision.datasets.ImageFolder

    dataset = dataset(**train_args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    buffer_dim = [
        3*IMAGE_SIZE*IMAGE_SIZE,
        2
    ]

    return loader, buffer_dim

def imagenet_dataloader(batch_size):
    IMAGENET_DIRECTORY = '/home/datasets/Imagenet/'

    if not os.path.exists(IMAGENET_DIRECTORY):
        print("IMAGENET Dataset not present")
        exit(0)

    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    args = {
        'train': False,
        'transform': transform,
        'root': IMAGENET_DIRECTORY,
        'sample_size': None
    }

    dataset = ImageNet(**args)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    ICH = 3
    IH = 224
    IW = 224
    OCH = 1000
    OH = 1
    OW = 1

    buffer_dim = [
        ICH*IW*IH,
        OCH*OH*OW
    ]

    return test_loader, buffer_dim

# class fotovoltaic(Dataset): 
#     def __init__(self, root, train, transform=None, sample_size=None):
#         self.samples = []
#         self.targets = []
#         self.transform = transform
#         self.syn_to_class = {}
#         items = os.listdir(root + "/train/")
#         sorted_items = sorted(items)
#         for class_id, syn_id in enumerate(sorted_items):
#             self.syn_to_class[syn_id] = class_id
#         if train:
#             image_path = root + "/train/"
#         else:
#             image_path = root + "/val/"
#         items = os.listdir(image_path)
#         sorted_items = sorted(items)
#         syn_folder = image_path
#         for syn_id in sorted_items:
#             class_id = 0
#             sample_path = os.path.join(syn_folder, syn_id)
#             self.samples.append(sample_path)
#             self.targets.append(class_id)
#         if sample_size is not None:
#             # Randomly sample a subset of the dataset and targets
#             assert len(self.samples) == len(self.targets)
#             indices = np.random.choice(len(self.samples), sample_size, replace=False)
#             self.samples = [self.samples[i] for i in indices]
#             self.targets = [self.targets[i] for i in indices]
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         x = Image.open(self.samples[idx]).convert("RGB")
#         x = self.transform(x)
#         return torch.tensor(x), torch.tensor(self.targets[idx])

# def fotovoltaic_dataloader(batch_size):
#     IMAGE_SIZE = 640
#     BASE_DIR = '/home/datasets/fotovoltaic_dataset/images/'

#     if not os.path.exists(BASE_DIR):
#         print("Fotovoltaic Dataset not present")
#         exit(0)

#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize(IMAGE_SIZE),
#         torchvision.transforms.CenterCrop(IMAGE_SIZE),
#         torchvision.transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     dataset = fotovoltaic(
#         root=BASE_DIR,
#         train=False,
#         transform=transform,
#         sample_size=None
#     )
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=1
#     )

#     buffer_dim = [
#         3 * 640 * 640,    # Input: CHW = 3×640×640 = 1,228,800
#         # 3 * 600 * 600,    # Input: CHW = 3×640×640 = 1,228,800   I blocks between 1,205,868 and 1,202,067
#         115200 + 28800 + 7200  # Output: sum of all heads = 151,200
#     ]

    # return loader, buffer_dim


import os
import numpy as np
import torch
import cv2
import glob
from pathlib import Path
import sys

IMG_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class LoadImages:
    def __init__(self, root, splits=['train', 'val'], img_size=640, stride=32, auto=True, transforms=None):
        files = []
        for split in splits:
            split_dir = os.path.join(root, 'images', split)
            print(f"[DEBUG] Scanning directory: {split_dir}")
            if os.path.isdir(split_dir):
                found = sorted(glob.glob(os.path.join(split_dir, '*.*')))
                print(f"[DEBUG] Found {len(found)} images in split '{split}'")
                files.extend(found)
            else:
                print(f"[WARNING] Split directory not found: {split_dir}")
        self.files = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        print(f"[DEBUG] Total images loaded from splits {splits}: {len(self.files)}")
        assert len(self.files) > 0, f'No images found in {root} for splits {splits}.'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.count = 0
        self.splits = splits
        self.root = root

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.files):
            raise StopIteration
        path = self.files[self.count]
        im0 = cv2.imread(path)
        assert im0 is not None, f'Image Not Found: {path}'
        if self.transforms:
            im = self.transforms(im0)
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
            im = np.ascontiguousarray(im)
        # Ricava lo split corrente dal path
        split = None
        for s in self.splits:
            if f'/images/{s}/' in path or f'\\images\\{s}\\' in path:
                split = s
                break
        if split is None:
            split = self.splits[0]
        print(f"[DEBUG] Processing image: {path} (split: {split})")
        self.count += 1
        return path, im, im0, split, f'image {self.count}/{len(self.files)} {path}'
 
# class LoadImages(torch.utils.data.Dataset):
#     def __init__(self, root, splits=('train', 'val'),
#                  img_size=640, stride=32, auto=True, transforms=None):
#         self.stride, self.auto = stride, auto
#         self.img_size, self.transforms = img_size, transforms
#         self.splits, self.root = splits, root
#         self.files = []

#         for split in splits:
#             split_dir = os.path.join(root, 'images', split)
#             if os.path.isdir(split_dir):
#                 self.files += [f for f in glob.glob(f"{split_dir}/**/*.*", recursive=True)
#                                if f.split('.')[-1].lower() in IMG_FORMATS]
#         if not self.files:
#             raise RuntimeError(f'No images found in {root} for splits {splits}')

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         im0 = cv2.imread(path)
#         assert im0 is not None, f"Image Not Found: {path}"

#         if self.transforms:                       # user‑supplied Albumentations / torchvision pipeline
#             im = self.transforms(im0)
#         else:                                     # built‑in letterbox
#             im, _, _ = letterbox(im0, self.img_size,
#                                  stride=self.stride, auto=self.auto)
#             im = im.transpose(2, 0, 1)[::-1]      # BGR→RGB, HWC→CHW
#             im = np.ascontiguousarray(im)         # (3, H, W), uint8

#         # figure out split name once
#         split = next((s for s in self.splits if f"/images/{s}/" in path
#                                          or f"\\images\\{s}\\" in path), self.splits[0])
#         return path, im, im0, split

    
def collate_fn(batch):
    paths, ims, im0s, splits = zip(*batch)        # each a tuple length=batch_size
    ims  = torch.from_numpy( np.stack(ims) )      # (B, 3, H, W)  uint8 → tensor
    return paths, ims, im0s, splits

      
def fotovoltaic_dataloader(batch_size):
    IMAGE_SIZE = 640
    BASE_DIR = '/home/datasets/fotovoltaic_dataset/'

    if not os.path.exists(BASE_DIR):
        print("Fotovoltaic Dataset not present")
        exit(0)
    dataset_root = Path(BASE_DIR)
    H, W = IMAGE_SIZE, IMAGE_SIZE
    dataset = LoadImages(dataset_root, splits=["val"], img_size=(H, W), stride=32, auto=False)
    
    # loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=2,
    #     collate_fn=collate_fn
    # )
    
    loader = dataset  # Use the custom dataset loader directly
        
    buffer_dim = [
        3 * IMAGE_SIZE * IMAGE_SIZE,  # Input: CHW = 3×640×640 = 1,228,800
        115200 + 28800 + 7200  # Output: sum of all heads = 151,200
    ]
    
    return loader, buffer_dim