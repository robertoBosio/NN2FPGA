from utils.preprocess import *
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageNet(Dataset):
    def __init__(self, root, train, transform=None):
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
        for class_id, syn_id in enumerate(sorted_items):
            syn_folder = os.path.join(image_path, syn_id)
            for sample in os.listdir(syn_folder):
                sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(class_id)

    def __len__(self):
            return len(self.samples)
    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            x = self.transform(x)
            return x, self.targets[idx]

def get_dataset(dataset, cifar=10):
    print('#### Loading dataset..')
    if dataset == 'cifar10':
        transforms_sel = cifar_transform
        BASE_DIR = "/home-ssd/datasets/cifar10/"
        train_args = {
            'train': True,
            'download': True,
            'transform': transforms_sel(is_training=True),
            'root': BASE_DIR
        }
        val_args = {
            'train': False,
            'download': True,
            'transform': transforms_sel(is_training=False),
            'root': BASE_DIR
        }
        if cifar == 10:
            print('#### Selected CIFAR-10 !')
            dataset = torchvision.datasets.CIFAR10
        elif cifar == 100:
            print('#### Selected CIFAR-100 !')
            dataset = torchvision.datasets.CIFAR100
        else:
            assert False, 'dataset unknown !'
        input_shape = (1, 3, 32, 32)
        train_dataset = dataset(**train_args)
        eval_dataset = dataset(**val_args)
    elif dataset == 'imagenet':
        print('#### Selected ImageNet !')
        IMG_SIZE = 256
        BASE_DIR = "/home-ssd/datasets/Imagenet/"
        transforms_sel = imagenet_transform
        train_args = {
            'train': True,
            'transform': transforms_sel(is_training=True, IMAGE_SIZE=IMG_SIZE),
            'root': BASE_DIR
        }
        val_args = {
            'train': False,
            'transform': transforms_sel(is_training=False, IMAGE_SIZE=IMG_SIZE),
            'root': BASE_DIR
        }
        dataset = ImageNet
        input_shape = (1, 3, IMG_SIZE, IMG_SIZE)
        train_dataset = dataset(**train_args)
        eval_dataset = dataset(**val_args)
    elif dataset == 'vww':
        print('#### Selected VWW !')
        IMG_SIZE = 96
        BASE_DIR = os.path.join("/home-ssd/datasets/vw", 'vw_coco2014_96')
        transforms_sel=vww_transform
        train_args = {
            'transform': transforms_sel(is_training=True, IMAGE_SIZE=IMG_SIZE),
            'root': BASE_DIR
        }
        val_args = {
            'transform': transforms_sel(is_training=False, IMAGE_SIZE=IMG_SIZE),
            'root': BASE_DIR
        }
        dataset = torchvision.datasets.ImageFolder
        input_shape = (1, 3, IMG_SIZE, IMG_SIZE)

        dataset = dataset(**train_args)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        assert False, 'dataset unknown !'

    return train_dataset, eval_dataset, input_shape
