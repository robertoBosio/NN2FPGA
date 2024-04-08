from utils.preprocess import *
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import glob
import random
from torch.utils.data import Dataset, DataLoader

class ToyADMOSDataset_train(Dataset):
    def __init__(self, data_array):
        self.data = torch.from_numpy(data_array).float()
        self.data = self.data.unsqueeze(2)
        self.data = self.data.unsqueeze(3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

class ToyADMOSDataset_test(Dataset):
    def __init__(self, data_array, labels):
        self.data = torch.from_numpy(data_array).float()
        self.data = self.data.unsqueeze(2)
        self.data = self.data.unsqueeze(3)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index / 196]

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
        return torch.tensor(x), torch.tensor(self.targets[idx])

def get_dataset(dataset, cifar=10, sample_size=None):
    if dataset == 'cifar10':

        transforms_sel = cifar_transform
        BASE_DIR = "/home-ssd/datasets/cifar10/"
        train_args = {
            'train': True,
            'download': False,
            'transform': transforms_sel(is_training=True),
            'root': BASE_DIR
        }
        val_args = {
            'train': False,
            'download': False,
            'transform': transforms_sel(is_training=False),
            'root': BASE_DIR
        }
        if cifar == 10:
            print('#### Selected CIFAR-10!')
            dataset = torchvision.datasets.CIFAR10
        elif cifar == 100:
            print('#### Selected CIFAR-100!')
            dataset = torchvision.datasets.CIFAR100
        else:
            assert False, 'dataset unknown !'
        input_shape = (1, 3, 32, 32)
        train_dataset = dataset(**train_args)
        eval_dataset = dataset(**val_args)    
    
    elif dataset == 'cifar10_4bit':

        transforms_sel = cifar_transform_4bit
        BASE_DIR = "/home-ssd/datasets/cifar10/"
        train_args = {
            'train': True,
            'download': False,
            'transform': transforms_sel(is_training=True),
            'root': BASE_DIR
        }
        val_args = {
            'train': False,
            'download': False,
            'transform': transforms_sel(is_training=False),
            'root': BASE_DIR
        }
        
        if cifar == 10:
            dataset = torchvision.datasets.CIFAR10
        elif cifar == 100:
            dataset = torchvision.datasets.CIFAR100
        
        input_shape = (1, 3, 32, 32)
        train_dataset = dataset(**train_args)
        eval_dataset = dataset(**val_args)    
    
    elif dataset == 'ToyADMOS_train':
        print('#### Selected ToyADMOS train!')
        BASE_DIR = "/home/datasets/tinyML/anomaly_detection/ToyCar"
        def file_list_generator(target_dir,
                                dir_name="train",
                                ext="wav"):
            """
            target_dir : str
                base directory path of the dev_data or eval_data
            dir_name : str (default="train")
                directory name containing training data
            ext : str (default="wav")
                file extension of audio files

            return :
                train_files : list [ str ]
                    file list for training
            """
            print("target_dir : {}".format(target_dir))

            # generate training list
            training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
            files = sorted(glob.glob(training_list_path))
            if len(files) == 0:
                print("no_wav_file!!")

            # print("train_file num : {num}".format(num=len(files)))
            return files
        
        files = file_list_generator(BASE_DIR)
        split_index = int(len(files) * 0.9)

        # Shuffle the list randomly
        random.shuffle(files)

        # Divide the list into two parts
        train_files = files[:split_index]
        eval_files =  files[split_index:]
        params = {"n_mels": 128, "frames": 5, "n_fft": 1024, "hop_length": 512, "power": 2.0}

        train_dataset = fft_transform_train(train_files, params)
        eval_dataset = fft_transform_train(eval_files, params)
        train_dataset = ToyADMOSDataset_train(train_dataset)
        eval_dataset = ToyADMOSDataset_train(eval_dataset)
        input_shape = (1, 640, 1, 1)

    elif dataset == 'ToyADMOS_test':
        print('#### Selected ToyADMOS test!')
        BASE_DIR = "/home/datasets/tinyML/anomaly_detection/ToyCar"
        param = {"n_mels": 128, "frames": 5, "n_fft": 1024, "hop_length": 512, "power": 2.0}
        data, labels = fft_transform_test(BASE_DIR, param)
        eval_dataset = []
        for machine_data, machine_labels in zip(data, labels):
            eval_dataset.append(ToyADMOSDataset_test(machine_data, machine_labels))
        input_shape = (1, 640, 1, 1)
        train_dataset = []

    elif dataset == 'imagenet':
        IMG_SIZE = 256
        BASE_DIR = "/home-ssd/datasets/Imagenet/"
        transforms_sel = imagenet_transform
        train_args = {
            'train': True,
            'transform': transforms_sel(is_training=True, IMAGE_SIZE=IMG_SIZE),
            'root': BASE_DIR,
            'sample_size': sample_size
        }
        val_args = {
            'train': False,
            'transform': transforms_sel(is_training=False, IMAGE_SIZE=IMG_SIZE),
            'root': BASE_DIR,
            'sample_size': None
        }
        dataset = ImageNet
        input_shape = (1, 3, IMG_SIZE, IMG_SIZE)
        train_dataset = dataset(**train_args)
        eval_dataset = dataset(**val_args)
    
    elif dataset == 'vww':
        print('#### Selected VWW!')
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
        assert False, 'dataset unknown!'

    return train_dataset, eval_dataset, input_shape
