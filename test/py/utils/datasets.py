from utils.preprocess import *
import torchvision
import os

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
