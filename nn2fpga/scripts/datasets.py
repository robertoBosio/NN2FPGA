import torch
import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from coco import CocoDetection
from coco import preproc as coco_preproc


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
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
