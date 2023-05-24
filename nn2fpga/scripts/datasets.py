import torch
import torchvision
# import torchvision.transforms as transforms
import os


def cifar10_dataloader(batch_size):
    CIFAR10_DIRECTORY = './data'
    os.system('mkdir -p %s' % CIFAR10_DIRECTORY)

    test_data = torchvision.datasets.CIFAR10(
        CIFAR10_DIRECTORY,
        train=False,
        download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.float(),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
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
