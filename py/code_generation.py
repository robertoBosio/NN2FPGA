import torch
import onnx
import sys
import os
from backend.write_network import write_network
import network_graph
import models.resnet20 as resnet20
import torchvision

CIFAR10_DIRECTORY = './data'
os.system('mkdir -p %s' % CIFAR10_DIRECTORY)

train_data = torchvision.datasets.CIFAR10(
    CIFAR10_DIRECTORY,
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: x.float()
    ])
)

test_data = torchvision.datasets.CIFAR10(
    CIFAR10_DIRECTORY,
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: x.float()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=1,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    shuffle=True,
    num_workers=2
)

model = resnet20.resnet20()
state_dict = torch.load(
    "./tmp/resnet20_w8a8.weights",
    map_location=torch.device('cpu')
)

model.load_state_dict(state_dict, strict=False)
model.eval()

with torch.no_grad():

    for _, (train_features, train_labels) in enumerate(train_loader):
        break

    onnx_model = network_graph.export_onnx(model, train_features)
    write_network(onnx_model)

    
