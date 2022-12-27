import torch
import onnx
import sys
import os
from backend.write_network import write_network
import network_graph
# import models.resnet20 as resnet20
import models.cifar_resnet_dorefa as resnet20
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

DATASET = 'CIFAR10'
MODEL = 'TESTMODEL'

DIRECTORY = './data'
os.system('mkdir -p %s' % DIRECTORY)

if (DATASET == 'CIFAR10'):
    test_data = torchvision.datasets.CIFAR10(
        DIRECTORY,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.float()
        ])
    )

if (DATASET == 'IMAGENET'):

    test_data = torchvision.datasets.FakeData(
        num_classes = 1000,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.float()
        ])
    )
    # Imagenet is too large
    # test_data = torchvision.datasets.ImageNet(
    #     DIRECTORY,
    #     train=False,
    #     # download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         lambda x: x.float()
    #     ])
    # )

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    shuffle=True,
    num_workers=2
)

if (MODEL == 'RESNET20'):
    model = resnet20.resnet20(wbits=8, abits=8)
    # state_dict = torch.load(
    #     "./tmp/resnet20.weights",
    #     map_location=torch.device('cpu')
    # )
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(
        torch.load(
            "./tmp/resnet_w8a8/checkpoint.t7",
            map_location=torch.device('cpu')
        )
    )
    model.act_q.export = True
    for name, module in model.named_modules():
      module.export = True
      module.requires_grad_(False)

if (MODEL == 'TESTMODEL'):
    model = resnet20.testmodel(wbits=8, abits=8)
    # state_dict = torch.load(
    #     "./tmp/resnet20.weights",
    #     map_location=torch.device('cpu')
    # )
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(
        torch.load(
            "./ckpt/testmodel1_w8a8/checkpoint.t7",
            map_location=torch.device('cpu')
        )
    )
    model.act_q.export = True
    for name, module in model.named_modules():
      module.export = True
      module.requires_grad_(False)

if (MODEL == 'RESNET50'):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

model.eval()

with torch.no_grad():

    for _, (test_features, test_labels) in enumerate(test_loader):
        break

    onnx_model = network_graph.export_onnx(model, test_features)
    write_network(
        onnx_model,
        off_chip_storage=False
    )

    
