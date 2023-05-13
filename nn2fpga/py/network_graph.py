import torch
import onnx
import sys
import os
import torchvision
import models.resnet_brevitas_int as resnet20
def io_nodes(onnx_model):
    layers_info = {}

    for node in model.nodes:
        layers_info[model.name] = {}
        layers_info[model.name]["inputs"] = node.input
        layers_info[model.name]["outputs"] = node.output

    return layers_info

def export_onnx(model, x):
    torch.onnx.export(
        model,
        args=x,
        f="onnx/Brevonnx_resnet_final.onnx",
        export_params = True,
        do_constant_folding = True,
        input_names = ["input"],
        output_names = ["output"]
    )

    onnx_model = onnx.load("onnx/Brevonnx_resnet_final.onnx")

    return onnx_model

def main():

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
        "/home/minnellf/Xilinx/NN2FPGA/tmp/resnet20.weights",
        map_location=torch.device('cpu')
    )

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():

        for _, (train_features, train_labels) in enumerate(train_loader):
            break

        export_onnx(model, train_features)

if __name__ == '__main__':
    main()
