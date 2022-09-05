import torch
import torchvision
import sys
import os
import models.resnet20 as resnet20
from torchviz import make_dot
from torch.utils.tensorboard.writer import graph
from torchinfo import summary


def extract_leaves(layers_hier, layers, to_be_analyzed):

    to_be_analyzed = []
    for _, elem in layers_hier[1].items():
        if elem[1] == {}:
            if elem[0] != "":
                layers[elem[0]] = [elem[2]]
        else:
            
            extract_leaves(elem, layers, to_be_analyzed)

    return layers, to_be_analyzed

def generate_layers_graph(model, x):

    layers = {}

    def get_layer_info(layer):

        graph = {}

        def process_line(line):
            print(line)
            if "self" in line:
                processed_line = line.split(" = ")
                if "getattr" in line:
                    processed_line = processed_line[1].split("\"")
                    graph[processed_line[0]] = [
                        "self",
                        "id",
                        ["self." + processed_line[1]]
                    ]
                else:
                    graph[processed_line[0]] = [
                        "self",
                        "id",
                        ["self." + processed_line[1]]
                    ]
            else:
                processed_line = line.split(" = ")
                output_value = [processed_line[0]]
                processed_line = processed_line[1].split("(")
                print(processed_line)
                for elem in processed_line:
                    if "forward" not in elem:
                        if "torch" in elem:
                            continue



        if layer.name not in layers.keys():
            # print(layer.name)
            # print(layer.code)
            for line in layer.code.split("\n"):
                if "=" in line:
                    process_line(line)
                # print(line.split(""))
        # print(layer)
        # print(layer.code)
        # print(layer.name)

    # y = model(x)
    to_be_analyzed = [model]
    layers_hier = ["", {}, model]
    while to_be_analyzed != []:
        for model_seq in to_be_analyzed:
            for name, layer in model_seq.named_modules():
                name_list = name.split(".")
                start_point = layers_hier
                for hier_name in name_list:
                    if hier_name not in start_point[1].keys():
                        start_point[1][hier_name] = [
                            name,
                            {},
                            layer
                        ]
                    start_point = start_point[1][hier_name]

        layers, to_be_analyzed = extract_leaves(
            layers_hier, 
            layers, 
            to_be_analyzed
        )

    # for name, layer in layers.items():
    #     print(name)
    #     print(type(layer))

    y = model(x)
    traced_model = torch.jit.trace(model, x)

    for name, module in traced_model.named_modules():
        module.name = name
    traced_model.apply(get_layer_info)
    # print(layers)
    # print(traced_model.code)

# def generate_layers_graph(model, x):

#     layers = {}

#     def get_connections(name):
#         def hook(model, inputs, output):
#             layers[name] = []
#             for input in inputs:
#                 layers[name].append(input.prod_name)
#             output.prod_name = model.name

#         return hook

#     for name, module in model.named_modules():
#         module.name = name
#         module.register_forward_hook(get_connections(name))

#     x.prod_name = "input"
#     y = model(x)

#     for name, values in layers.items():

#         print(name, values)

def generate_activations_stats(model, x):

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name))

    y = model(x)

    for name, values in activation.items():
        print(name)
        print(torch.std_mean(values))

def generate_weights_stats(model):
    print(model)
    for name, parameters in model.named_parameters():
        print(name)
        print(torch.std_mean(parameters))

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

        generate_layers_graph(model, train_features)

        # generate_activations_stats(model, train_features)
        
        # generate_weights_stats(model)

if __name__ == '__main__':
    main()
