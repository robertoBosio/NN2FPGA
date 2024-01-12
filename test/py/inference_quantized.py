import onnxruntime
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from utils.datasets import get_dataset
from torch.utils.data import DataLoader
from utils.preprocess import cifar_transform
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx_and_make_model
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model

def float_to_fixed_point(value):
    # Define the number of bits for integer and fractional parts
    num_bits_integer = 2
    num_bits_fractional = 8

    # Calculate the scaling factor based on the number of fractional bits
    scale_factor = 2 ** num_bits_fractional

    # Scale the floating-point value and round to the nearest integer
    scaled_value = round(value * scale_factor)

    # Clip the scaled value to fit within the 8-bit range
    clipped_value = np.clip(scaled_value, 0, scale_factor - 1)

    # Convert the clipped value to 8-bit binary representation
    binary_representation = float(clipped_value) / 2**num_bits_fractional

    return binary_representation

def print_acts_tensor(tensor_name, model, ow_ops=1, och_ops=1):
    """Print the tensor with name tensor_name and replicate the output of the framework.

    :param tensor_name: name of the tensor to print
    :param model: model to use
    :param ow_ops: ow_ops_out parameter of the convolutional layer
    :param och_ops: ops parameter of the convolutional layer
    """
    tensor = model.get_initializer(tensor_name)
    
    # Transform each "/" in the tensor name into "_" to avoid creating subdirectories
    tensor_name = tensor_name.replace("/", "_")
    with open(f"tmp/logs/{tensor_name}_acts.txt", 'w') as f:
        for rows in range(tensor.shape[2]):
            for cols in range(0, tensor.shape[3], ow_ops):
                for channels in range(0, tensor.shape[1], och_ops):
                    for ow in range(ow_ops):
                        for och in range(och_ops):
                            value = tensor[0][channels + och][rows][cols + ow].item()
                            
                            # if the string of the value is "*.0", then print "*" only
                            if (str(value).split(".")[1] == "0"):
                                print(f'[{channels + och},{rows},{cols + ow}] {str(value).split(".")[0]}', file=f)
                            else:
                                print(f"[{channels + och},{rows},{cols + ow}] {value}", file=f)

def print_weights_tensor(tensor_name, model, ich_ops=1, och_ops=1):
    """Print the tensor with name tensor_name and replicate the output of the framework.

    :param tensor_name: name of the tensor to print
    :param model: model to use
    :param ow_ops: ow_ops_out parameter of the convolutional layer
    :param och_ops: ops parameter of the convolutional layer
    """
    tensor = model.get_initializer(tensor_name)
    
    # Transform each "/" in the tensor name into "_" to avoid creating subdirectories
    tensor_name = tensor_name.replace("/", "_")
    with open(f"tmp/logs/{tensor_name}_weights.txt", 'w') as f:
        for channels in range(0, tensor.shape[1], ich_ops):
            for och in range(0, tensor.shape[0], och_ops):
                for rows in range(tensor.shape[2] - 1, -1, -1):
                    for cols in range(tensor.shape[3] - 1, -1, -1):
                        for och_in in range(och_ops):
                            for ich_in in range(ich_ops):
                                value = tensor[och + och_in][channels + ich_in][rows][cols].item()
                                
                                # if the string of the value is "*.0", then print "*" only
                                if (str(value).split(".")[1] == "0"):
                                    print(f'[{och + och_in},{channels + ich_in},{rows},{cols}] {str(value).split(".")[0]}', file=f)
                                else:
                                    print(f"[{och + och_in},{channels + ich_in},{rows},{cols}] {value}", file=f)

def print_bias_tensor(tensor_name, model):
    """Print the tensor with name tensor_name and replicate the output of the framework.

    :param tensor_name: name of the tensor to print
    :param model: model to use
    :param ow_ops: ow_ops_out parameter of the convolutional layer
    :param och_ops: ops parameter of the convolutional layer
    """
    tensor = model.get_initializer(tensor_name)
    
    # Transform each "/" in the tensor name into "_" to avoid creating subdirectories
    tensor_name = tensor_name.replace("/", "_")
    with open(f"tmp/logs/{tensor_name}_bias.txt", 'w') as f:
        for channels in range(tensor.shape[0]):
            value = tensor[channels].item()
            
            # if the string of the value is "*.0", then print "*" only
            if (str(value).split(".")[1] == "0"):
                print(f'[{channels}] {str(value).split(".")[0]}', file=f)
            else:
                print(f"[{channels}] {value}", file=f)

def BFS(model):
    """Breadth-first search to find the node with name node_name."""
    
    queue = [model.graph.node[0]]
    while len(queue) != 0:
        node = queue.pop(0)
        print(node.name)
        successors = model.find_direct_successors(node)
        if successors is not None:
            for successor in successors:
                queue.insert(0, successor)
    return None

if __name__ == '__main__':
    # Replace 'path_to_quantized_resnet8.onnx' with the path to your quantized ResNet-8 ONNX model.
    onnx_path = "../test/onnx/resnet8.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    log_name = "resnet8"

    # hook the intermediate tensors with their name
    os.system(f"mkdir -p tmp/logs/{log_name}")
    os.system(f"rm -rf tmp/logs/{log_name}/feature.txt")

    # CIFAR-10 dataset loading
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    cifar10_dataset = torchvision.datasets.CIFAR10(root="/home-ssd/datasets/cifar10/", train=False, download=False, transform=transform)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in cifar10_loader:
            func = np.vectorize(float_to_fixed_point)
            new_input = func(images.numpy())

            # print the tensor flattened by channel one value per line in a file
            # with open('conv1_act_tensor_csim.txt', 'r') as f:
            #     for r in range(32):
            #         for g in range(32):
            #             for b in range(3):
            #                 # read value from file and convert to float
            #                 value = float(f.readline().split()[0])
            #                 images[0][b][r][g] = value
            #                 # f.write(str(images[0][b][r][g].item()) + "\n")
            
            inferred_model = execute_onnx_and_make_model(inferred_model, {'inp.1': new_input})
            # outputs = execute_onnx(inferred_model, {'inp.1': images.numpy()})
            # outputs = outputs['132']
            # outputs = np.squeeze(outputs)
            # predictions = np.argmax(outputs)
            # print(outputs)
            # # print(f"Expected: {labels}, computed: {predictions}")
            # total += 1
            # for i in outputs:
            #     print(i)
            # correct += (predictions == labels.item())
            # if (total % 100 == 0):
            #     print(f"Images: {total}\tAccuracy on CIFAR-10: {100 * correct / total:.2f}%")
            break

    # for tensor in inferred_model.get_all_tensor_names():
    #     print(tensor)
    
    # for node in inferred_model.graph.node:
    #     # if node.name == "/conv1/output_quant/export_handler/Quant":
    #     print(f"Node Name: {node.name}")
    #     print(f"Op Type: {node.op_type}")
    #     print("Input Names:")
    #     for input_name in node.input:
    #         print(f"  {input_name}")
    #     print("Output Names:")
    #     for output_name in node.output:
    #         print(f"  {output_name}")
    #     print("\nNode Attributes:")
    #     for attr in node.attribute:
    #         print(f"  {attr.name}: {attr}")

    # Get initializer of the first convolutional layer
    # print_tensor('/relu/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=4, och_ops=4) 
    # print_tensor('/layer1/layer1.0/conv1/input_quant/export_handler/Quant_output_0', inferred_model, 2, 4) 
    # print_tensor('/conv1/Conv_output_0', inferred_model, 2, 4) 
    # print_tensor('/conv1/output_quant/export_handler/Quant_output_0', inferred_model, 2, 4) 
    # print_tensor('/layer1/layer1.0/relu/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=4, och_ops=8) 
    print_acts_tensor('/layer1/layer1.0/conv1/input_quant/export_handler/Quant_output_0', inferred_model, ow_ops=4, och_ops=4) 
    print_weights_tensor('/layer1/layer1.0/conv1/weight_quant/export_handler/Quant_output_0', inferred_model, ich_ops=4, och_ops=2) 
    print_bias_tensor('/layer1/layer1.0/conv1/bias_quant/export_handler/Quant_output_0', inferred_model) 
    print_acts_tensor('/layer1/layer1.0/relu/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=4, och_ops=8) 
    # BFS(inferred_model)