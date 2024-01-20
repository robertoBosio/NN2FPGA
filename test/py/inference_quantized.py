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
from utils.datasets import ImageNet

def float_to_fixed_point(value, num_bits_fractional=8, act_width=8):

    # Calculate the scaling factor based on the number of fractional bits
    scale_factor = 2 ** act_width

    # Scale the floating-point value and round to the nearest integer
    scaled_value = round(value * scale_factor)

    # Clip the scaled value to fit within the 8-bit range
    # clipped_value = np.clip(scaled_value, 0, scale_factor - 1)

    # Convert the clipped value to 8-bit binary representation
    binary_representation = np.float32(float(scaled_value) / 2**num_bits_fractional)

    return binary_representation

def print_acts_tensor(tensor_name, model, ow_ops=1, och_ops=1):
    """Print the tensor with name tensor_name and replicate the output of the framework.

    :param tensor_name: name of the tensor to print
    :param model: model to use
    :param ow_ops: ow_ops_out parameter of the convolutional layer
    :param och_ops: ops parameter of the convolutional layer
    """
    tensor = model.get_initializer(tensor_name)
    
    # In case of output results
    # Translate this with numpy

    if tensor.ndim == 2:
        tensor = tensor[:, :, np.newaxis, np.newaxis]
        ow_ops = 1
        och_ops = 1
    
    # Transform each "/" in the tensor name into "_" to avoid creating subdirectories
    tensor_name = tensor_name.replace("/", "_")
    with open(f"tmp/logs/{tensor_name}_qonnx_acts.txt", 'w') as f:
        for rows in range(tensor.shape[2]):
            for cols in range(0, tensor.shape[3], ow_ops):
                for channels in range(0, tensor.shape[1], och_ops):
                    for ow in range(ow_ops):
                        for och in range(och_ops):
                            value = tensor[0][channels + och][rows][cols + ow].item()

                            if str(value).split(".")[0] == "-0" and value == 0.0:
                                value = -value 
                            
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
    with open(f"tmp/logs/{tensor_name}_qonnx_weights.txt", 'w') as f:
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
    with open(f"tmp/logs/{tensor_name}_qonnx_bias.txt", 'w') as f:
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

def print_image(image, file_name):

    flattened_arr = image.flatten()
    
    with open(file_name, 'w') as file:
        for value in flattened_arr:
            file.write(str(value) + "\n")

def read_floats_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read all lines from the file and convert each line to a float
        floats = [np.float32(line.strip()) for line in file]

    return np.array(floats)

def image_from_file(file_path):

    # Read float values from the file
    float_values = read_floats_from_file(file_path)

    # Reshape the float values into a numpy array with the desired shape
    array_3d = float_values.reshape((3, 224, 224))

    return array_3d

if __name__ == '__main__':
    # Replace 'path_to_quantized_resnet8.onnx' with the path to your quantized ResNet-8 ONNX model.
    # onnx_path = "../test/onnx/resnet8.onnx"
    onnx_path = "../test/onnx/mobilenet_v2_a8w8b32.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    # log_name = "resnet8"
    log_name = "mobilenet_v2"

    # hook the intermediate tensors with their name
    os.system(f"mkdir -p tmp/logs/{log_name}")

    # CIFAR-10 dataset loading
    # transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])
    IMG_SIZE = 256
    # BASE_DIR = "/home-ssd/datasets/Imagenet/"
    BASE_DIR = "/tools/datasets/Imagenet/"
    transforms_sel = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    args = {
        'train': True,
        'transform': transforms_sel,
        'root': BASE_DIR,
        'sample_size': None
    }
    # dataset = ImageNet
    # dataset = dataset(**args)
    # train_dataset, eval_dataset, input_shape = get_dataset("imagenet")

    # imagenet_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
    #                                            num_workers=1)
    # cifar10_dataset = torchvision.datasets.CIFAR10(root="/home-ssd/datasets/cifar10/", train=False, download=False, transform=transform)
    # cifar10_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=False)
    
    # correct = 0
    # total = 0
    # outputs = 0
    # with torch.no_grad():
    #     for images, labels in imagenet_loader:
    #         print(images.shape)
    #         print_image(images.numpy(), "tmp/logs/flatten_image.txt")
    #         func = np.vectorize(float_to_fixed_point)
    #         new_input = func(images.numpy())

    #         # print the tensor flattened by channel one value per line in a file
    #         # with open('conv1_act_tensor_csim.txt', 'r') as f:
    #         #     for r in range(32):
    #         #         for g in range(32):
    #         #             for b in range(3):
    #         #                 # read value from file and convert to float
    #         #                 value = float(f.readline().split()[0])
    #         #                 images[0][b][r][g] = value
    #         #                 # f.write(str(images[0][b][r][g].item()) + "\n")
            
    #         inferred_model = execute_onnx_and_make_model(inferred_model, {'global_in': images.numpy()})
    #         # outputs = execute_onnx(inferred_model, {'global_in': images.numpy()})
    #         # outputs = outputs['global_out']
    #         # outputs = np.squeeze(outputs)
    #         # predictions = np.argmax(outputs)
    #         # print(outputs)
    #         # print(f"Expected: {labels[0].item()}, computed: {predictions}")
    #         # total += 1
    #         # for i in outputs:
    #         #     print(i)
    #         # correct += (predictions == labels[0].item())
    #         # if (total % 10 == 0):
    #         #     print(f"Images: {total}\tAccuracy on CIFAR-10: {100 * correct / total:.2f}%")
    #         break

    images = image_from_file("/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/image_preprocessed_opencv.txt")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    func = np.vectorize(float_to_fixed_point)
    images = func(images, 8, 8)
    with open(f"tmp/logs/image_premean.txt", 'w') as f:
        for r in range(224):
            for c in range(224):
                for ch in range(3):
                    print(images[ch][r][c], file=f) 

    img_transposed = np.transpose(images, (1, 2, 0))
    normalized_arr = (img_transposed - mean) / std
    images = np.transpose(normalized_arr, (2, 0, 1))

    images = np.float32(images)
    images = np.expand_dims(images, 0)

    inferred_model = execute_onnx_and_make_model(inferred_model, {'global_in': images})
    

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
    
    # print_weights_tensor('DequantizeLinear_71_out0', inferred_model, ich_ops=8, och_ops=2) 
    # print_bias_tensor('DequantizeLinear_18_out0', inferred_model) 
    # print_acts_tensor('DequantizeLinear_128_out0', inferred_model, ow_ops=4, och_ops=2) 
    # print_acts_tensor('global_out', inferred_model, ow_ops=1, och_ops=1) 
    # BFS(inferred_model)