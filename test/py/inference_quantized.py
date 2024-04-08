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

def inference_imagenet():
    onnx_path = "../test/onnx/mobilenet_v2_a8w8b32.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    log_name = "mobilenet_v2"

    # hook the intermediate tensors with their name
    os.system(f"mkdir -p tmp/logs/{log_name}")
    train_dataset, eval_dataset, input_shape = get_dataset("imagenet")

    print(f"Input shape: {input_shape}")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Eval dataset length: {len(eval_dataset)}")
    
    correct = 0
    total = 0
    outputs = 0
    with torch.no_grad():
        for images, labels in train_dataset:
            np_images = images.numpy()
            np_images = np.expand_dims(np_images, axis=0)

            inferred_model = execute_onnx_and_make_model(inferred_model, {'global_in': np_images})
            # outputs = execute_onnx(inferred_model, {'global_in': images.numpy()})
            # outputs = outputs['global_out']
            # outputs = np.squeeze(outputs)
            # predictions = np.argmax(outputs)
            # print(outputs)
            # print(f"Expected: {labels[0].item()}, computed: {predictions}")
            # total += 1
            # for i in outputs:
            #     print(i)
            # correct += (predictions == labels[0].item())
            # if (total % 10 == 0):
            #     print(f"Images: {total}\tAccuracy on CIFAR-10: {100 * correct / total:.2f}%")
            break
    
    # print_weights_tensor('DequantizeLinear_71_out0', inferred_model, ich_ops=8, och_ops=2) 
    # print_bias_tensor('DequantizeLinear_18_out0', inferred_model) 
    print_acts_tensor('DequantizeLinear_106_out0', inferred_model, ow_ops=1, och_ops=3) 

def inference_cifar10():
    onnx_path = "../test/onnx/resnet8.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    log_name = "resnet8"

    # hook the intermediate tensors with their name
    os.system(f"mkdir -p tmp/logs/{log_name}")

    train_dataset, eval_dataset, input_shape = get_dataset("cifar10_4bit")
    
    correct = 0
    total = 0
    outputs = 0
    with torch.no_grad():
        for images, labels in eval_dataset:
            np_images = images.numpy()
            np_images = np.expand_dims(np_images, axis=0)

            inferred_model = execute_onnx_and_make_model(inferred_model, {'inp.1': np_images})
            break
    
    print_acts_tensor('DequantizeLinear_25_out0', inferred_model, ow_ops=4, och_ops=4) 

def inference_cifar10_4bit():
    onnx_path = "../test/onnx/resnet8_a4w4b32.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    log_name = "resnet8_4bit"

    # hook the intermediate tensors with their name
    os.system(f"mkdir -p tmp/logs/{log_name}")

    train_dataset, eval_dataset, input_shape = get_dataset("cifar10_4bit")
    
    correct = 0
    total = 0
    outputs = 0
    with torch.no_grad():
        for images, labels in eval_dataset:
            np_images = images.numpy()
            np_images = np.expand_dims(np_images, axis=0)

            inferred_model = execute_onnx_and_make_model(inferred_model, {'global_in': np_images})
            break
    
    print_acts_tensor('DequantizeLinear_25_out0', inferred_model, ow_ops=4, och_ops=4) 

if __name__ == '__main__':
    inference_cifar10()