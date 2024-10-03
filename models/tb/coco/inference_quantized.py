import onnxruntime
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from utils.datasets import get_dataset
from torch.utils.data import DataLoader
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx_and_make_model
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model
from PIL import Image

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
    array_3d = float_values.reshape((3, 416, 416))

    return array_3d

def inference_coco():
    # onnx_path = "../test/onnx/mobilenet_v2_a8w8b16_71.734.onnx"
    onnx_path = "/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/onnx/yolov3_tiny_a8w8_28.4.onnx"
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    log_name = "yolov3_tiny"

    # hook the intermediate tensors with their name
    os.system(f"mkdir -p tmp/logs/{log_name}")
    train_dataset, eval_dataset, input_shape = get_dataset("coco")

    print(f"Input shape: {input_shape}")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Eval dataset length: {len(eval_dataset)}")
    
    correct = 0
    total = 0
    outputs = 0
    with torch.no_grad():
        
        # for images, labels in eval_dataset:
        #read image from file
            images_path = "/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/coco/hls_lab.jpeg"
            #read image jpeg and convert to numpy array
            img = Image.open(images_path).convert('RGB')
            transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
            img = transform(img)
            image = torch.tensor(img)
            
            #resize image to 416x416
            # np_images = np.resize(np_images, (3, 416, 416))
            
            np_images = np.expand_dims(image.numpy(), axis=0)
            print_image(np_images, "tmp/logs/yolov3_tiny/input_image.txt")
            inferred_model = execute_onnx_and_make_model(inferred_model, {'images': np_images})
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
            #break
    
    # print_weights_tensor('DequantizeLinear_71_out0', inferred_model, ich_ops=8, och_ops=2) 
    # print_bias_tensor('DequantizeLinear_18_out0', inferred_model) 
    print_acts_tensor('/model.0/id/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=1, och_ops=3)
    print_acts_tensor('/model.17/act/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=1, och_ops=16) 
    print_acts_tensor('/model.13/act/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=1, och_ops=16) 
    print_acts_tensor('/model.0/act/act_quant/export_handler/Quant_output_0', inferred_model, ow_ops=16, och_ops=16)
    
if __name__ == '__main__':
    inference_coco()