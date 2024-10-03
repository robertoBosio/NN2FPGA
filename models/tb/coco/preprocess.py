import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import numpy as np
import os
import qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx_and_make_model
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model
from utils.datasets import get_dataset
from PIL import Image
    
def process_image(n_images, onnx_path , dataset):
    
    onnx_model = ModelWrapper(onnx_path)
    cleanup_model(onnx_model)
    inferred_model = onnx_model.transform(infer_shapes.InferShapes())
    inferred_model = inferred_model.transform(InferDataTypes())
    
    # multiple_output = True
    # shape_info = qonnx.shape_inference.infer_shapes(inferred_model)
    # if multiple_output:
    #     #add outptut to the model
    #     new_outputs_name = ['/model.13/act/act_quant/export_handler/Quant_output_0', '/model.17/act/act_quant/export_handler/Quant_output_0']
    #     new_protos = []
    #     for output_name in new_outputs_name:
    #         new_protos.append(inferred_model.get_tensor_proto(output_name))
    #     inferred_model.add_outputs(new_protos)
        
         
    batch_size = 1
    train_dataset, eval_dataset, input_shape = get_dataset(dataset='coco')

    counter = 0
    # remove /tmp/images_preprocessed.bin if it exists
    try:
        os.remove("/tmp/images_preprocessed.bin")
    except FileNotFoundError:
        pass

    # remove /tmp/labels_preprocessed.bin if it exists
    try:
        os.remove("/tmp/labels_preprocessed.bin")
    except FileNotFoundError:
        pass

    # remove /tmp/result_preprocessed.bin if it exists
    try:
        os.remove("/tmp/results_preprocessed.bin")
    except FileNotFoundError:
        pass

    with torch.no_grad():
        with open("/tmp/images_preprocessed.bin", "ab") as f_image, \
             open("/tmp/labels_preprocessed.bin", "ab") as f_labels, \
             open("/tmp/results_preprocessed.bin", "ab") as f_res:
            
            for images, labels in eval_dataset:
                ###### file img
                images_path = "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/models/tb/coco/hls_lab.jpeg"
                # image = torchvision.io.read_image(images_path)
                img = Image.open(images_path).convert('RGB')
                #resize the image to 416x416
                transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
                image = transform(img)
                
                f_image.write(np.asarray(torch.permute(image, (1, 2, 0))).flatten().astype(np.float32).tobytes())
                #f_labels.write(np.asarray(labels).astype(np.uint32).tobytes())


                np_images = np.expand_dims(image.numpy(), axis=0)
                
                outputs = execute_onnx(inferred_model, {'images': np_images})
                outputs = outputs['output']
                outputs = np.squeeze(outputs)
                f_res.write(outputs.astype(np.float32).tobytes())

                counter += 1
                if counter == int(n_images):
                    break

if __name__ == "__main__":
    n_images = sys.argv[1]
    onnx_path = sys.argv[2]
    dataset = sys.argv[3]
    n_images = 1
    # onnx_path = 'runs/train/yolov3_tiny_quant3/weights/best.qonnx.onnx'
    # onnx_path = '/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/onnx/yolov3_tiny_a8w8_28.4.onnx'
    # dataset = 'coco'
    process_image(n_images, onnx_path, dataset)
