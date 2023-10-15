#code to convert onnx qcdc model to qonnx model
import onnx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
#import qcdq_to_qonnx.py
from  qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from utils import ImageNetKaggle



def main():
    parser = argparse.ArgumentParser(description='Convert onnx model to qonnx model')
    parser.add_argument('--onnx_model', type=str, default='/home/teodoro/nn2fpga/py/utils/model_vit.onnx', help='onnx model path')
    parser.add_argument('--qonnx_model', type=str, default='modelq.onnx', help='qonnx model path')
    args = parser.parse_args()

    #imoport onnx model
    #qonnx conversion
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model
    import qonnx
    onnx_model = ModelWrapper(args.onnx_model)
    #onnx_model = onnx.load(args.onnx_model)
    onnx_model = cleanup_model(onnx_model) 
    qonnx_model = onnx_model.transform(QCDQToQuant())
    #from model wrapper to onnx model
    qonnx_model = qonnx_model.model
    onnx.save(qonnx_model,args.onnx_model)
    print("QONNX model converted")
    exit()
    #save qonnx model
    # val_dataset = ImageNetKaggle(cfg.data, "val", val_transform) 
    # eval_loader = dataloader = DataLoader(
    #             val_dataset,
    #             batch_size=cfg.eval_batch_size, # may need to reduce this depending on your GPU 
    #             num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
    #             shuffle=False,
    #             drop_last=False,
    #             pin_memory=True
    #         )
    
    onnx.save(qonnx_model,args.qonnx_model)
    print("QONNX model converted")    
    #save qonnx model
    #onnx.save(onnx_model, args.qonnx_model)
    # from brevitas.export.onnx.qonnx.manager import QONNXManager
    # QONNXManager.export(onnx_model, input_shape=(1, 3, 224, 224), export_path=args.qonnx_model) 
    print("QONNX model exported to ",args.qonnx_model)
    
if __name__ == "__main__":
    main()