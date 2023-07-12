#!/usr/bin/env python
# coding: utf-8

# # PLEASE, TO RUN THIS NOTEBOOK INSTALL ONNX, ONNXRUNTIME AND PYTORCH

# In[20]:


import sys
import os

sys.path.append(os.path.abspath("../common"))

import math
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot
import cv2
from datetime import datetime

import pynq
import dac_sdc
from IPython.display import display

team_name = 'overlay'
dac_sdc.BATCH_SIZE = 1
team = dac_sdc.Team(team_name)

import torch


# **Your team directory where you can access your bitstream, notebook, and any other files you submit, is available as `team.team_dir`.**

# In[21]:


bitfile = team.get_bitstream_path()
print(bitfile)
overlay = pynq.Overlay(bitfile)
print("Loaded overlay")
dma = overlay.axi_dma_0
#print(overlay.__dict__)
print("Loading URAM")
dma_uram = overlay.axi_dma_1
uram_vector = np.load("uram.npy")
print(uram_vector.shape)
uram_buffer = pynq.allocate(shape=(uram_vector.shape[0], ), dtype=np.uint8)
uram_buffer[:] = uram_vector[:]
dma_uram.sendchannel.transfer(uram_buffer)

Y_AXIS=64
X_AXIS=64

# In[18]:


#in_buffer = pynq.allocate(shape=(dac_sdc.BATCH_SIZE, 360, 640, 3), dtype=np.uint8, cacheable = 1)
in_buffer = pynq.allocate(shape=(dac_sdc.BATCH_SIZE, Y_AXIS, X_AXIS, 3), dtype=np.uint8, cacheable = 1)
out_buffer = pynq.allocate(shape=(3000), dtype=np.uint16, cacheable = 1)

def dma_transfer():
    dma.recvchannel.transfer(out_buffer)
    dma.sendchannel.transfer(in_buffer)  
    dma_uram.sendchannel.wait()
    print("Shifted uram")
    dma.sendchannel.wait()
    dma.recvchannel.wait()

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def non_max_suppression_custom(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    prediction = xywh2xyxy(np.asarray(prediction))  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    output = nms_custom(prediction.tolist(), iou_thres, 0)

    if len(output) > 0: 
        return [torch.stack(output)]
    else:
        return []

def nms_custom(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    #bboxes = [box for box in bboxes if box[4] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[5] != chosen_box[5]
            or intersection_over_union(
                torch.tensor(chosen_box[:4]),
                torch.tensor(box[:4]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] - x[..., 2] / 2).astype(np.int32)  # top left x
    y[..., 1] = (x[..., 1] - x[..., 3] / 2).astype(np.int32)  # top left y
    y[..., 2] = (x[..., 0] + x[..., 2] / 2).astype(np.int32)  # bottom right x
    y[..., 3] = (x[..., 1] + x[..., 3] / 2).astype(np.int32)  # bottom right y
    return y

def preproc(img):
    # Resize the image (this is part of your runtime)
    #im = np.asarray(cv2.resize(img, (640,360), interpolation = cv2.INTER_LINEAR))
    im = np.asarray(cv2.resize(img, (Y_AXIS,X_AXIS), interpolation = cv2.INTER_LINEAR))
    im = im[:,:,::-1]  # HWC to CHW, BGR to RGB

    im = np.ascontiguousarray(im)  # contiguous
    print(im.shape)
    return im

def postproc(pred, shape, orig_shape):
                
    start = time.time()
    pred = non_max_suppression_custom(pred)
    if len(pred) > 0:
        pred = pred[0]
    end = time.time()
    #print("Image postproc non max suppr: %f" % (end-start))

    object_locations = []

    for _, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:4] = scale_boxes(shape, det[:4], orig_shape).round()
            det[:4] = xyxy2xywh(det[:4])
            det = det.numpy().astype(int)
            det[0] = det[0] - (det[0]/(det[2]/2))
            det[1] = det[1] - (det[1]/(det[3]/2))
            object_locations.append({"type":int(det[5]), "x":int(det[0]), "y":int(det[1]), "width":int(det[2]), "height":int(det[3])})
    return object_locations
        

def my_callback(rgb_imgs):
    
    object_locations_by_image = {}
    
    start = time.time()
    img_list = []
    for i, (img_path, img) in enumerate(rgb_imgs):
        img_list.append(preproc(img))
    in_buffer[:] = np.asarray(img_list)
    end = time.time()
    print("Image preproc: %f" % (end-start))
    start = time.time()
    dma_transfer()
    end = time.time()
    print("Inference: %f" % (end-start))
    start = time.time()

    pred = np.asarray(out_buffer)
    print(dma.read(0x58))
    num_data = int(dma.read(0x58)/16)-1
    if num_data > 0:
        print(pred.shape)
        pred = np.array([pred[i*16:(i+1)*16] for i in range(num_data)])
        print(pred.shape)
        pred = pred[:, :6]

        # Save to dictionary by image filename
        object_locations_by_image[img_path.name] = postproc(pred, (X_AXIS,Y_AXIS), img.shape)
        end = time.time()
        #print(object_locations_by_image[img_path.name])
        #print("Image postproc: %f" % (end-start))
    else:
        print("No object detected")
        object_locations_by_image[img_path.name] = []

    return object_locations_by_image


# In[19]:


team = dac_sdc.Team(team_name)
team.run(my_callback, debug=True)


# In[ ]:


# Remember to free the contiguous memory after usage.
del in_buffer
del out_buffer


# In[ ]:




