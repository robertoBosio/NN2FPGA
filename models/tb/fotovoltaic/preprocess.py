#!/usr/bin/env python3

import os
import numpy as np
import torch
import cv2
import glob
from pathlib import Path
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation import infer_shapes
from qonnx.transformation.infer_datatypes import InferDataTypes

IMG_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class LoadImages:
    def __init__(self, root, splits=['train', 'val'], img_size=640, stride=32, auto=True, transforms=None):
        files = []
        for split in splits:
            split_dir = os.path.join(root, 'images', split)
            print(f"[DEBUG] Scanning directory: {split_dir}")
            if os.path.isdir(split_dir):
                found = sorted(glob.glob(os.path.join(split_dir, '*.*')))
                print(f"[DEBUG] Found {len(found)} images in split '{split}'")
                files.extend(found)
            else:
                print(f"[WARNING] Split directory not found: {split_dir}")
        self.files = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        print(f"[DEBUG] Total images loaded from splits {splits}: {len(self.files)}")
        assert len(self.files) > 0, f'No images found in {root} for splits {splits}.'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.count = 0
        self.splits = splits
        self.root = root

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.files):
            raise StopIteration
        path = self.files[self.count]
        im0 = cv2.imread(path)
        assert im0 is not None, f'Image Not Found: {path}'
        if self.transforms:
            im = self.transforms(im0)
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
            im = np.ascontiguousarray(im)
        # Ricava lo split corrente dal path
        split = None
        for s in self.splits:
            if f'/images/{s}/' in path or f'\\images\\{s}\\' in path:
                split = s
                break
        if split is None:
            split = self.splits[0]
        print(f"[DEBUG] Processing image: {path} (split: {split})")
        self.count += 1
        return path, im, im0, split, f'image {self.count}/{len(self.files)} {path}'


def process_images(n_images, onnx_path, dataset_root, splits=["val"], size=None):
    print(f"[INFO] Processing {n_images} images with ONNX model: {onnx_path} for splits: {splits}")

    # Load ONNX model
    model = ModelWrapper(onnx_path)
    model = model.transform(infer_shapes.InferShapes())
    model = model.transform(InferDataTypes())

    inp_name = model.model.graph.input[0].name
    _, ch, H, W = model.get_tensor_shape(inp_name)

    if size:
        if len(size) == 1:
            H = W = size[0]
        elif len(size) == 2:
            H, W = size
        else:
            raise ValueError("Invalid --size format")
    dataset = LoadImages(dataset_root, splits=splits, img_size=(H, W), stride=32, auto=False)

    # Cleanup output files
    for fn in [
        "/tmp/images_preprocessed.bin",
        "/tmp/labels_preprocessed.bin",
        "/tmp/results_preprocessed_0.bin",
        "/tmp/results_preprocessed_1.bin",
        "/tmp/results_preprocessed_2.bin"
    ]:
        if os.path.exists(fn):
            os.remove(fn)
            print(f"[DEBUG] Removed existing file: {fn}")

    count = 0
    with torch.no_grad(), \
         open("/tmp/images_preprocessed.bin", "ab") as f_img, \
         open("/tmp/labels_preprocessed.bin", "ab") as f_lbl, \
         open("/tmp/results_preprocessed_0.bin", "ab") as f_out0, \
         open("/tmp/results_preprocessed_1.bin", "ab") as f_out1, \
         open("/tmp/results_preprocessed_2.bin", "ab") as f_out2:

        for path, im, im0, split, _ in dataset:
            if count >= n_images:
                break

            print(f"[DEBUG] im (CHW, uint8) shape: {im.shape}, dtype: {im.dtype}, min: {im.min()}, max: {im.max()}")
            print(f"[DEBUG] Image path: {path} | Split: {split}")
            # Converti immagine da C,H,W a H,W,C e normalizza [0,1]
            im_hwc = np.transpose(im, (1, 2, 0)).astype(np.float32) / 255.0
            print(f"[DEBUG] im_hwc (HWC, float32, [0,1]) shape: {im_hwc.shape}, dtype: {im_hwc.dtype}, min: {im_hwc.min()}, max: {im_hwc.max()}")
            f_img.write(im_hwc.flatten().tobytes())
            print(f"[DEBUG] Salvato: /tmp/images_preprocessed.bin, shape salvata: {im_hwc.shape}, dtype: float32")

            # Prepara immagine per ONNX (CHW, float32, normalizzata [0,1])
            im_chw = np.transpose(im_hwc, (2, 0, 1))
            print(f"[DEBUG] im_chw (CHW, float32, [0,1]) shape: {im_chw.shape}, dtype: {im_chw.dtype}, min: {im_chw.min()}, max: {im_chw.max()}")
            im_onnx = np.expand_dims(im_chw, axis=0)  # shape (1, C, H, W)
            print(f"[DEBUG] im_onnx (1,CHW, float32, [0,1]) shape: {im_onnx.shape}, dtype: {im_onnx.dtype}, min: {im_onnx.min()}, max: {im_onnx.max()}")
            #print first 3 values of im_onnx
            print(f"[DEBUG] First 3 values of im_onnx: {im_onnx.flatten()[:3]}")
            # Process labels (relative allo split corretto)
            base = os.path.splitext(os.path.basename(path))[0]
            label_path = os.path.join(dataset_root, "labels", split, base + ".txt")
            print(f"[DEBUG] Looking for label: {label_path}")

            if os.path.exists(label_path):
                labels = np.loadtxt(label_path, ndmin=2)
                print(f"[DEBUG] Loaded label shape: {labels.shape}")
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
                print(f"[DEBUG] No label found, using empty array.")

            if labels.shape[0]:
                h0, w0 = im0.shape[:2]
                labels[:, 1] *= w0
                labels[:, 2] *= h0
                labels[:, 3] *= w0
                labels[:, 4] *= h0

                _, (gain_w, gain_h), (dw, dh) = letterbox(im0, new_shape=(H, W), auto=False, scaleFill=False)
                labels[:, 1] = labels[:, 1] * gain_w + dw
                labels[:, 2] = labels[:, 2] * gain_h + dh
                labels[:, 3] *= gain_w
                labels[:, 4] *= gain_h
                labels[:, 1:] /= [W, H, W, H]
                print(f"[DEBUG] Label after transform: {labels}")

            f_lbl.write(labels.astype(np.uint32).tobytes())
            print(f"[DEBUG] Salvato: /tmp/labels_preprocessed.bin, shape salvata: {labels.shape}, dtype: uint32")
            #print first 3 values of im_onnx
            print(f"[DEBUG] First 3 values of im_onnx: {im_onnx.flatten()[:3]}")
            # ONNX Inference (input: im_onnx, shape (1, C, H, W))
            out_dict = execute_onnx(model, {inp_name: im_onnx})
            out0 = out_dict["global_out"]
            out1 = out_dict["global_out_1"]
            out2 = out_dict["global_out_2"]

            for i, out in enumerate([out0, out1, out2]):
                out = np.squeeze(out)
                assert out.ndim == 3, f"Unexpected output shape: {out.shape}"
                out = np.transpose(out, (1, 2, 0))  # CHW → HWC
                out = out.astype(np.float32)
                [f_out0, f_out1, f_out2][i].write(out.flatten().tobytes())
                print(f"[DEBUG] Salvato: /tmp/results_preprocessed_{i}.bin, shape salvata: {out.shape}, dtype: float32")

            count += 1

    print("[INFO] Processing completed successfully.")


if __name__ == "__main__":
    # Solo chiamata posizionale: python3 preprocess.py <n_images> <onnx_path> <dataset_name>
    if len(sys.argv) != 4:
        print(sys.argv)
        print("Usage: python3 preprocess.py <n_images> <onnx_path> <dataset_name>")
        sys.exit(1)
    n_images = int(sys.argv[1])
    onnx_path = sys.argv[2]
    dataset_name = sys.argv[3]
    dataset_name = "fotovoltaic_dataset"
    base_data_dir = "/home-ssd/datasets/" #Uncomment for smaug
    dataset_root = os.path.join(base_data_dir, dataset_name.upper()) if os.path.isdir(os.path.join(base_data_dir, dataset_name.upper())) else os.path.join(base_data_dir, dataset_name.lower())
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Dataset directory not found: {dataset_root}")
        sys.exit(1)
    print(f"[INFO] Chiamata posizionale: n_images={n_images}, onnx_path={onnx_path}, dataset_root={dataset_root}")
    # splits di default: ["train", "val"]
    process_images(n_images, onnx_path, dataset_root, splits=["val"], size=[640])
