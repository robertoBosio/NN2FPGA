import numpy as np
import torch
import sys
import torchvision
import os

def read_output(path, shape):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)

def read_output_hwc_to_chw(path, shape_hwc, dtype = np.int8):
    data = np.fromfile(path, dtype=dtype)
    data = data.reshape(shape_hwc)
    data = np.transpose(data, (0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h, w)
    return data

def make_grid(nx, ny, device, dtype):
    yv, xv = torch.meshgrid(torch.arange(ny, device=device, dtype=dtype), torch.arange(nx, device=device, dtype=dtype), indexing='ij')
    grid = torch.stack((xv, yv), 2).unsqueeze(0).unsqueeze(0)
    return grid

def postprocessing(preds):
    grid = [torch.empty(0) for _ in range(3)]
    z = []
    anchor_grid = [torch.empty(0) for _ in range(3)]
    stride = torch.tensor([8., 16., 32.], device=preds[0].device)
    anchors = torch.tensor([
        [1.25000, 1.62500, 2.00000, 3.75000, 4.12500, 2.87500],
        [1.87500, 3.81250, 3.87500, 2.81250, 3.68750, 7.43750],
        [3.62500, 2.81250, 4.87500, 6.18750, 11.65625, 10.18750]
    ], device=preds[0].device).float().view(3, -1, 2)
    for i in range(3):
        bs, _, ny, nx = preds[i].shape
        preds[i] = preds[i].view(bs, 3, 6, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if grid[i].shape[2:4] != preds[i].shape[2:4]:
            grid[i], anchor_grid[i] = make_grid(nx, ny, i, anchors, stride)
        xy, wh, conf = preds[i].sigmoid().split((2, 2, 2), dim=4)
        xy = (xy * 2 + grid[i]) * stride[i]
        wh = (wh * 2) ** 2 * anchor_grid[i]
        y = torch.cat((xy, wh, conf), dim=4)
        z.append(y.view(bs, -1, 6))
    return torch.cat(z, dim=1)

def make_grid(nx=20, ny=20, i=0, anchors=None, stride=None):
    d = anchors[i].device
    t = anchors[i].dtype
    shape = 1, 3, ny, nx, 2
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing='ij') if torch.__version__ >= '1.10.0' else torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
    anchor_grid = (anchors[i] * stride[i]).view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        c = x[:, 5:6] * (0 if agnostic else 4096)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i][:max_det]
    return output

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., 0].clamp_(0, img0_shape[1])
    boxes[..., 1].clamp_(0, img0_shape[0])
    boxes[..., 2].clamp_(0, img0_shape[1])
    boxes[..., 3].clamp_(0, img0_shape[0])
    return boxes

def postprocess_from_files(out0_path, out1_path, out2_path):
    n_images = os.path.getsize(out0_path) // (18 * 80 * 80)
    img1_shape = (640, 640)
    for i in range(n_images):
        out0 = read_output_hwc_to_chw(out0_path, (n_images, 80, 80, 18))[i] * 0.125
        out1 = read_output_hwc_to_chw(out1_path, (n_images, 40, 40, 18))[i] * 0.125
        out2 = read_output_hwc_to_chw(out2_path, (n_images, 20, 20, 18))[i] * 0.125
        t0 = torch.from_numpy(out0)[None, ...]
        t1 = torch.from_numpy(out1)[None, ...]
        t2 = torch.from_numpy(out2)[None, ...]
        preds_list = [t0, t1, t2]
        preds = postprocessing(preds_list)
        dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
        dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
        print(f"\n[Image {i}] Detections:")
        if dets.shape[0] == 0:
            print("No detections.")
        else:
            img0_shape = (640, 640)
            ratio, pad = (1.0, 1.0), (0.0, 0.0)
            dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape[::-1], (ratio, pad)).round()
            for det in dets:
                x1, y1, x2, y2, conf, cls = det.tolist()
                w, h = x2 - x1, y2 - y1
                if w >= 1 and h >= 1 and 0 <= x1 < img0_shape[0] and 0 <= y1 < img0_shape[1] and 0 < x2 <= img0_shape[0] and 0 < y2 <= img0_shape[1]:
                    print(f"Class: {int(cls)}, Confidence: {conf:.2f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

def postprocess_from_preproc_files():
    # Determina n_images dai file di preprocess
    out0_path = "/tmp/results_preprocessed_0.bin"
    out1_path = "/tmp/results_preprocessed_1.bin"
    out2_path = "/tmp/results_preprocessed_2.bin"
    n_images = os.path.getsize(out0_path) // (80 * 80 * 18 * 4)
    out0 = read_output_hwc_to_chw(out0_path, (n_images, 80, 80, 18), np.float32)
    out1 = read_output_hwc_to_chw(out1_path, (n_images, 40, 40, 18), np.float32)
    out2 = read_output_hwc_to_chw(out2_path, (n_images, 20, 20, 18), np.float32)
    img1_shape = (640, 640)
    for i in range(n_images):
        t0 = torch.from_numpy(out0[i])[None, ...]
        t1 = torch.from_numpy(out1[i])[None, ...]
        t2 = torch.from_numpy(out2[i])[None, ...]
        preds_list = [t0, t1, t2]
        preds = postprocessing(preds_list)
        dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
        dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
        print(f"\n[Image {i}] Detections:")
        if dets.shape[0] == 0:
            print("No detections.")
        else:
            img0_shape = (640, 640)
            ratio, pad = (1.0, 1.0), (0.0, 0.0)
            dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape[::-1], (ratio, pad)).round()
            for det in dets:
                x1, y1, x2, y2, conf, cls = det.tolist()
                w, h = x2 - x1, y2 - y1
                if w >= 1 and h >= 1 and 0 <= x1 < img0_shape[0] and 0 <= y1 < img0_shape[1] and 0 < x2 <= img0_shape[0] and 0 < y2 <= img0_shape[1]:
                    print(f"Class: {int(cls)}, Confidence: {conf:.2f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

if __name__ == "__main__":
    # Se chiamato senza argomenti o con --preproc-test, usa direttamente i file tmp di preprocess
    if (len(sys.argv) == 1) or (len(sys.argv) == 2 and sys.argv[1] == "--preproc-test"):
        print("[INFO] Test: uso direttamente i file /tmp/results_preprocessed_*.bin prodotti da preprocess.py (con conversione HWC->CHW)")
        postprocess_from_preproc_files()
    elif len(sys.argv) == 4 and all(os.path.isfile(p) for p in sys.argv[1:4]):
        postprocess_from_files(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python3 postFix.py <out0.bin> <out1.bin> <out2.bin>  OR  python3 postFix.py --preproc-test")
        sys.exit(1)
