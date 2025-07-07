import numpy as np
import torch
import sys
import torchvision  # Import globale per NMS
import cv2
from pathlib import Path
import os

# --- Funzione process_batch da YOLOv5 (numpy version) ---
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
        iouv (array[10]): IoU thresholds
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct
    iou = box_iou(torch.tensor(labels[:, 1:]), torch.tensor(detections[:, :4])).numpy()
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = np.stack(x, 1)
            if x[0].shape[0] > 1:
                # If multiple matches, use best iou per detection
                iou_vals = iou[x[0], x[1]]
                order = iou_vals.argsort()[::-1]
                matches = matches[order]
                # Remove duplicate detections
                _, unique_idx = np.unique(matches[:, 1], return_index=True)
                matches = matches[unique_idx]
            correct[matches[:, 1], i] = True
    return correct

def read_output(path, shape):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)

# --- Funzione box_iou standard YOLOv5 (torch, xyxy) ---
def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    if not isinstance(box1, torch.Tensor):
        box1 = torch.tensor(box1)
    if not isinstance(box2, torch.Tensor):
        box2 = torch.tensor(box2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def make_grid(nx, ny, device, dtype):
    yv, xv = torch.meshgrid(torch.arange(ny, device=device, dtype=dtype), torch.arange(nx, device=device, dtype=dtype), indexing='ij')
    grid = torch.stack((xv, yv), 2).unsqueeze(0).unsqueeze(0)  # (1,1,ny,nx,2)
    return grid

def yolo_decode(outputs, anchors, stride):
    # outputs: list of 3 tensors (H,W,C) -> (C,H,W)
    z = []
    for i, out in enumerate(outputs):
        out = torch.from_numpy(out).unsqueeze(0)  # (1,C,H,W)
        bs, c, h, w = out.shape
        num_classes = c // 3 - 5  # (C = 3 * (5+num_classes))
        out = out.view(bs, 3, 5 + num_classes, h, w).permute(0,1,3,4,2).contiguous()  # (1,3,h,w,5+num_classes)
        grid = make_grid(w, h, out.device, out.dtype)
        anchor_grid = anchors[i].view(1, 3, 1, 1, 2)
        xy = (out[..., 0:2].sigmoid() * 2 + grid) * stride[i]
        wh = (out[..., 2:4].sigmoid() * 2) ** 2 * anchor_grid
        obj_conf = out[..., 4:5].sigmoid()
        cls_conf = out[..., 5:].sigmoid()  # shape (..., num_classes)
        y = torch.cat((xy, wh, obj_conf, cls_conf), -1)  # (..., 5+num_classes)
        z.append(y.view(bs, -1, 5 + num_classes))
    return torch.cat(z, 1)

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    """Non-Maximum Suppression (NMS) on inference results (YOLOv5 style)"""
    xc = prediction[..., 4] > conf_thres  # candidates
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
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
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    # Clip boxes
    boxes[..., 0].clamp_(0, img0_shape[1])  # x1
    boxes[..., 1].clamp_(0, img0_shape[0])  # y1
    boxes[..., 2].clamp_(0, img0_shape[1])  # x2
    boxes[..., 3].clamp_(0, img0_shape[0])  # y2
    return boxes

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
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

# --- Funzione postprocessing identica a detectTensor.py ---
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

def postprocess(n_images, onnx_path=None, dataset=None, img_path=None):
    print(f"[INFO] Postprocessing {n_images} image(s) from dataset '{dataset}' (HLS output mode)")
    img1_shape = (640, 640)  # shape input rete
    all_dets = []  # Aggiunto per raccogliere tutte le detections
    for i in range(int(n_images)):
        # --- Leggi i risultati hardware (HLS) ---
        out0 = read_output("/tmp/hls_output_0.bin", (int(n_images), 18, 80, 80))[i]
        out1 = read_output("/tmp/hls_output_1.bin", (int(n_images), 18, 40, 40))[i]
        out2 = read_output("/tmp/hls_output_2.bin", (int(n_images), 18, 20, 20))[i]
        print(f"[DEBUG] out0 shape: {out0.shape}, dtype: {out0.dtype}, min: {out0.min():.4f}, max: {out0.max():.4f}")
        print(f"[DEBUG] out1 shape: {out1.shape}, dtype: {out1.dtype}, min: {out1.min():.4f}, max: {out1.max():.4f}")
        print(f"[DEBUG] out2 shape: {out2.shape}, dtype: {out2.dtype}, min: {out2.min():.4f}, max: {out2.max():.4f}")
        # (H,W,C) -> (C,H,W)
        #  # out0 = np.transpose(out0, (2,0,1))
        # out1 = np.transpose(out1, (2,0,1))
        # out2 = np.transpose(out2, (2,0,1))
        # Converti in torch e aggiungi batch dim 
        # t0 = torch.from_numpy(out0).unsqueeze(0)
        # t1 = torch.from_numpy(out1).unsqueeze(0)
        # t2 = torch.from_numpy(out2).unsqueeze(0)
        t0 = torch.from_numpy(out0)[None, ...]  # shape (1, 18, 80, 80)
        t1 = torch.from_numpy(out1)[None, ...]
        t2 = torch.from_numpy(out2)[None, ...]
        preds_list = [t0, t1, t2]
        # Decodifica YOLO identica a detectTensor
        preds = postprocessing(preds_list)
        print(f"[DEBUG] preds shape: {preds.shape}, dtype: {preds.dtype}, min: {preds.min():.4f}, max: {preds.max():.4f}")
        # Applica la NMS YOLOv5 originale
        dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
        dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
        all_dets.append(dets)  # Aggiunto per raccogliere le detections
        print(f"\n[Image {i}] Detections:")
        if dets.shape[0] == 0:
            print("No detections.")
        else:
            def iou_matrix(boxes):
                n = boxes.shape[0]
                ious = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            ious[i, j] = 1.0
                        else:
                            box1 = boxes[i]
                            box2 = boxes[j]
                            xA = max(box1[0], box2[0])
                            yA = max(box1[1], box2[1])
                            xB = min(box1[2], box2[2])
                            yB = min(box1[3], box2[3])
                            interArea = max(0, xB - xA) * max(0, yB - yA)
                            boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            ious[i, j] = interArea / (boxAArea + boxBArea - interArea + 1e-6)
                return ious
            boxes_np = dets[:, :4].cpu().numpy() if torch.is_tensor(dets) else dets[:, :4]
            ious = iou_matrix(boxes_np)
            print("IoU matrix between detections:")
            np.set_printoptions(precision=2, suppress=True)
            print(ious)
            # Ridimensiona le box alle dimensioni originali usando ratio e pad
            if img_path is not None:
                img0 = cv2.imread(str(img_path))
                assert img0 is not None, f"Image not found: {img_path}"
                img0_shape = img0.shape[:2][::-1]  # (width, height)
                _, ratio, pad = letterbox(img0, new_shape=img1_shape, auto=False)
            else:
                img0_shape = (640, 640)
                ratio, pad = (1.0, 1.0), (0.0, 0.0)
            dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape[::-1], (ratio, pad)).round()
            # Filtro YOLOv5: elimina box troppo piccole o fuori dai limiti
            filtered = []
            for det in dets:  # rimosso reversed()
                x1, y1, x2, y2, conf, cls = det.tolist()
                w, h = x2 - x1, y2 - y1
                if w >= 1 and h >= 1 and 0 <= x1 < img0_shape[0] and 0 <= y1 < img0_shape[1] and 0 < x2 <= img0_shape[0] and 0 < y2 <= img0_shape[1]:
                    filtered.append((x1, y1, x2, y2, conf, cls))
            # Stampa solo le predizioni testuali
            for x1, y1, x2, y2, conf, cls in filtered:
                print(f"BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Confidence: {conf:.2f}, Class: {int(cls)}")
            # ---
            # Per disegnare le bounding box su un'immagine, decommenta e usa il seguente codice:
            # img0 = cv2.imread(str(img_path))
            # for x1, y1, x2, y2, conf, cls in filtered:
            #     cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            #     cv2.putText(img0, f"{int(cls)} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            # cv2.imwrite("output_with_boxes.jpg", img0)
            # ---

# --- Funzione ap_per_class e compute_ap (da YOLOv5, numpy) ---
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1] if tp.ndim > 1 else 1)), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0] if recall.ndim > 1 else recall, left=0)  # negative x, xp because xp decreases
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0] if precision.ndim > 1 else precision, left=1)  # p at pr_score
        # AP from recall-precision curve
        for j in range(tp.shape[1] if tp.ndim > 1 else 1):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j] if recall.ndim > 1 else recall, precision[:, j] if precision.ndim > 1 else precision)
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes] if hasattr(names, 'items') else names  # list: only classes that have data
    names = dict(enumerate(names)) if names else {}
    i = smooth(f1.mean(0), 0.1).argmax() if f1.ndim > 1 else 0  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i] if f1.ndim > 1 else f1
    tp_ = (r * nt).round()  # true positives
    fp_ = (tp_ / (p + eps) - tp_).round()  # false positives
    return tp_, fp_, p, r, f1, ap, unique_classes.astype(int)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec

def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: test su output preprocess e label data/PV/labels (come postFix)
        out0_path = "/tmp/results_preprocessed_0.bin"
        out1_path = "/tmp/results_preprocessed_1.bin"
        out2_path = "/tmp/results_preprocessed_2.bin"
        n_images = 2
        img1_shape = (640, 640)
        def read_output_hwc_to_chw(path, shape_hwc):
            data = np.fromfile(path, dtype=np.float32)
            data = data.reshape(shape_hwc)
            data = np.transpose(data, (0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h, w)
            return data
        out0 = read_output_hwc_to_chw(out0_path, (n_images, 80, 80, 18))
        out1 = read_output_hwc_to_chw(out1_path, (n_images, 40, 40, 18))
        out2 = read_output_hwc_to_chw(out2_path, (n_images, 20, 20, 18))
        label_files = [
            ("/home/nikkaz/Scrivania/tesi/Fotovolt/final/py/data/PV/labels/train/0001.txt", 0, "/home/nikkaz/Scrivania/tesi/Fotovolt/final/py/data/PV/images/train/0001.jpg"),
            ("/home/nikkaz/Scrivania/tesi/Fotovolt/final/py/data/PV/labels/val/0000.txt", 1, "/home/nikkaz/Scrivania/tesi/Fotovolt/final/py/data/PV/images/val/0000.jpg")
        ]
        all_gt_cls = []
        all_dets = []
        for label_file, i, img_path in label_files:
            t0 = torch.from_numpy(out0[i])[None, ...]
            t1 = torch.from_numpy(out1[i])[None, ...]
            t2 = torch.from_numpy(out2[i])[None, ...]
            preds_list = [t0, t1, t2]
            preds = postprocessing(preds_list)
            dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
            dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
            all_dets.append(dets)
            img0 = cv2.imread(str(img_path))
            assert img0 is not None, f"Image not found: {img_path}"
            img0_shape = img0.shape[:2][::-1]
            _, ratio, pad = letterbox(img0, new_shape=img1_shape, auto=False)
            if dets.shape[0] > 0:
                dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape[::-1], (ratio, pad)).round()
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c, xc, yc, w, h = map(float, parts[:5])
                            imgw, imgh = img0_shape
                            x1 = (xc - w/2) * imgw
                            y1 = (yc - h/2) * imgh
                            x2 = (xc + w/2) * imgw
                            y2 = (yc + h/2) * imgh
                            all_gt_cls.append([c, x1, y1, x2, y2, i])
            else:
                print(f"[WARNING] Label file not found: {label_file}")
        iouv = np.linspace(0.5, 0.95, 10)
        correct_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []
        gt_arr = np.array(all_gt_cls)
        for img_idx, dets in enumerate(all_dets):
            if dets.shape[0] == 0:
                continue
            labels = gt_arr[gt_arr[:, -1] == img_idx][:, :5]
            detections = dets.cpu().numpy() if hasattr(dets, 'cpu') else dets
            print(f"[Image {img_idx}] det: {detections.shape}, labels: {labels.shape}")
            for l in labels:
                print(f"  GT: class={int(l[0])}, x1={l[1]:.1f}, y1={l[2]:.1f}, x2={l[3]:.1f}, y2={l[4]:.1f}")
            for d in detections:
                print(f"  Det: class={int(d[5])}, x1={d[0]:.1f}, y1={d[1]:.1f}, x2={d[2]:.1f}, y2={d[3]:.1f}, conf={d[4]:.2f}")
            if labels.shape[0] > 0 and detections.shape[0] > 0:
                iou_matrix = box_iou(torch.tensor(labels[:, 1:]), torch.tensor(detections[:, :4])).numpy()
                print(f"  IoU matrix:\n{iou_matrix}")
            correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool) if labels.shape[0] == 0 else process_batch(detections, labels, iouv)
            correct_list.append(correct)
            conf_list.append(detections[:, 4])
            pred_cls_list.append(detections[:, 5])
            target_cls_list.append(labels[:, 0] if labels.shape[0] > 0 else np.array([]))
        stats = [np.concatenate(x, 0) if len(x) else np.array([]) for x in (correct_list, conf_list, pred_cls_list, target_cls_list)]
        if len(stats[0]):
            tp_, fp_, p, r, f1, ap, ap_class = ap_per_class(*stats)
            mp, mr, map50, map = p.mean(), r.mean(), ap[:,0].mean(), ap.mean()
            print(f"\n{'Class':>10s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP':>8s}")
            print(f"{'all':>10s} {mp:8.3f} {mr:8.3f} {map50:8.3f} {map:8.3f}")
        else:
            print("[WARNING] Nessuna predizione valida per il calcolo delle metriche.")
    elif len(sys.argv) == 2 and sys.argv[1] == "--preproc-test-pv":
        # Modalità test: usa output preprocess e label data/PV/labels
        out0_path = "/tmp/results_preprocessed_0.bin"
        out1_path = "/tmp/results_preprocessed_1.bin"
        out2_path = "/tmp/results_preprocessed_2.bin"
        n_images = 2
        img1_shape = (640, 640)
        # Carica output preprocess (HWC, float32)
        def read_output_hwc_to_chw(path, shape_hwc):
            data = np.fromfile(path, dtype=np.float32)
            data = data.reshape(shape_hwc)
            data = np.transpose(data, (0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h, w)
            return data
        out0 = read_output_hwc_to_chw(out0_path, (n_images, 80, 80, 18))
        out1 = read_output_hwc_to_chw(out1_path, (n_images, 40, 40, 18))
        out2 = read_output_hwc_to_chw(out2_path, (n_images, 20, 20, 18))
        label_files = [
            ("data/PV/labels/train/0001.txt", 0),
            ("data/PV/labels/val/0000.txt", 1)
        ]
        all_pred, all_conf, all_cls, all_gt_cls = [], [], [], []
        all_dets = []  # Aggiunto per raccogliere tutte le detections
        for label_file, i in label_files:
            t0 = torch.from_numpy(out0[i])[None, ...]
            t1 = torch.from_numpy(out1[i])[None, ...]
            t2 = torch.from_numpy(out2[i])[None, ...]
            preds_list = [t0, t1, t2]
            preds = postprocessing(preds_list)
            dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
            dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
            all_dets.append(dets)  # Aggiunto per raccogliere le detections
            print(f"\n[Image {i}] Detections:")
            if dets.shape[0] == 0:
                print("No detections.")
            else:
                img0_shape = (640, 640)
                ratio, pad = (1.0, 1.0), (0.0, 0.0)
                # Se vuoi testare con immagini originali, qui puoi caricare l'immagine e calcolare ratio/pad reali
                # img0 = cv2.imread(img_path) ...
                print(f"[DEBUG] img_idx={i}, ratio={ratio}, pad={pad}")
                if dets.shape[0] > 0:
                    dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape[::-1], (ratio, pad)).round()
                for det in dets:
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    w, h = x2 - x1, y2 - y1
                    if w >= 1 and h >= 1 and 0 <= x1 < img0_shape[0] and 0 <= y1 < img0_shape[1] and 0 < x2 <= img0_shape[0] and 0 < y2 <= img0_shape[1]:
                        print(f"Class: {int(cls)}, Confidence: {conf:.2f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            # Carica ground truth
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c, xc, yc, w, h = map(float, parts[:5])
                            imgw, imgh = 640, 640
                            x1 = (xc - w/2) * imgw
                            y1 = (yc - h/2) * imgh
                            x2 = (xc + w/2) * imgw
                            y2 = (yc + h/2) * imgh
                            all_gt_cls.append([c, x1, y1, x2, y2, i])  # aggiungi image_id
                            print(f"[Image {i}] GT label: class={int(c)}, x1={x1:.7f}, y1={y1:.7f}, x2={x2:.7f}, y2={y2:.7f}")
            else:
                print(f"[WARNING] Label file not found: {label_file}")
        # Calcola metriche se ci sono label
        if all_gt_cls:
            # Prepara array per stats come in YOLOv5
            iouv = np.linspace(0.5, 0.95, 10)
            correct_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []
            gt_arr = np.array(all_gt_cls)
            for img_idx, dets in enumerate(all_dets):  # all_dets: lista di dets per immagine
                if dets.shape[0] == 0:
                    continue
                # Filtra tutte le label per questa immagine tramite image_id
                labels = gt_arr[gt_arr[:, -1] == img_idx][:, :5]  # Usa direttamente le label già in pixel
                detections = dets.cpu().numpy() if hasattr(dets, 'cpu') else dets
                print(f"[DEBUG] img_idx={img_idx}, detections.shape={detections.shape}, labels.shape={labels.shape}")
                print(f"[DEBUG] detections: {detections}")
                print(f"[DEBUG] labels: {labels}")
                # Stampa aree e limiti per debug
                for i, l in enumerate(labels):
                    c, x1, y1, x2, y2 = l
                    area = max(0, x2-x1) * max(0, y2-y1)
                    print(f"[DEBUG] label[{i}]: class={int(c)}, x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, area={area:.2f}")
                for i, d in enumerate(detections):
                    x1, y1, x2, y2, conf, cls = d
                    area = max(0, x2-x1) * max(0, y2-y1)
                    print(f"[DEBUG] det[{i}]: class={int(cls)}, x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, area={area:.2f}, conf={conf:.2f}")
                # Stampa matrice IoU tra predizioni e label
                if labels.shape[0] > 0 and detections.shape[0] > 0:
                    iou_matrix = box_iou(torch.tensor(labels[:, 1:]), torch.tensor(detections[:, :4])).numpy()
                    print(f"[DEBUG] IoU matrix (labels x detections):\n{iou_matrix}")
                if labels.shape[0] == 0:
                    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
                else:
                    correct = process_batch(detections, labels, iouv)
                correct_list.append(correct)
                conf_list.append(detections[:, 4])
                pred_cls_list.append(detections[:, 5])
                target_cls_list.append(labels[:, 0] if labels.shape[0] > 0 else np.array([]))
            # Concatena come in YOLOv5
            stats = [np.concatenate(x, 0) if len(x) else np.array([]) for x in (correct_list, conf_list, pred_cls_list, target_cls_list)]
            if len(stats[0]):
                tp_, fp_, p, r, f1, ap, ap_class = ap_per_class(*stats)
                mp, mr, map50, map = p.mean(), r.mean(), ap[:,0].mean(), ap.mean()
                print(f"\n{'Class':>10s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP':>8s}")
                print(f"{'all':>10s} {mp:8.3f} {mr:8.3f} {map50:8.3f} {map:8.3f}")
            else:
                print("[WARNING] Nessuna predizione valida per il calcolo delle metriche.")
        else:
            print("[WARNING] Nessuna ground truth trovata, metriche non calcolate.")
    elif len(sys.argv) == 2 and sys.argv[1] == "--preproc-test-pv":
        # Modalità test: usa output preprocess e label data/PV/labels
        out0_path = "/tmp/results_preprocessed_0.bin"
        out1_path = "/tmp/results_preprocessed_1.bin"
        out2_path = "/tmp/results_preprocessed_2.bin"
        n_images = 2
        img1_shape = (640, 640)
        # Carica output preprocess (HWC, float32)
        def read_output_hwc_to_chw(path, shape_hwc):
            data = np.fromfile(path, dtype=np.float32)
            data = data.reshape(shape_hwc)
            data = np.transpose(data, (0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h, w)
            return data
        out0 = read_output_hwc_to_chw(out0_path, (n_images, 80, 80, 18))
        out1 = read_output_hwc_to_chw(out1_path, (n_images, 40, 40, 18))
        out2 = read_output_hwc_to_chw(out2_path, (n_images, 20, 20, 18))
        label_files = [
            ("data/PV/labels/train/0001.txt", 0),
            ("data/PV/labels/val/0000.txt", 1)
        ]
        all_pred, all_conf, all_cls, all_gt_cls = [], [], [], []
        all_dets = []  # Aggiunto per raccogliere tutte le detections
        for label_file, i in label_files:
            t0 = torch.from_numpy(out0[i])[None, ...]
            t1 = torch.from_numpy(out1[i])[None, ...]
            t2 = torch.from_numpy(out2[i])[None, ...]
            preds_list = [t0, t1, t2]
            preds = postprocessing(preds_list)
            dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
            dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
            all_dets.append(dets)  # Aggiunto per raccogliere le detections
            print(f"\n[Image {i}] Detections:")
            if dets.shape[0] == 0:
                print("No detections.")
            else:
                img0_shape = (640, 640)
                ratio, pad = (1.0, 1.0), (0.0, 0.0)
                # Se vuoi testare con immagini originali, qui puoi caricare l'immagine e calcolare ratio/pad reali
                # img0 = cv2.imread(img_path) ...
                print(f"[DEBUG] img_idx={i}, ratio={ratio}, pad={pad}")
                if dets.shape[0] > 0:
                    dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape[::-1], (ratio, pad)).round()
                for det in dets:
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    w, h = x2 - x1, y2 - y1
                    if w >= 1 and h >= 1 and 0 <= x1 < img0_shape[0] and 0 <= y1 < img0_shape[1] and 0 < x2 <= img0_shape[0] and 0 < y2 <= img0_shape[1]:
                        print(f"Class: {int(cls)}, Confidence: {conf:.2f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            # Carica ground truth
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c, xc, yc, w, h = map(float, parts[:5])
                            imgw, imgh = 640, 640
                            x1 = (xc - w/2) * imgw
                            y1 = (yc - h/2) * imgh
                            x2 = (xc + w/2) * imgw
                            y2 = (yc + h/2) * imgh
                            all_gt_cls.append([c, x1, y1, x2, y2, i])  # aggiungi image_id
                            print(f"[Image {i}] GT label: class={int(c)}, x1={x1:.7f}, y1={y1:.7f}, x2={x2:.7f}, y2={y2:.7f}")
            else:
                print(f"[WARNING] Label file not found: {label_file}")
        # Calcola metriche se ci sono label
        if all_gt_cls:
            # Prepara array per stats come in YOLOv5
            iouv = np.linspace(0.5, 0.95, 10)
            correct_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []
            gt_arr = np.array(all_gt_cls)
            for img_idx, dets in enumerate(all_dets):  # all_dets: lista di dets per immagine
                if dets.shape[0] == 0:
                    continue
                # Filtra tutte le label per questa immagine tramite image_id
                labels = gt_arr[gt_arr[:, -1] == img_idx][:, :5]  # Usa direttamente le label già in pixel
                detections = dets.cpu().numpy() if hasattr(dets, 'cpu') else dets
                print(f"[DEBUG] img_idx={img_idx}, detections.shape={detections.shape}, labels.shape={labels.shape}")
                print(f"[DEBUG] detections: {detections}")
                print(f"[DEBUG] labels: {labels}")
                # Stampa aree e limiti per debug
                for i, l in enumerate(labels):
                    c, x1, y1, x2, y2 = l
                    area = max(0, x2-x1) * max(0, y2-y1)
                    print(f"[DEBUG] label[{i}]: class={int(c)}, x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, area={area:.2f}")
                for i, d in enumerate(detections):
                    x1, y1, x2, y2, conf, cls = d
                    area = max(0, x2-x1) * max(0, y2-y1)
                    print(f"[DEBUG] det[{i}]: class={int(cls)}, x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, area={area:.2f}, conf={conf:.2f}")
                # Stampa matrice IoU tra predizioni e label
                if labels.shape[0] > 0 and detections.shape[0] > 0:
                    iou_matrix = box_iou(torch.tensor(labels[:, 1:]), torch.tensor(detections[:, :4])).numpy()
                    print(f"[DEBUG] IoU matrix (labels x detections):\n{iou_matrix}")
                if labels.shape[0] == 0:
                    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
                else:
                    correct = process_batch(detections, labels, iouv)
                correct_list.append(correct)
                conf_list.append(detections[:, 4])
                pred_cls_list.append(detections[:, 5])
                target_cls_list.append(labels[:, 0] if labels.shape[0] > 0 else np.array([]))
            # Concatena come in YOLOv5
            stats = [np.concatenate(x, 0) if len(x) else np.array([]) for x in (correct_list, conf_list, pred_cls_list, target_cls_list)]
            if len(stats[0]):
                tp_, fp_, p, r, f1, ap, ap_class = ap_per_class(*stats)
                mp, mr, map50, map = p.mean(), r.mean(), ap[:,0].mean(), ap.mean()
                print(f"\n{'Class':>10s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP':>8s}")
                print(f"{'all':>10s} {mp:8.3f} {mr:8.3f} {map50:8.3f} {map:8.3f}")
            else:
                print("[WARNING] Nessuna predizione valida per il calcolo delle metriche.")
        else:
            print("[WARNING] Nessuna ground truth trovata, metriche non calcolate.")
    elif len(sys.argv) == 5:
        # Modalità solo stampa detection: python3 postMetrics.py <out0.bin> <out1.bin> <out2.bin> <dataset_name>
        out0_path, out1_path, out2_path, dataset_name = sys.argv[1:5]
        base_data_dir = "/home-ssd/datasets/"
        # Cerca la cartella dataset in base_data_dir, sia maiuscolo che minuscolo
        dataset_root = os.path.join(base_data_dir, dataset_name.upper()) if os.path.isdir(os.path.join(base_data_dir, dataset_name.upper())) else os.path.join(base_data_dir, dataset_name.lower())
        img1_shape = (640, 640)
        import glob
        import os
        import cv2
        # Deduci n_images dalla dimensione di out0.bin
        out0_size = os.path.getsize(out0_path)
        n_images = out0_size // (80 * 80 * 18 * 4)  # float32
        # Trova tutte le immagini nei sottodir /images/{train,val,test}/
        img_files = []
        splits = ["train", "val", "test"]
        for split in splits:
            img_dir = os.path.join(dataset_root, "images", split)
            if os.path.isdir(img_dir):
                img_files += sorted(glob.glob(os.path.join(img_dir, '*.*')))
        assert len(img_files) >= n_images, f"Not enough images in {dataset_root}/images/ (found {len(img_files)}, expected {n_images})"
        # Carica output binari (HWC, float32)
        def read_output_hwc_to_chw(path, shape_hwc):
            data = np.fromfile(path, dtype=np.float32)
            data = data.reshape(shape_hwc)
            data = np.transpose(data, (0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h, w)
            return data
        out0 = read_output_hwc_to_chw(out0_path, (n_images, 80, 80, 18))
        out1 = read_output_hwc_to_chw(out1_path, (n_images, 40, 40, 18))
        out2 = read_output_hwc_to_chw(out2_path, (n_images, 20, 20, 18))
        for i in range(n_images):
            # Nella modalità con argomenti (len(sys.argv) == 5), NON fare reshape/transpose/unsqueeze:
            # I tensori vanno caricati direttamente come torch.from_numpy(outX[i]) senza alcuna manipolazione,
            # a differenza di bin2pt.py e del preprocess (NO .reshape, .transpose, .unsqueeze!)
            t0 = torch.from_numpy(out0[i])  # shape (18, 80, 80)
            t1 = torch.from_numpy(out1[i])
            t2 = torch.from_numpy(out2[i])
            preds_list = [t0, t1, t2]
            preds = postprocessing(preds_list)
            dets_list = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
            dets = dets_list[0] if len(dets_list) > 0 else torch.zeros((0, 6))
            # Ricava path immagine originale
            img_path = img_files[i]
            img0 = cv2.imread(img_path)
            img0_shape = img0.shape[:2][::-1] if img0 is not None else (640, 640)
            _, ratio, pad = letterbox(img0, new_shape=img1_shape, auto=False) if img0 is not None else ((640, 640), (1.0, 1.0), (0.0, 0.0))
            if dets.shape[0] > 0:
                dets[:, :4] = scale_boxes(img1_shape, dets[:, :4], img0_shape, (ratio, pad)).round()
            print(f"\n[Image {i}] Detections:")
            if dets.shape[0] == 0:
                print("No detections.")
            else:
                for det in dets:
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    w, h = x2 - x1, y2 - y1
                    if w >= 1 and h >= 1 and 0 <= x1 < img0_shape[0] and 0 <= y1 < img0_shape[1] and 0 < x2 <= img0_shape[0] and 0 < y2 <= img0_shape[1]:
                        print(f"Class: {int(cls)}, Confidence: {conf:.2f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
