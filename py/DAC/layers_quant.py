import torch.nn as nn
import torch

from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat 
from brevitas.nn import QuantIdentity 
from brevitas.quant_tensor import QuantTensor
from common import *

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilatation=1, act=True):
        super().__init__()
        self.conv = QuantConv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding = autopad(kernel_size,padding, dilatation),
                              groups=groups, dilation=dilatation,
                              bias=False,
                              weight_quant = Int8WeightPerTensorFixedPoint,
                              bias_quant = Int16Bias,
                              input_quant = Int8ActPerTensorFixedPoint,
                              output_quant = Int8ActPerTensorFixedPoint
                              )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = QuantReLU(quant_type=QuantType.INT,
                             restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                             max_val=6.0,
                             scaling_impl_type=ScalingImplType.CONST,
                             act_quant=CommonUintActQuant,
                             bit_width=8)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super().__init__()
#         self.d = dimension

#     def forward(self, x1, x2):
#         return torch.cat([x1, x2], self.d)
    
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.quant_inp = QuantIdentity(bit_width=8, return_quant_tensor=False)
     
    def forward(self, x1, x2):
        torch.use_deterministic_algorithms(False)
        qx1 = self.quant_inp(x1)
        qx2 = self.quant_inp(x2)
        return QuantTensor.cat([qx1, qx2], dim=1)
    
    
class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(QuantConv2d(x, self.no * self.na, 1, weight_quant = Int8WeightPerTensorFixedPoint,
                              bias_quant = Int16Bias,
                              input_quant = Int8ActPerTensorFixedPoint,
                              output_quant = Int8ActPerTensorFixedPoint
                              ) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        # print(x.shape)
        # print(self.m[0](x[0]).shape)
        # print(self.m[1](x[1]).shape)
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        self.stride = [16, 32]
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # print(self.stride)
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
    
class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(QuantConv2d(x, self.no * self.na, 1 ,weight_quant = Int8WeightPerTensorFixedPoint,
                              bias_quant = Int16Bias,
                              input_quant = Int8ActPerTensorFixedPoint,
                              output_quant = Int8ActPerTensorFixedPoint
                              ) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])
    
    
class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def _init(self, c1, c=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))