import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, export=False):
      if (not export):
        if k == 32:
          out = input
        elif k == 1:
          out = torch.sign(input)
        else:
          # TODO: Modified with respect to original DOREFA
          # More hardware friendly
          n = float(2 ** k)
          out = (torch.round(input * n) - 1) / n
          # One is the new 0
          out = torch.clamp(out, 0, 1)
      else:
        out = input
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input, None

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)
    self.export = False

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      # weight = weight / 2 / max_w + 0.5
      # weight_q = max_w * (2 * self.uniform_q(weight, self.export) - 1)

      weight = weight / 2 + 0.5
      weight_q = (2 * self.uniform_q(weight, self.export) - 1)
      print(weight_q.shape)
      for och in range(weight_q.shape[0]):
          for ich in range(weight_q.shape[1]):
              print(weight_q[och][ich])

    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)
    self.export = False

  def forward(self, x):
    if not self.export:
      if self.a_bit == 32:
        activation_q = x
      else:
        # TODO: Modified with respect to original DOREFA
        activation_q = self.uniform_q(torch.clamp(x, 0, 1), self.export)
        # activation_q = activation_q - 0.5
        # print(np.unique(activation_q.detach().numpy()))
    else:
      activation_q = x
    return activation_q


def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)
      self.export = False

    def forward(self, input, order=None):
      if not self.export:
        weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
        return F.conv2d(input, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
      else:
        self.quantize_fn.export = True
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)
      self.export = False

    def forward(self, input):
      if not self.export:
        weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
        return F.linear(input, weight_q, self.bias)
      else:
        self.quantize_fn.export = True
        return F.linear(input, self.weight, self.bias)

  return Linear_Q


if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt

  a = torch.rand(1, 3, 32, 32)

  Conv2d = conv2d_Q_fn(w_bit=1)
  conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

  img = torch.randn(1, 256, 56, 56)
  print(img.max().item(), img.min().item())
  out = conv(img)
  print(out.max().item(), out.min().item())
