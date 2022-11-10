import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from quant_dorefa import *


class PreActBlock_conv_Q(nn.Module):
  '''Pre-activation version of the BasicBlock.'''

  def __init__(self, wbit, abit, in_planes, out_planes, stride=1, act_q=None):
    super(PreActBlock_conv_Q, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit=wbit)
    self.act_q = act_q

    self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn0 = nn.BatchNorm2d(out_planes)
    self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)

    self.skip_conv = None
    if stride != 1:
      self.skip_conv = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      self.skip_bn = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    # out = self.act_q(x)

    if self.skip_conv is not None:
      shortcut = self.skip_conv(x)
      # shortcut = self.skip_bn(shortcut)
      shortcut = self.act_q(shortcut)
    else:
      shortcut = x

    out = self.conv0(x)
    # out = self.bn0(out)
    out = F.relu(out)
    out = self.act_q(out)
    out = self.conv1(out)
    # out = self.bn1(out)
    out = self.act_q(out)
    out += shortcut
    out = F.relu(out)
    out = self.act_q(out)
    return out


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, wbit, abit, num_classes):
    super(PreActResNet, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit=wbit)
    Linear = linear_Q_fn(w_bit=wbit)
    self.abit = abit
    self.conv0 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.act_q = activation_quantize_fn(a_bit=abit)
    self.bn = nn.BatchNorm2d(16)

    self.layers = nn.ModuleList()
    in_planes = 16
    strides = [1] * (num_units[0]) + \
              [2] + [1] * (num_units[1] - 1) + \
              [2] + [1] * (num_units[2] - 1)
    channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
    for stride, channel in zip(strides, channels):
      self.layers.append(block(wbit, abit, in_planes, channel, stride, self.act_q))
      in_planes = channel

    self.avgpool = nn.AvgPool2d(8, stride=1)
    self.fc = nn.Conv2d(64, num_classes, 1, bias=False)
    self.export = False
    self.show = False

  def forward(self, x):
    # out = x - 0.5;

    if (self.show):
        # show = torch.round(out*(2**self.abit))
        show = x

        for b in range(show.shape[0]):
            for ih in range(show.shape[2]):
                for iw in range(show.shape[3]):
                    for ich in range(show.shape[1]):
                        # weight_value = np.random.randint(0, 256)
                        print(show[b][ich][ih][iw])
                    print("----------------------")
                    if (iw >= 2):
                        break
                if (ih >= 2):
                    break
            break

    out = self.conv0(x)

    # out = self.bn(out)
    out = F.relu(out)
    out = self.act_q(out)

    if (self.show):
        # show = torch.round(out*(2**self.abit))
        show = torch.round(out*(2**self.abit))

        for b in range(show.shape[0]):
            for ih in range(show.shape[2]):
                for iw in range(show.shape[3]):
                    for ich in range(show.shape[1]):
                        # weight_value = np.random.randint(0, 256)
                        print(show[b][ich][ih][iw])
                    sys.exit(0)

    for layer in self.layers:
      out = layer(out)
    out = self.avgpool(out)
    out = self.fc(out)
    if not self.export:
      return out.view(out.size(0), -1)
    return out


def resnet20(wbits, abits, num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [3, 3, 3], wbits, abits, num_classes=num_classes)


def resnet56(wbits, abits, num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [9, 9, 9], wbits, abits, num_classes=num_classes)


if __name__ == '__main__':
  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = resnet20(wbits=1, abits=2)
  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())
