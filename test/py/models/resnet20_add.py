# 2020.01.10-Replaced conv with adder
#         Haoran & Xiaohan

# from models import adder
from adder import adder
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet20_add', 'LeNet_add_vis']

def conv3x3(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
    " 3x3 convolution with padding "
    return adder.Adder2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                         quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
        super(ResNet, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.quantize_v = quantize_v
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)


        # init (new add)
        for m in self.modules():
            # if isinstance(m, adder.Adder2D):
            #     nn.init.kaiming_normal_(m.adder, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                adder.Adder2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity, quantize_v=self.quantize_v), # adder.Adder2D
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity, quantize_v=self.quantize_v))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity, quantize_v=self.quantize_v))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        print(x.shape)
        x = self.fc(x)
        x = self.bn2(x)
        return x.view(x.size(0), -1)

class ResNet_vis(nn.Module):

    def __init__(self, block, layers, num_classes=10, quantize=False, weight_bits=8, sparsity=0):
        super(ResNet_vis, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc_1 = nn.Linear(64 * block.expansion, 2)
        self.fc_2 = nn.Linear(2, num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                adder.Adder2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity), # adder.Adder2D
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x.view(x.size(0), -1)

class LeNet_vis(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet_vis, self).__init__()
        self.conv1_1 = adder.Adder2D(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = adder.Adder2D(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        # self.conv2_1 = adder.Adder2D(32, 64, kernel_size=5, padding=2)
        # self.prelu2_1 = nn.PReLU()
        # self.conv2_2 = adder.Adder2D(64, 64, kernel_size=5, padding=2)
        # self.prelu2_2 = nn.PReLU()
        # self.conv3_1 = adder.Adder2D(64, 128, kernel_size=5, padding=2)
        # self.prelu3_1 = nn.PReLU()
        # self.conv3_2 = adder.Adder2D(128, 128, kernel_size=5, padding=2)
        # self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(32*3*3, 2)
        self.ip2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        # x = self.prelu2_1(self.conv2_1(x))
        # x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        # x = self.prelu3_1(self.conv3_1(x))
        # x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 32*3*3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        # return ip1, F.log_softmax(ip2, dim=1)
        return ip1, ip2


def resnet20_add(num_classes=10, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm', **kwargs):
    print(quantize, sparsity)
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, quantize=quantize,
                    weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)

def resnet20_add_vis(num_classes=10, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    return ResNet_vis(BasicBlock, [3, 3, 3], num_classes=num_classes, quantize=quantize,
                    weight_bits=weight_bits, sparsity=sparsity)

def LeNet_add_vis(num_classes=10, **kwargs):
    return LeNet_vis(num_classes=num_classes)
