import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from brevitas.nn import QuantConv2d, QuantReLU, QuantAdaptiveAvgPool2d
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from common import CommonIntActQuant, CommonUintActQuant
from brevitas.quant import Int16Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat 
import torchvision
from preprocess import *
# from resnet8 import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
from preprocess import *
from bar_show import progress_bar
cudnn.benchmark = True
best_acc = 0  # best test accuracy
start_epoch = 0
f = open("output.txt", "a")

def conv3x3(in_planes, out_planes, stride=1, weight_bits=8):
    return QuantConv2d(in_planes,
                       out_planes,
                       kernel_size=(3,3), 
                       stride=stride,
                       padding=1,
                       bias=True,
                       weight_quant = Int8WeightPerTensorFixedPoint,
                       bias_quant = Int16Bias,
                       input_quant = Int8ActPerTensorFixedPoint,
                       output_quant = Int8ActPerTensorFixedPoint
                       )


def print_act(x, fd, red=False):
    if not red:
        fd.write("%0d\n" % x.shape[2])
        fd.write("%0d\n" % x.shape[3])
        fd.write("%0d\n" % x.shape[1])
        for ih in range(x.shape[2]):
            for iw in range(x.shape[3]):
                for ch in range(x.shape[1]):
                    fd.write("%f " % x[0, ch, ih, iw].detach().cpu().numpy())
                fd.write("\n")
    else:
        fd.write("%0d\n" % x.shape[1])
        for ch in range(x.shape[1]):
            fd.write("%f " % x[0, ch].detach().cpu().numpy())
        fd.write("\n")

class TestModel(nn.Module) :
    def __init__(self):
        
        super().__init__()
        self.conv1 = QuantConv2d(3, 16, kernel_size=(3, 3),
                     weight_bit_width = 8,
                     stride=1, 
                     padding=1, 
                     bias=True, 
                     weight_quant= Int8WeightPerTensorFixedPoint,
                     bias_quant=Int16Bias,
                     input_quant = Int8ActPerTensorFixedPoint,
                     output_quant = Int8ActPerTensorFixedPoint,
                     )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = QuantReLU(quant_type=QuantType.INT,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST, 
            act_quant=CommonUintActQuant,
            bit_width=8)
        self.avgpool = QuantAdaptiveAvgPool2d((1, 1),trunc_quant = None)
                     
        self.fc = QuantConv2d(16, 10,
                kernel_size=(1, 1), bias=False,
                weight_quant = Int8WeightPerTensorFixedPoint,
                input_quant = Int8ActPerTensorFixedPoint,
                output_quant = Int8ActPerTensorFixedPoint,
                bias_quant=Int16Bias
                )

    def forward(self, x, train=True):
        if (not train):
            with open("log.txt", "w+") as fd:
                fd.write("ACT IN\n")
                print_act(x, fd)
        x = self.conv1(x)
        # if (not train):
        #     print("CONV OUT\n")
        #     print_act(x)
        x = self.relu(x)
        if (not train):
            with open("log.txt", "a") as fd:
                fd.write("RELU OUT\n")
                print_act(x, fd)
                # for ih in range(2):
                #     for iw in range(2):
                #         for ch in range(3):
                #             fd.write("%f " % x[0, ch, ih, iw].detach().cpu().numpy())
                #         fd.write("\n")
        x = self.avgpool(x)
        # if (not train):
        #     with open("log.txt", "a") as fd:
        #         print("POOLOUT")
        #         fd.write("POOL OUT\n")
        #         print_act(x, fd)
        x = self.fc(x)
        if (not train):
            with open("log.txt", "a") as fd:
                fd.write("NET OUT\n")
                print_act(x, fd, red=True)
        return x

def testModel( **kwargs):
    return TestModel()



def main() :
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss, correct, total = 0, 0 ,0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
            optimizer.zero_grad()
            outputs = model(inputs).view(model(inputs).size(0),-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

               


    def test(epoch):
        # pass
        global best_acc
        model.eval()

        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                outputs = model(inputs).view(model(inputs).size(0),-1)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(eval_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

                

        acc = 100. * correct / total
        if acc > best_acc:
            print('Exporting..')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            best_acc = acc
    dataset = torchvision.datasets.CIFAR10
    train_dataset = dataset(root='./data', train=True, download=True,
                          transform=cifar_transform(is_training=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                             num_workers=4)

    eval_dataset = dataset(root='./data', train=False, download=True,
                         transform=cifar_transform(is_training=False))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=100, shuffle=False,
                                            num_workers=4)

    print('===> Building ResNet..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model =testModel().to('cuda:0')   
   
    #model = ResNet8()

         
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        #print("no cuda")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
   
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch,10):
        train(epoch)
        test(epoch)
        lr_schedu.step(epoch)
    #model.to('cpu') 
    dataset = torchvision.datasets.CIFAR10
    eval_dataset = dataset(root='./data', train=False, download=True,
                         transform=cifar_transform(is_training=False))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                            num_workers=4)

    correct, total = 0, 0
    print ("===> Testing Network")
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        
        print("inputs\n", inputs)
        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')

        outputs = model(inputs, False)
        outputs = outputs.view(outputs.size(0),-1)
        _, predicted = outputs.max(1)
        print("predicted :\n", predicted)
        print("true:\n", targets)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("correct :\n", correct)
        if(batch_idx == 0) :
            break
         

    model.to('cpu') 
    example_path = './onnx/'  
    os.makedirs(example_path, exist_ok=True)
    from brevitas.export.onnx.qonnx.manager import QONNXManager 
    QONNXManager.export(model.module, input_shape=(1, 3, 32, 32), export_path='onnx/2layer.onnx')
    print("model exported")

if __name__ == '__main__':
    main()


     
