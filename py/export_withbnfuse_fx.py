import os
import time
import argparse
from datetime import datetime
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d, QuantIdentity
cudnn.benchmark = True
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.convbn_merge import replace_layers
from utils.convbn_merge import fuse_layers
from models.resnet_brevitas_fx import *
from utils.preprocess import *
from utils.bar_show import progress_bar
import brevitas
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import onnx
from brevitas.export.onnx.qonnx.manager import QONNXManager
parser = argparse.ArgumentParser(description='brevitas_resnet fx implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='resnetq_8w8f_cifar_fx')
parser.add_argument('--pretrain', action='store_true', default=True)
parser.add_argument('--pretrain_dir', type=str, default='resnetq_8w8f_cifar_fx')
parser.add_argument('--cifar', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=250)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--Wbits', type=int, default=8)
parser.add_argument('--Abits', type=int, default=8)

cfg = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_dir)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

val=1
print_onnx = 0
writer = SummaryWriter()

def main():
    if cfg.cifar == 10:
        print('training CIFAR-10 !')
        dataset = torchvision.datasets.CIFAR10
    elif cfg.cifar == 100:
        print('training CIFAR-100 !')
        dataset = torchvision.datasets.CIFAR100
    else:
        assert False, 'dataset unknown !'

    print('===> Preparing data ..')
    train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)

    eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                         transform=cifar_transform(is_training=False))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)

    print('===> Building ResNet..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet20(wbit=cfg.Wbits,abit=cfg.Abits).to('cuda:0')

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        #print("no cuda")

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    #optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    summary_writer = SummaryWriter(cfg.log_dir)
    
    if cfg.pretrain:
        ckpt = torch.load(os.path.join(cfg.ckpt_dir, f'checkpoint_fx.t7'))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    retrain = 0

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.to('cuda:0')
        model.train()
        train_loss, correct, total = 0, 0 ,0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
            optimizer.zero_grad()
            outputs = model(inputs).view(model(inputs).size(0),-1)
            loss = criterion(outputs, targets)
            writer.add_scalar("loss/train",loss,epoch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            if batch_idx % cfg.log_interval == 0:  #every log_interval mini_batches...
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('Accuracy/train', 100. * correct / total, epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     summary_writer.add_histogram(tag, value.detach(), global_step=epoch * len(train_loader) + batch_idx)
                #     summary_writer.add_histogram(tag + '/grad', value.grad.detach(), global_step=epoch * len(train_loader) + batch_idx)




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

                if batch_idx % cfg.log_interval == 0:  # every log_interval mini_batches...
                    summary_writer.add_scalar('Loss/test', test_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)
                    summary_writer.add_scalar('Accuracy/test', 100. * correct / total, epoch * len(train_loader) + batch_idx)

        acc = 100. * correct / total
        if acc > best_acc and retrain:
            print('Exporting..')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(cfg.ckpt_dir, f'checkpoint_quant_fx.t7'))
            model.to('cpu')
            QONNXManager.export(model.module, input_shape=(1, 3, 32, 32), export_path='onnx/Brevonnx_resnet_final_fx.onnx')            
            best_acc = acc


    def print_partial(model):
        model.eval()
        model.to('cuda:0')

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                num_workers=1)

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        for name, module in model.named_modules():
            t = type(module)
            t = t.__name__
            if 'relu' in t.lower():
                print(t, name)
                module.register_forward_hook(get_activation(name))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                outputs = model(inputs)
                outputs = outputs.view(outputs.size(0),-1)
                break

    model.to('cuda:0')
    #print(model)
    #return 0
    
    if(val) :
        for epoch in range(start_epoch, start_epoch+1):
            test(epoch)
            lr_schedu.step(epoch)
        summary_writer.close()


    def calibrate_model(calibration_loader, quant_model):
        with torch.no_grad():
            # Put the model in training mode to collect statistics
            quant_model.train()
             #for i, (images, _) in enumerate(calibration_loader):
             #   print(f'Calibration iteration {i}')
             #   # Disable quantization while collecting statistics
             #   DisableQuantInference().apply(quant_model, images)
            #  Put the model in inference mode to use those statistics
            quant_model.eval()
            bc = BiasCorrection(iterations=len(calibration_loader))
            for i, (images, _) in enumerate(calibration_loader):
                print(f'Bias correction iteration {i}')
                # Apply bias correction
                quant_model = bc.apply(quant_model, images)
        quant_model.apply(finalize_collect_stats)    
        return quant_model
  
    #change not hand-written
    print("\n------------------------ MERGE BN ---------------------\n")
    
    #model = merge_conv_bn(model)
    fuse_layers(model)
    replace_layers(model,torch.nn.BatchNorm2d ,torch.nn.Identity())
    
    model.to('cuda:0')

    post_quant = 0
    if(post_quant) :
        model = calibrate_model(train_loader,model)
    if(val) :
         test(start_epoch)
         print("\n-------------------RETRAINING-----------------\n") 
         retrain = 1
         for epoch in range(start_epoch, start_epoch+150): 
            if(not(post_quant)) :
                train(epoch)
            test(epoch)
            lr_schedu.step(epoch)
         summary_writer.close()


    #print(model.module)




    model.to('cpu')
    # dummy_input = torch.randn(10, 3, 32, 32, device="cpu")
    # example_path = './onnx/'
    # path = example_path + 'Brevonnx_resnet_final_fx.onnx'   
    # os.makedirs(example_path, exist_ok=True)
    # QONNXManager.export(model.module, input_shape=(1, 3, 32, 32), export_path='onnx/Brevonnx_resnet_final_fx.onnx')
    # from qonnx.transformation import infer_shapes
   
    #onnx.save(onnx_model, 'onnx/Brevonnx_resnet_final.onnx')                                                                        
    if (print_onnx) :   
        from qonnx.core.modelwrapper import ModelWrapper                                
        from qonnx.transformation import infer_shapes 

        
        model = ModelWrapper(path)
        inferred_model = model.transform(infer_shapes.InferShapes())
        for node in inferred_model.graph.node:                                               
            print(node.name)
            print(node)                                                        
   
    print_partial(model)

if __name__ == '__main__':
    main()
