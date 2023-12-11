import os
import torch
import torch.optim as optim
import torchvision
from utils.convbn_merge import replace_layers
from utils.convbn_merge import fuse_layers
from models.resnet_brevitas_fx import resnet20
from models.resnet8 import resnet8
from models.test_depthwise import QuantizedCifar10Net
from tiny_torch.benchmark.training_torch.visual_wake_words.vww_torch import MobileNetV1
from tiny_torch.benchmark.training_torch.anomaly_detection.torch_model import Autoencoder
from utils.preprocess import *
from brevitas.export import export_onnx_qcdq
from tqdm import tqdm
from utils.datasets import get_dataset

os.environ.setdefault('ROOT_DIR', './tmp')
os.environ.setdefault('DATA_DIR', 'data')
os.environ.setdefault('LOG_NAME', 'resnetq_8w8f_cifar_fx')
os.environ.setdefault('PRETRAIN', 'True')
os.environ.setdefault('PRETRAIN_FILE', '')
os.environ.setdefault('CIFAR', '10')
os.environ.setdefault('DATASET', 'vww')
os.environ.setdefault('LR', '0.01')
os.environ.setdefault('WD', '1e-4')
os.environ.setdefault('TRAIN_BATCH_SIZE', '32')
os.environ.setdefault('EVAL_BATCH_SIZE', '100')
os.environ.setdefault('MAX_EPOCHS', '100')
os.environ.setdefault('LOG_INTERVAL', '40')
os.environ.setdefault('NUM_WORKERS', '4')
os.environ.setdefault('WBITS', '8')
os.environ.setdefault('ABITS', '8')
os.environ.setdefault('TRAINING_TYPE', 'regression')

def main():

    root_dir = os.environ['ROOT_DIR']
    data_dir = os.environ['DATA_DIR']
    log_name = os.environ['LOG_NAME']
    pretrain = os.environ['PRETRAIN'] == 'True'
    pretrain_file = os.environ['PRETRAIN_FILE']
    cifar = int(os.environ['CIFAR'])
    dataset = os.environ['DATASET']
    lr = float(os.environ['LR'])
    wd = float(os.environ['WD'])
    train_batch_size = int(os.environ['TRAIN_BATCH_SIZE'])
    eval_batch_size = int(os.environ['EVAL_BATCH_SIZE'])
    max_epochs = int(os.environ['MAX_EPOCHS'])
    log_interval = int(os.environ['LOG_INTERVAL'])
    num_workers = int(os.environ['NUM_WORKERS'])
    Wbits = int(os.environ['WBITS'])
    Abits = int(os.environ['ABITS'])
    training_type = os.environ['TRAINING_TYPE']

    best_acc = 0  # best test accuracy
    start_epoch = 0

    log_dir = os.path.join(root_dir, 'logs', log_name)
    ckpt_dir = os.path.join(root_dir, 'ckpt', log_name)
    onnx_dir = os.path.join(root_dir, 'onnx', log_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)

    val=1
    print_onnx = 0

    train_dataset, eval_dataset, input_shape = get_dataset(dataset, cifar=cifar)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                             num_workers=num_workers)

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                            num_workers=num_workers)

    print('#### Preparing data ..')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model = resnet20(wbit=Wbits,abit=Abits).to('cuda:0')
    # model = MobileNetV1(num_filters=8, num_classes=2).to(device)
    # model = resnet8(weight_bits=Wbits,act_bits=Abits).to('cuda:0')
    # model = QuantizedCifar10Net().to(device)
    model = Autoencoder(input_shape[1], weight_bits=Wbits, act_bits=Abits).to('cuda:0')
    model.to(device)

    if Wbits < 8:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    # optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    # if 'cuda' in device:
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True
    #     #print("no cuda")
    
    if pretrain:
        print("#### Loading pretrain model..")
        if pretrain_file != '':
            ckpt = torch.load(pretrain_file, map_location=device)
        else:
            ckpt = torch.load(os.path.join(ckpt_dir, f'checkpoint_fx.t7'), map_location=device)
        if 'model_state_dict' not in ckpt:
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt['model_state_dict'])
        # if 'optimizer_state_dict' in ckpt:
        #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # if 'epoch' in ckpt:
        #     start_epoch = ckpt['epoch']
        # else: start_epoch = 0
        start_epoch = 0
        # print('#### Load last checkpoint data')
        model = model.to(device)
    else:
        start_epoch = 0
        print('#### Start from scratch')

    retrain = 0

    def train(epoch, criterion, optimizer, best_acc):
        train_loss, correct, total = 0, 0 ,0

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for inputs, targets in tepoch:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if dataset != "ToyADMOS":
                    outputs = outputs.view(outputs.size(0),-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                postfix = {}
                if (training_type == "classification"):
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    postfix["Acc"] = f"{100.0 * correct / total:.2f}%"

                postfix["Loss"] = f"{loss.item():.4f}"

                tepoch.set_postfix(postfix)
            return test(epoch, criterion, best_acc)

    def test(epoch, criterion, best_acc, log=False):
        # pass

        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            with tqdm(eval_loader, unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    if dataset != "ToyADMOS":
                        outputs = outputs.view(outputs.size(0),-1)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    postfix = {}
                    if (training_type == "classification"):
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        postfix["Acc"] = f"{100.0 * correct / total:.2f}%"
                    elif (training_type == "regression"):
                        pass

                    postfix["Loss"] = f"{loss.item():.4f}"

                    tepoch.set_postfix(postfix)
                    if log and tepoch.n == 1:
                        with open(os.path.join(log_dir, 'test_inference.txt'), 'a') as f:
                            # log input and output
                            for i in range(inputs.size(0)):
                                f.write(f'input: {inputs[i].cpu().numpy().tolist()}\n')
                                f.write(f'output: {outputs[i].cpu().numpy().tolist()}\n')
                        break

        if (training_type == "classification"):
            acc = 100. * correct / total
        elif (training_type == "regression"):
            acc = 1/test_loss
        # if acc > best_acc and retrain:
        if acc > best_acc:
            print('#### Exporting..')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(ckpt_dir, f'checkpoint_quant_bnfuse_fx.t7'))
            # model.to('cpu')
            dummy_input = torch.randn(input_shape, device=device)
            accuracy_str = f'{acc:.2f}'.replace('.', '_')
            exported_model = export_onnx_qcdq(model, args=dummy_input, export_path=onnx_dir + "/%s_bnfuse.onnx" % (log_name), opset_version=13)
            best_acc = acc
        # model.to(device)
        return best_acc


    def print_partial(model):

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                num_workers=1)

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        def get_conv_activation(name):
            def hook(model, input, output):
                activation[name] = input[0].detach()
            return hook

        for name, module in model.named_modules():
            t = type(module)
            t = t.__name__
            if 'relu' in t.lower():
                # print(t, name)
                module.register_forward_hook(get_activation(name))
            if 'conv' in t.lower():
                # print(t, name)
                module.register_forward_hook(get_conv_activation(name))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.view(outputs.size(0),-1)
                break
        return activation

    #print(model)
    #return 0
    
    if(not(pretrain)) :
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
        if (training_type == "classification"):
            criterion = torch.nn.CrossEntropyLoss()
        elif (training_type == "regression"):
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        print("#### TRAINING")
        best_acc = 0
        for epoch in range(0, max_epochs): 
            best_acc = train(epoch, criterion, optimizer, best_acc)
            lr_schedu.step(epoch)

    if(val) :
        if (training_type == "classification"):
            criterion = torch.nn.CrossEntropyLoss()
        elif (training_type == "regression"):
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(start_epoch, start_epoch+1):
            test(epoch, criterion, best_acc=1)


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
                # print(f'Bias correction iteration {i}')
                # Apply bias correction
                quant_model = bc.apply(quant_model, images)
        quant_model.apply(finalize_collect_stats)    
        return quant_model
  
    test(epoch, criterion, 1, log=True)
    #change not hand-written
    print("#### MERGE BN")
    
    #model = merge_conv_bn(model)
    fuse_layers(model)
    replace_layers(model,torch.nn.BatchNorm2d ,torch.nn.Identity())
    
    post_quant = 0
    if(post_quant) :
        model = calibrate_model(train_loader,model)
    if(val) :
        test(start_epoch, criterion, best_acc=0)
        print("#### RETRAINING") 
        retrain = 1
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(start_epoch, start_epoch+20): 
            if(not(post_quant)) :
                best_acc = train(epoch, criterion, optimizer, best_acc)
            else:
                best_acc = test(epoch, criterion, best_acc)
            lr_schedu.step(epoch)
    print('#### Finished Training.. with best_acc: %f' % (best_acc))
    print('#### Logging data from best checkpoint')
    ckpt = torch.load(os.path.join(ckpt_dir, f'checkpoint_quant_bnfuse_fx.t7'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    test(epoch, criterion, 1, log=True)

if __name__ == '__main__':
    main()
