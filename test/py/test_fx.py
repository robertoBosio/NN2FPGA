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
from brevitas.export import export_onnx_qcdq
from tqdm import tqdm
from utils.datasets import get_dataset
import torchsummary

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

    start_epoch = 0

    log_dir = os.path.join(root_dir, 'logs', log_name)
    ckpt_dir = os.path.join(root_dir, 'ckpt', log_name)
    onnx_dir = os.path.join(root_dir, 'onnx', log_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)


    train_dataset, eval_dataset, input_shape = get_dataset(dataset, cifar=cifar)

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                            num_workers=num_workers)

    # if 'cuda' in device:
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True
    #     #print("no cuda")
    # exit(-1)
    print('#### Building model..')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = resnet8(weight_bits=Wbits,act_bits=Abits).to('cuda:0')
    # model = MobileNetV1(num_filters=8, num_classes=2).to(device)
    # model = QuantizedCifar10Net().to(device)
    # model = Autoencoder(input_shape[1], weight_bits=Wbits, act_bits=Abits).to('cuda:0')

    model.to(device)

    start_epoch = 0
    # print('#### Load last checkpoint data')
    model = model.to(device)

    def test(best_acc):
        # pass
        correct, total = 0, 0
        # hook the intermediate tensors with their name
        os.system("mkdir -p tmp/logs/%s" % (log_name))
        os.system("rm -rf tmp/logs/%s/feature.txt" % (log_name))
        def get_activation(name):
            def hook(module, input, output):
                with open("tmp/logs/%s/feature.txt" % (log_name), "a+") as f:
                    f.write(str(name) + " " + str(input[0].shape) + " " + str(output[0].shape) + "\n")
                    f.write("input" + "\n")
                    try:
                        for hh in range(input[0].shape[2]):
                            for ww in range(input[0].shape[3]):
                                for cc in range(input[0].shape[1]):
                                    f.write(str(input[0][0][cc][hh][ww].detach().cpu().numpy()) + " ")
                                f.write("\n")
                    except:
                        f.write("input" + "\n")
                        f.write(str(input[0].detach().cpu()) + "\n")

                    try:
                        f.write("output" + "\n")
                        for hh in range(output[0].shape[1]):
                            for ww in range(output[0].shape[2]):
                                for cc in range(output[0].shape[0]):
                                    f.write(str(output[0][cc][hh][ww].detach().cpu().numpy()) + " ")
                                f.write("\n")
                    except:
                        f.write("output" + "\n")
                        f.write(str(output[0].detach().cpu()) + "\n")
            return hook

        def get_activation_quant(name):
            def hook(module, input, output):
                with open("tmp/logs/%s/feature.txt" % (log_name), "a+") as f:
                    f.write(str(name) + " " + str(input[0].shape) + " " + str(output[0].shape) + "\n")
                    f.write("input" + "\n")
                    for hh in range(input[0].shape[2]):
                        for ww in range(input[0].shape[3]):
                            for cc in range(input[0].shape[1]):
                                f.write(str(input[0].value[0][cc][hh][ww].detach().cpu().numpy()) + " ")
                            f.write("\n")

                    f.write("output" + "\n")
                    for hh in range(output[0].shape[2]):
                        for ww in range(output[0].shape[3]):
                            for cc in range(output[0].shape[1]):
                                f.write(str(output[0][0][cc][hh][ww].detach().cpu().numpy()) + " ")
                            f.write("\n")
            return hook

        def get_weight_quant(name):
            def hook(module, input, output):
                with open("tmp/logs/%s/feature.txt" % (log_name), "a+") as f:
                    f.write(str(name) + " " + str(input[0].shape) + " " + str(output[0].shape) + "\n")
                    f.write("input" + "\n")
                    for hh in range(input[0].shape[3]):
                        for ww in range(input[0].shape[2]):
                            for cc in range(input[0].shape[1]):
                                for occ in range(output[0].shape[0]):
                                    f.write(str(input[0][occ][cc][hh][ww].detach().cpu().numpy()) + " ")
                                f.write("\n")

                    f.write("output" + "\n")
                    for occ in range(output[0].shape[0]):
                        for cc in range(output[0].shape[1]):
                            for hh in range(output[0].shape[2]):
                                for ww in range(output[0].shape[3]):
                                    f.write(str(output[0][occ][cc][hh][ww].detach().cpu().numpy()) + " ")
                            f.write("\n")
            return hook

        def get_bias_quant(name):
            def hook(module, input, output):
                with open("tmp/logs/%s/feature.txt" % (log_name), "a+") as f:
                    f.write(str(name) + " " + str(input[0].shape) + " " + str(output[0].shape) + "\n")
                    f.write("input" + "\n")
                    for occ in range(output[0].shape[0]):
                        f.write(str(input[0][occ].detach().cpu().numpy()) + " ")
                        f.write("\n")

                    f.write("output" + "\n")
                    for occ in range(output[0].shape[0]):
                        f.write(str(output[0][occ].detach().cpu().numpy()) + " ")
                        f.write("\n")
            return hook

        for name, module in model.named_modules():
            print(name)
            if isinstance(module, torch.nn.MaxPool2d):
                module.register_forward_hook(get_activation(name))
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(get_activation(name))
            if "add" in name:
                module.register_forward_hook(get_activation(name))
            if name == "conv1.input_quant":
                module.register_forward_hook(get_activation_quant(name))
            if name == "conv1.weight_quant":
                module.register_forward_hook(get_weight_quant(name))
            if name == "layer1.0.conv1.weight_quant":
                module.register_forward_hook(get_weight_quant(name))
            if name == "conv1.bias_quant":
                module.register_forward_hook(get_bias_quant(name))
            

        model.eval()
        with torch.no_grad():
            with tqdm(eval_loader, unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # normalize the input using the same mean and std of training set
                    outputs = model(inputs)
                    outputs = outputs.view(outputs.size(0),-1)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    tepoch.set_postfix({"Acc": f"{100.0 * correct / total:.2f}%"})
                    exit()

        acc = 100. * correct / total
        # if acc > best_acc and retrain:
        # model.to('cpu')
        if acc > best_acc:
            best_acc = acc
        print('#### Exporting..')
        # model.to('cpu')
        dummy_input = torch.randn(input_shape, device=device)
        accuracy_str = f'{acc:.2f}'.replace('.', '_')
        exported_model = export_onnx_qcdq(model, args=dummy_input, export_path=onnx_dir + "/%s_bnfuse.onnx" % (log_name), opset_version=13)
        best_acc = acc
        return best_acc
        # model.to(device)

    best_acc = 0  # best test accuracy
    print("#### Merge BN..")
    #model = merge_conv_bn(model)
    fuse_layers(model)
    replace_layers(model,torch.nn.BatchNorm2d ,torch.nn.Identity())
    print("#### Loading pretrain model..")
    ckpt = torch.load(os.path.join(ckpt_dir, f'checkpoint_quant_bnfuse_fx.t7'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    print('#### Testing..')
    best_acc = test(best_acc)
    print('#### Finished Testing .. with best_acc: %f' % (best_acc))

   
if __name__ == '__main__':
    main()
