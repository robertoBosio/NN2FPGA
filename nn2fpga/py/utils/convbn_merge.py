import torch
from brevitas.nn.utils import merge_bn

def replace_layers(model, old, new) :
    for name, module in model.named_children() :
        if(len(list(module.children()))) > 0 :
            replace_layers(module, old, new)
        if isinstance(module, old) :
            setattr(model, name, new)

def fuse_layers(model) :
    for name, module in model.named_children() :
        if(len(list(module.children()))) > 0 :
            fuse_layers(module) 
        if(isinstance(module, torch.nn.Conv2d)) :
            conv_tmp = module
            name_tmp = name
        if(isinstance(module, torch.nn.BatchNorm2d)) :
            bn_tmp = module
            merge_bn(conv_tmp, bn_tmp)
            setattr(model, name_tmp, conv_tmp)
