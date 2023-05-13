import pynq
from pynq import Overlay
from pynq import allocate

import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import numpy as np
import time

from pynq import GPIO

CIFAR10_DIRECTORY = './data'
os.system('mkdir -p %s' % CIFAR10_DIRECTORY)

test_data = torchvision.datasets.CIFAR10(
    CIFAR10_DIRECTORY,
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.float(),
    ])
)

# Uploading all data
#BATCH_SIZE = len(test_data)
BATCH_SIZE = 1
OFF_CHIP_MEMORY = False

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=1
)

print("LOADING OVERLAY")
overlay = Overlay('./tmp/design_1.bit')
print("LOADED OVERLAY")
print("AVAILABLE OVERLAY PERIPHERALS")
dma = overlay.axi_dma_0
if (OFF_CHIP_MEMORY):
    network = overlay.Network_0
#if (OFF_CHIP_MEMORY):
    # TODO: add object memory management
    #MemoryManagement = overlay.MemoryManagement_0

ICH = 3
IH = 32
IW = 32
OCH = 10
OH = 1
OW = 1

in_buffer  = allocate(shape=(BATCH_SIZE*IH*IW*ICH, ), dtype=np.uint8)
#in_buffer  = allocate(shape=(BATCH_SIZE*IH*IW*ICH+1*16*16, ), dtype=np.uint8)
out_buffer = allocate(shape=(BATCH_SIZE, OH*OW*OCH, ), dtype=np.int32)
if (OFF_CHIP_MEMORY):
    weights0 = np.load("tmp/weights_conv0_weight.npy")
    weights1 = np.load("tmp/weights_conv1_weight.npy")
    weights2 = np.load("tmp/weights_conv2_weight.npy")
    weights3 = np.load("tmp/weights_conv3_weight.npy")
    weights4 = np.load("tmp/weights_conv4_weight.npy")
    weights5 = np.load("tmp/weights_fc_weight.npy")
    weights0_buffer = allocate(shape=(weights0.shape[0], ), dtype=np.int8)
    weights1_buffer = allocate(shape=(weights1.shape[0], ), dtype=np.int8)
    weights2_buffer = allocate(shape=(weights2.shape[0], ), dtype=np.int8)
    weights3_buffer = allocate(shape=(weights3.shape[0], ), dtype=np.int8)
    weights4_buffer = allocate(shape=(weights4.shape[0], ), dtype=np.int8)
    weights5_buffer = allocate(shape=(weights5.shape[0], ), dtype=np.int8)
    #weights_buffer[:] = 0
    #print(weights_buffer)
    #print(weights_buffer.physical_address)
    #print(weights_buffer)
    #weights_buffer.flush()
    network.write(0x14, weights0_buffer.physical_address)
    network.write(0x18, 0)
    network.write(0x20, weights1_buffer.physical_address)
    network.write(0x24, 0)
    network.write(0x2c, weights2_buffer.physical_address)
    network.write(0x30, 0)
    network.write(0x38, weights3_buffer.physical_address)
    network.write(0x3c, 0)
    network.write(0x44, weights4_buffer.physical_address)
    network.write(0x48, 0)
    network.write(0x50, weights5_buffer.physical_address)
    network.write(0x54, 0)
    weights0_buffer[:] = weights0[:]
    weights1_buffer[:] = weights1[:]
    weights2_buffer[:] = weights2[:]
    weights3_buffer[:] = weights3[:]
    weights4_buffer[:] = weights4[:]
    weights5_buffer[:] = weights5[:]

#################################Inference##################################
interval_time = 0
total_time = 0
total_energy = 0
result = list()

rails = pynq.get_rails()

#recorder = pynq.DataRecorder(rails["5V"].power)

#with recorder.record(0.05): 

accuracy = 0
NUM_BATCH = len(test_loader)

if (OFF_CHIP_MEMORY):
    network.write(0x0, 0x81)
for batch, (features, results) in enumerate(test_loader):
    np_features = (np.asarray(torch.permute(features, (0, 2, 3, 1))).flatten()*255).astype(np.uint8)
    in_buffer[0:BATCH_SIZE*IH*IW*ICH] = np_features[0:BATCH_SIZE*IH*IW*ICH]

    print("SENDING DATA")
    start = time.time()    
    dma.recvchannel.transfer(out_buffer)
    dma.sendchannel.transfer(in_buffer)

    dma.sendchannel.wait()
    #dma.sendchannel.transfer(in_buffer)

    dma.recvchannel.wait()


    end = time.time()
    batch_time = end - start
    total_time += batch_time

    predicted = np.argmax(np.asarray(out_buffer[:]), axis=-1)
    accuracy_batch = np.equal(predicted, results)
    accuracy_batch = accuracy_batch.sum()
    accuracy += accuracy_batch

    print("Total time:", batch_time, "seconds")
    print("Inference time:", batch_time/BATCH_SIZE, "seconds")
    print("Batch accuracy:", accuracy_batch/BATCH_SIZE)
if (OFF_CHIP_MEMORY):
    network.write(0x0, 0x0)

# # Energy measurements    
# energy = recorder.frame["5V_power"].mean() * t    
# # energy = 0

# total_energy = energy
print("Total time:", total_time, "seconds")
# print("Total energy:", total_energy, "J")
print("Batch size %d" % BATCH_SIZE)
print('images nums: {} .'.format(BATCH_SIZE*NUM_BATCH))
print('fps: {} .'.format(BATCH_SIZE * NUM_BATCH / total_time))
print('Accuracy: {} .'.format(accuracy / (BATCH_SIZE * NUM_BATCH)))
