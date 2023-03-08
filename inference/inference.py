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
BATCH_SIZE = 200
OFF_CHIP_MEMORY = False
#BATCH_SIZE = 1

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
#dbg = overlay.axi_gpio_0
# network = overlay.Network_0
if (OFF_CHIP_MEMORY):
	# TODO: add object memory management
	MemoryManagement1 = overlay.i_data_1
	MemoryManagement2 = overlay.i_data_2

ICH = 3
#IH = 16
#IW = 16
IH = 32
IW = 32
OCH = 10
OH = 1
OW = 1

#in_buffer  = allocate(shape=(BATCH_SIZE*IH*IW*ICH, ), dtype=np.uint8)
in_buffer  = allocate(shape=(BATCH_SIZE*IH*IW*ICH, ), dtype=np.uint8)
out_buffer = allocate(shape=(BATCH_SIZE, OH*OW*OCH, ), dtype=np.int32)
if (OFF_CHIP_MEMORY):
	weights = np.load("tmp/weights.npy")
	weights_buffer = allocate(shape=(weights.shape[0]), dtype=np.int8)
	MemoryManagement1.i_data.write("0x0", weights_buffer.physical_address())
	MemoryManagement2.i_data.write("0x0", weights_buffer.physical_address())

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

for batch, (features, results) in enumerate(test_loader):
    np_features = (np.asarray(torch.permute(features, (0, 2, 3, 1))).flatten()*255).astype(np.uint8)
    in_buffer[:] = np_features

    start = time.time()    
    dma.recvchannel.transfer(out_buffer)
    dma.sendchannel.transfer(in_buffer)

    #while(dbg.read(0x0) == 0):
    #    time.sleep(1)
    #    print(dbg.read(0x0))

    #print(dma._registers)
    #print(dma.read(0))
    #print(bin(dma.read(4)))
    #print(dma.buffer_max_size)
    print("WAITING SENDING IMAGES")
    dma.sendchannel.wait()

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
