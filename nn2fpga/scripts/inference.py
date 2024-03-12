import pynq
from pynq import Overlay
from pynq import allocate

import sys
import os
import numpy as np

from boards import kria_inference, ultra96_inference, zcu102_inference
from datasets import cifar10_dataloader, coco_dataloader, vw_dataloader, imagenet_dataloader
from coco import postprocess as coco_postprocess
from cifar10 import postprocess as cifar10_postprocess
from vw import postprocess as vw_postprocess
from imagenet import postprocess as imagenet_postprocess

def print_sorted_nested_dict(dictionary, indent=0):
    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_sorted_nested_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


if __name__ == "__main__":
    # Uploading all data

    supported_boards = ["KRIA", "ULTRA96v2", "ZCU102"]
    supported_datasets = ["cifar10", "vw", "coco", "imagenet"]
    supported_uram_storage = [0, 1]

    if (len(sys.argv) < 3):
        print(
            "Wrong number of arguments: inference.py board dataset uram_storage"
        )
        sys.exit(0)

    sel_board = sys.argv[1]
    if (sel_board not in supported_boards):
        print(f"Error: selected {sel_board}, allowed board selection: {supported_boards}")
        sys.exit(0)

    sel_dataset = sys.argv[2]
    if (sel_dataset not in supported_datasets):
        print(f"Error: selected {sel_dataset}, allowed dataset selection: {supported_datasets}")
        sys.exit(0)

    sel_uram_storage = int(sys.argv[3])
    if (sel_uram_storage not in supported_uram_storage):
        print(f"Error: selected {sel_uram_storage}, allowed uram_storage selection: {supported_uram_storage}")
        sys.exit(0)

    off_chip_memory = False

    batch_size = 400

    if (sel_dataset == "cifar10"):
        dataloader = cifar10_dataloader
        postprocess = cifar10_postprocess
    if (sel_dataset == "vw"):
        dataloader = vw_dataloader
        postprocess = vw_postprocess
    if (sel_dataset == "coco"):
        dataloader = coco_dataloader
        postprocess = coco_postprocess
    if (sel_dataset == "imagenet"):
        dataloader = imagenet_dataloader
        postprocess = imagenet_postprocess

    test_loader, buffer_dim = dataloader(batch_size)

    print("Loading overlay")
    overlay = Overlay('./overlay/design_1.bit')
    
    print("Loaded overlay")
    dma = overlay.axi_dma_0
    if (sel_uram_storage == 1):
        print("Loading URAM")
        dma_uram = overlay.axi_dma_1
        uram_vector = np.load("overlay/uram.npy")
        uram_buffer = allocate(shape=(uram_vector.shape[0], ), dtype=np.int8)
        uram_buffer[:] = uram_vector[:]
        dma_uram.sendchannel.transfer(uram_buffer)
        dma_uram.sendchannel.wait()
    print("Loaded URAM", flush=True)

    if (off_chip_memory):
        network = overlay.Network_0
    else:
        network = None

    in_buffer = allocate(shape=(batch_size*buffer_dim[0], ), dtype=np.int8)
    out_buffer = allocate(shape=(batch_size, buffer_dim[1], ), dtype=np.int8)

    #################################Inference##################################

    if (sel_board == "ULTRA96v2"):
        inference = ultra96_inference
    if (sel_board == "KRIA"):
        inference = kria_inference
    if (sel_board == "ZCU102"):
        inference = zcu102_inference

    inference(
        test_loader,
        in_buffer,
        out_buffer,
        dma,
        batch_size,
        off_chip_memory,
        network,
        postprocess,
        sel_uram_storage
    )

    if (sel_uram_storage == 1):
        del uram_buffer

    del in_buffer, out_buffer
