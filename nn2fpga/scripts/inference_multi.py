import pynq
from pynq import Overlay
from pynq import allocate
from pynq import Clocks
from pynq import PL

import sys
import os
import numpy as np

from boards import hw_inference_multi_dma, ZCU102PowerSensor
from datasets import cifar10_dataloader, coco_dataloader, vw_dataloader, imagenet_dataloader, cifar10_4bit_dataloader, fotovoltaic_dataloader
from fotovoltaic import postprocess as fotovoltaic_postprocess
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

    PL.reset()
    supported_boards = ["KRIA", "ULTRA96v2", "ZCU102"]
    supported_datasets = ["cifar10", "vw", "coco", "imagenet", "fotovoltaic"]

    if len(sys.argv) < 5:
        print("Usage: inference.py <board> <dataset> <input_scale_factor> <frequency>")
        sys.exit(1)

    sel_board = sys.argv[1]
    if sel_board not in supported_boards:
        print(f"Error: selected {sel_board}, allowed board selection: {supported_boards}")
        sys.exit(1)

    sel_dataset = sys.argv[2]
    if sel_dataset not in supported_datasets:
        print(f"Error: selected {sel_dataset}, allowed dataset selection: {supported_datasets}")
        sys.exit(1)

    try:
        scale_factor = int(sys.argv[3])
        if scale_factor <= 0:
            raise ValueError
    except ValueError:
        print("Error: input scale factor must be a positive integer number")
        sys.exit(1)

    try:
        frequency = int(sys.argv[4])
        if frequency <= 0:
            raise ValueError
    except ValueError:
        print("Error: frequency (MHz) must be a positive integer number")
        sys.exit(1)

    batch_size = 1

    # Dataset and postprocess selector
    if sel_dataset == "cifar10":
        dataloader = cifar10_dataloader
        postprocess = cifar10_postprocess
    elif sel_dataset == "cifar10_4bit":
        dataloader = cifar10_4bit_dataloader
        postprocess = cifar10_postprocess
    elif sel_dataset == "vw":
        dataloader = vw_dataloader
        postprocess = vw_postprocess
    elif sel_dataset == "coco":
        dataloader = coco_dataloader
        postprocess = coco_postprocess
    elif sel_dataset == "imagenet":
        dataloader = imagenet_dataloader
        postprocess = imagenet_postprocess
    elif sel_dataset == "fotovoltaic":
        dataloader = fotovoltaic_dataloader
        postprocess = fotovoltaic_postprocess
        
    test_loader, buffer_dim = dataloader(batch_size)

    # Board-specific power sensor configuration
    board = {}
    pl_divisor = 10
    if sel_board == "ULTRA96v2":
        rails = pynq.get_rails()
        board["sensor"] = rails["INT"].power
        board["sensor_name"] = "INT_power"
        pl_divisor = 1500 // frequency

    elif sel_board == "KRIA":
        rails = pynq.get_rails()
        board["sensor"] = rails["power1"].power
        board["sensor_name"] = "power1_power"
        pl_divisor = 1000 // frequency

    elif sel_board == "ZCU102":
        board["sensor"] = ZCU102PowerSensor("INT_power", "W")
        board["sensor_name"] = "INT_power"
        pl_divisor = 1500 // frequency

    # ── Overlay load ─────────────────────────────────────────
    PL.reset()
    print("Loading overlay …", flush=True)
    ov = Overlay("./overlay/design_1.bit")
    Clocks._instance.PL_CLK_CTRLS[0].DIVISOR0 = pl_divisor
    print("Overlay loaded, PL clock set\n", flush=True)
    # batch_size = 202
    # ── DMA handles ──────────────────────────────────────────
    dma_0 = ov.axi_dma_0   # activations  + head-1
    dma_1 = ov.axi_dma_1   # weights once + head-2
    dma_2 = ov.axi_dma_2   # head-3 only
    dmas = [dma_0, dma_1, dma_2]

    # ── Allocate buffers ─────────────────────────────────────
    act_buf = allocate(shape=(batch_size * buffer_dim[0],), dtype=np.int8)
    print("Loading weights …", flush=True)
    # print(ov.ip_dict, flush=True)
    weights = np.load("overlay/uram.npy").astype(np.int8)
    w_buf   = allocate(shape=(weights.shape[0], ), dtype=np.int8)
    w_buf[:] = weights[:]

    # transfer weights once through dma_1
    # dma_1.sendchannel.transfer(w_buf)
    # dma_1.sendchannel.wait()
    # print("Weights loaded into IP\n", flush=True)
    
    print("Starting DMA transfer of weights", flush=True)
    dma_1.sendchannel.transfer(w_buf)
    print("Waiting for DMA transfer completion", flush=True)
    dma_1.sendchannel.wait()
    print("DMA transfer completed", flush=True)


    # import time

    # dma_1.sendchannel.transfer(w_buf)
    # start_time = time.time()
    # timeout = 5  # seconds

    # while not dma_1.sendchannel.wait(0):  # non-blocking wait
    #     if time.time() - start_time > timeout:
    #         print("DMA transfer timeout")
    #         break
    #     time.sleep(0.01)
    
    
    head1_buf = allocate(shape=(batch_size, 18*40*40), dtype=np.int8)
    head2_buf = allocate(shape=(batch_size, 18*80*80), dtype=np.int8)
    head3_buf = allocate(shape=(batch_size, 18*20*20), dtype=np.int8)
    out_bufs  = [head1_buf, head2_buf, head3_buf]
    outputs = None
    # ── Inference ────────────────────────────────────────────
    hw_inference_multi_dma(
        test_loader,
        act_buf,
        out_bufs,
        dmas,
        batch_size,
        scale_factor,
        postprocess,
        board,
        outputs
    )

    # ── Optional dump to files for later inspection ─────────
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/head1.npy", np.copy(head1_buf))
    np.save("outputs/head2.npy", np.copy(head2_buf))
    np.save("outputs/head3.npy", np.copy(head3_buf))
    # save also as .bin the head1, head2, head3
    head1_buf.tofile("outputs/head1.bin")
    head2_buf.tofile("outputs/head2.bin")
    head3_buf.tofile("outputs/head3.bin")
    
    print("Saved raw outputs to ./outputs/*.npy")

    # ── Cleanup ──────────────────────────────────────────────
    for buf in (act_buf, w_buf, head1_buf, head2_buf, head3_buf):
        del buf
