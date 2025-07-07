import pynq
from pynq import GPIO
from pynq import Clocks

import time
import torch
import numpy as np
import os

class ZCU102PowerSensor:
    """A class representing a power sensor on the ZCU102 board 

    Attributes
    ----------
    name : str
        The name of the sensor
    value : float
        The current value of the sensor

    """
    def __init__(self, name, unit):
        """Create a new sensor object reading values from a file

        Parameters
        ----------
        file_path : str
            Path to the file containing sensor values
        name : str
            Name of the sensor
        unit : str
            Unit to append to the value when creating a string representation

        """

        def populate_ina_array(self):
            directory = "/sys/class/hwmon/"

            for entry in os.listdir(directory):
                if entry.startswith("."):
                    continue

                fname_buff = os.path.join(directory, entry, "name")

                with open(fname_buff, "r") as fptr:
                    buffer = fptr.read(10)

                if buffer.startswith("ina"):
                    fname_buff = fname_buff[:-5]
                    ina = {}
                    ina["current_path"] = os.path.join(fname_buff, "curr1_input")
                    ina["voltage_path"] = os.path.join(fname_buff, "in2_input")
                    ina["rail"] = None

                    # search fname_buff in railname_arr
                    for railname, sensor_name in self.railname_arr.items():
                        if sensor_name in buffer:
                            ina["rail"] = railname
                            break

                    if ina["rail"] is not None:
                        if ina["rail"] in ["VCCINT", "VCCBRAM", "VCCAUX", "VCC1V2", "VCC3V3"]:
                            self.inaspl.append(ina)
                        elif ina["rail"] in ["VCCPSINTFP", "VCCINTLP", "VCCPSAUX", "VCCPSPLL", "VCCPSDDR", "VCCOPS", "VCCOPS3", "VCCPSDDRPLL"]:
                            self.inasps.append(ina)
                        elif ina["rail"] in ["MGTRAVCC", "MGTRAVTT", "MGTAVCC", "MGTAVTT"]:
                            self.inasmgt.append(ina)

        
        self._value = 0  # Default value
        self.name = name
        self._unit = unit
        self.parents = tuple()
        self.railname_arr = {
            "VCCPSINTFP" : "u76",
            "VCCINTLP" : "u77",
            "VCCPSAUX" : "u78",
            "VCCPSPLL" : "u87",
            "MGTRAVCC" : "u85",
            "MGTRAVTT" : "u86",
            "VCCPSDDR" : "u93",
            "VCCOPS" : "u88",
            "VCCOPS3" : "u15",
            "VCCPSDDRPLL" : "u92",
            "VCCINT" : "u79",
            "VCCBRAM" : "u81",
            "VCCAUX" : "u80",
            "VCC1V2" : "u84",
            "VCC3V3" : "u16",
            "VADJ_FMC" : "u65",
            "MGTAVCC" : "u74",
            "MGTAVTT" : "u75"}
        
        # Three categories of power rails
        self.inaspl = []
        self.inasps = []
        self.inasmgt = []
        populate_ina_array(self)

    @property
    def value(self):
        """Read the current value of the sensor from the file

        """
        self._value = 0
        for ina in self.inaspl:
            with open(ina["current_path"], "r") as fptr:
                curr = float(fptr.read())
            with open(ina["voltage_path"], "r") as fptr:
                volt = float(fptr.read())
            self._value += (curr * volt) / 1000000.0
        
        for ina in self.inasps:
            with open(ina["current_path"], "r") as fptr:
                curr = float(fptr.read())
            with open(ina["voltage_path"], "r") as fptr:
                volt = float(fptr.read())
            self._value += (curr * volt) / 1000000.0
        
        for ina in self.inasmgt:
            with open(ina["current_path"], "r") as fptr:
                curr = float(fptr.read())
            with open(ina["voltage_path"], "r") as fptr:
                volt = float(fptr.read())
            self._value += (curr * volt) / 1000000.0
        
        return self._value
    
    def get_value(self, parents=None):
        return self.value

    def __repr__(self):
        return f"ZCU102PowerSensor(name={self.name}, value={self.value}{self._unit})"

def hw_inference(
    test_loader,
    in_buffer,
    out_buffer,
    dma,
    batch_size,
    scale_factor,
    postprocess,
    board,
    tot_batches = None
):
    """ Perform inference on the given board. """

    if tot_batches is None:
        tot_batches = len(test_loader)

    recorder = pynq.DataRecorder(board["sensor"])

    start_exe = time.time()    

    total_time = 0
    accuracy = 0
    mean_power = 0
    total_energy = 0

    for batch, (features, results) in enumerate(test_loader):
        np_features = (np.asarray(torch.permute(features, (0, 2, 3, 1))).flatten() * scale_factor).astype(np.int8)
        in_buffer[:] = np_features[:]
        
        recorder.reset()
        recorder.record(0.01)
        start = time.time()    
        dma.recvchannel.transfer(out_buffer)
        dma.sendchannel.transfer(in_buffer)

        dma.sendchannel.wait()
        dma.recvchannel.wait()
        end = time.time()
        recorder.stop()

        accuracy, accuracy_batch = postprocess(out_buffer, results, accuracy, batch_size)
        batch_time = end - start
        total_time += batch_time
        batch_power = recorder.frame[board["sensor_name"]].mean()
        batch_power_points = len(recorder.frame[board["sensor_name"]])
        # batch_power = 0
        # batch_power_points = 0
        batch_energy = batch_power * batch_time
        mean_power += batch_power
        total_energy += batch_energy
        
        print(f"Batch {batch} time: {batch_time:.2f}s, mean power: {batch_power:.2f}W on {batch_power_points} points.", flush=True)
        
        if (batch == tot_batches - 1):
            break 
        
    mean_power /= tot_batches

    with open("overlay/results.txt", "w+") as fd:
        fd.write(f"Total time: {total_time:.2f} seconds\n")
        fd.write(f"Mean power: {mean_power:.2f} W\n")
        fd.write(f"Total energy: {total_energy:.2f} J\n")
        fd.write(f"Batch size {batch_size}\n")
        fd.write(f"Images nums: {batch_size * tot_batches}\n")
        fd.write(f"Fps: {(batch_size * tot_batches / total_time)}\n")
        fd.write(f"Accuracy: {(accuracy / (batch_size * tot_batches))}\n")

    print(f"\nTotal time: {total_time:.2f} seconds.")
    print(f"Mean power: {mean_power:.2f} W")
    print(f"Total energy: {total_energy:.2f} J")
    print(f"Batch size {batch_size}")
    print(f'Tot images: {batch_size * tot_batches}')
    print(f"FPS: {batch_size * tot_batches / total_time:.2f}")
    print(f'Accuracy: {accuracy / (batch_size * tot_batches):.4f}')
    
    end_exe = time.time()
    print(f"Total execution time: {end_exe - start_exe:.2f} seconds")

def hw_inference_multi_dma(
    test_loader,
    act_buf,                  # activations  (dma_0  send)
    out_bufs,                 # [head1, head2, head3]
    dmas,                     # [dma_0, dma_1, dma_2]
    batch_size,
    scale_factor,
    postprocess_fn,           # receives out_bufs
    board,
    outputs=None,            # list to store outputs
    tot_batches=None
):
    out_head1, out_head2, out_head3 = [], [], []

    if tot_batches is None:
        # tot_batches = len(test_loader)
        tot_batches = 202
    dma_0, dma_1, dma_2 = dmas
    head1_buf, head2_buf, head3_buf = out_bufs

    recorder = pynq.DataRecorder(board["sensor"])
    wall_start = time.time()
    total_time = 0.0
    mean_power_acc = 0.0
    total_energy = 0.0
    img_counter = 0





    # for batch_idx, (paths, images, im0s, splits) in enumerate(test_loader):
    #     print(f"Processing batch {batch_idx + 1}/{tot_batches} with {len(images)} images", flush=True)
    #     # ── Preprocess batch: (B, 3, H, W) → flat int8 ────────────────────────────────
    #     print("Normalizing and converting to int8...", flush=True)
    #     images = images.detach().cpu().numpy()
    #     print(f"[DEBUG] Numpy shape: {images.shape}, dtype: {images.dtype}", flush=True)

    #     images = images.transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    #     np_feat = (images * scale_factor).flatten().astype(np.int8)

    #     print(f"[DEBUG] np_feat size: {len(np_feat)}, act_buf size: {len(act_buf)}", flush=True)

    #     assert len(np_feat) <= len(act_buf), "❌ act_buf is too small for the current batch"

    #     act_buf[:len(np_feat)] = np_feat
    #     print(f"Copied {len(np_feat)} bytes to act_buf", flush=True)

    #     # ─ DMA Setup ─
    #     print("Setting up DMA transfers...", flush=True)
    #     recorder.reset()
    #     recorder.record(0.01)
    #     t0 = time.time()

    #     dma_0.recvchannel.transfer(head1_buf)
    #     dma_1.recvchannel.transfer(head2_buf)
    #     dma_2.recvchannel.transfer(head3_buf)

    #     dma_0.sendchannel.transfer(act_buf)
    #     # print("Input DMA transfer started", flush=True)

    #     dma_0.sendchannel.wait()
    #     # print("Input DMA done", flush=True)

    #     dma_0.recvchannel.wait()
    #     # print("Output DMA head1 done", flush=True)

    #     dma_1.recvchannel.wait()
    #     # print("Output DMA head2 done", flush=True)

    #     dma_2.recvchannel.wait()
    #     # print("Output DMA head3 done", flush=True)

    #     t1 = time.time()
    #     recorder.stop()
    #     print("All transfers completed", flush=True)
    #     # ── Postprocess and store ─────────────────────────────────────────────────────
    #     postprocess_fn(out_bufs, batch_size)

    #     # out_head1.append(head1_buf.copy())
    #     # out_head2.append(head2_buf.copy())
    #     # out_head3.append(head3_buf.copy())
    #     for bi in range(len(images)):  # handle partial last batch
    #         out_head1.append(np.copy(head1_buf[bi]))
    #         out_head2.append(np.copy(head2_buf[bi]))
    #         out_head3.append(np.copy(head3_buf[bi]))
    #     # ── Metrics ───────────────────────────────────────────────────────────────────
    #     batch_time = t1 - t0
    #     total_time += batch_time
    #     pwr = recorder.frame[board["sensor_name"]].mean()
    #     mean_power_acc += pwr
    #     total_energy += pwr * batch_time

    #     print(f"[Batch {batch_idx:4d}] time {batch_time:6.3f}s ─ power {pwr:6.2f} W", flush=True)

    #     if batch_idx == tot_batches - 1:
    #         break

    
    batch_idx = 0
    for path, images, im0, split, _ in test_loader:
        # ── Pack & scale activations (NHWC→flat int8) ─────────────────────────
        # np_feat = (np.asarray(images.permute(0, 2, 3, 1)).flatten()
        #            * scale_factor).astype(np.int8)
        images = np.transpose(images, (1, 2, 0)).astype(np.float32) / 255.0
        
        # check fisrt tree values
        images = images * scale_factor
        #clip to 127
        images = np.clip(images, -128, 127)
        # print first 3 values
        print(f"[DEBUG] First 3 values of images: {images.flatten()[:3]}")
        np_feat = images.flatten().astype(np.int8)
        # print first 3 values
        print(f"[DEBUG] First 3 values of np_feat: {np_feat[:3]}")
        act_buf[:len(act_buf)] = np_feat[:len(act_buf)]

        # ── Start power log & DMA transactions ───────────────────────────────
        recorder.reset(); recorder.record(0.01)
        t0 = time.time()

        # receive on all three heads
        dma_0.recvchannel.transfer(head1_buf)
        # print("Waiting for head1 transfer to complete", flush=True)
        dma_1.recvchannel.transfer(head2_buf)
        # print("Waiting for head2 transfer to complete", flush=True)
        dma_2.recvchannel.transfer(head3_buf)
        # print("Waiting for head3 transfer to complete", flush=True)

        # send activations (only dma_0)
        dma_0.sendchannel.transfer(act_buf) 
        # print("Waiting for activations transfer to complete", flush=True)
        # wait
        dma_0.sendchannel.wait()
        # print("Waiting for head1 transfer to complete", flush=True)
        dma_0.recvchannel.wait()
        # print("Waiting for head2 transfer to complete", flush=True)
        dma_1.recvchannel.wait()
        # print("Waiting for head3 transfer to complete", flush=True)
        dma_2.recvchannel.wait()
        # print("All transfers completed", flush=True)

        t1 = time.time()
        recorder.stop()

        # ── Optional user post-processing  ───────────────────────────────────
        postprocess_fn(out_bufs, batch_size)
        # concatenate outputs if provided in 3 heads
        out_head1.append(head1_buf.copy())
        out_head2.append(head2_buf.copy())
        out_head3.append(head3_buf.copy())
        # ── Metrics  ─────────────────────────────────────────────────────────
        batch_time   = t1 - t0
        total_time  += batch_time
        pwr          = recorder.frame[board["sensor_name"]].mean()
        total_energy += pwr * batch_time
        mean_power_acc += pwr

        print(f"[Batch {batch_idx:4d}] time {batch_time:6.3f}s ─ "
              f"power {pwr:6.2f} W", flush=True)
        

        
        if batch_idx == tot_batches - 1:
            break
        batch_idx += 1
        
        
        
    mean_power = mean_power_acc / tot_batches
    print("\n──────────  Summary  ──────────")
    print(f"Images      : {batch_size * tot_batches}")
    print(f"Total time  : {total_time:.2f} s")
    print(f"FPS         : {(batch_size * tot_batches) / total_time:.2f}")
    print(f"Mean power  : {mean_power:.2f} W")
    print(f"Energy      : {total_energy:.2f} J")
    print(f"Wall-clock  : {time.time() - wall_start:.2f} s")
    print("────────────────────────────────")
    
    np.concatenate(out_head1, axis=0).tofile("outputs/head1_tot.bin")
    np.concatenate(out_head2, axis=0).tofile("outputs/head2_tot.bin")
    np.concatenate(out_head3, axis=0).tofile("outputs/head3_tot.bin")
    print("Saved total outputs to ./outputs/head*_tot.bin")
