import pynq
from pynq import GPIO
from pynq import Clocks

import time
import torch
import numpy as np

def kria_inference(
    test_loader,
    in_buffer,
    out_buffer,
    dma,
    batch_size,
    off_chip_memory,
    network,
    postprocess,
    uram_storage
):

    NUM_BATCH = len(test_loader)

    Clocks._instance.PL_CLK_CTRLS[0].DIVISOR0 = 4

    # Clocks.fclk0_mhz = 200
    if (off_chip_memory):
        network.write(0x0, 0x81)

    for i in range(1):
        total_time = 0
        accuracy = 0

        for batch, (features, results) in enumerate(test_loader):
            np_features = (np.asarray(torch.permute(features, (0, 2, 3, 1))).flatten()*255).astype(np.uint8)
            in_buffer[:] = np_features[:]

            # print("SENDING DATA")
            start = time.time()    
            dma.recvchannel.transfer(out_buffer)
            dma.sendchannel.transfer(in_buffer)

            dma.sendchannel.wait()

            dma.recvchannel.wait()

            end = time.time()
            batch_time = end - start
            total_time += batch_time

            accuracy, accuracy_batch = postprocess(out_buffer, results, accuracy, batch_size)

            # print("Total time:", batch_time, "seconds")
            # print("Inference time:", batch_time/BATCH_SIZE, "seconds")

        if (off_chip_memory):
            network.write(0x0, 0x0)

        with open("overlay/results.txt", "w+") as fd:
            # total_energy = energy
            fd.write("Total time: %f seconds" % total_time)
            # fd.write("Total power: %f W" % mean_power)
            # fd.write("Total energy: %f J" % total_energy)
            fd.write("Batch size %d" % batch_size)
            fd.write('images nums: {} .'.format(batch_size*NUM_BATCH))
            fd.write('fps: {} .'.format(batch_size * NUM_BATCH / total_time))
            fd.write('Accuracy: {} .'.format(accuracy / (batch_size * NUM_BATCH)))

            # total_energy = energy
            print("Total time:", total_time, "seconds")
            # print("Total energy:", mean_power , "W")
            # print("Total energy:", total_energy, "J")
            print("Batch size %d" % batch_size)
            print('images nums: {} .'.format(batch_size*NUM_BATCH))
            print('fps: {} .'.format(batch_size * NUM_BATCH / total_time))
            print('Accuracy: {} .'.format(accuracy / (batch_size * NUM_BATCH)))

def zcu102_inference(
    test_loader,
    in_buffer,
    out_buffer,
    dma,
    batch_size,
    off_chip_memory,
    network,
    postprocess,
    uram_storage
):
    
        NUM_BATCH = len(test_loader)
        MAX_BATCH = 1
        Clocks._instance.PL_CLK_CTRLS[0].DIVISOR0 = 15
        start_exe = time.time()    
        time_to_copy = 0
        time_post_proc = 0
        time_enum = 0
    
        # Clocks.fclk0_mhz = 200
        if (off_chip_memory):
            network.write(0x0, 0x81)
    
        for i in range(1):
            total_time = 0
            accuracy = 0
    
            for batch, (features, results) in enumerate(test_loader):
                start_copy = time.time()
                if (batch > 0):
                    time_enum += (start_copy - post_proc)
                np_features = (np.asarray(torch.permute(features, (0, 2, 3, 1))).flatten()*32).astype(np.int8)
                in_buffer[:] = np_features[:]
                # in_buffer[:] = 0
    
                # print("SENDING DATA")
                start = time.time()    
                dma.recvchannel.transfer(out_buffer)
                dma.sendchannel.transfer(in_buffer)
    
                dma.sendchannel.wait()
    
                dma.recvchannel.wait()
    
                end = time.time()
    
                accuracy, accuracy_batch = postprocess(out_buffer, results, accuracy, batch_size)
                post_proc = time.time()
                
                batch_time = end - start
                total_time += batch_time
                time_to_copy += (start - start_copy)
                time_post_proc = post_proc - end
                print(f"Batch {batch} time: {batch_time:.2f} seconds", flush=True)
                if (batch == MAX_BATCH - 1):
                    break 
                # print("Total time:", batch_time, "seconds")
                # print("Inference time:", batch_time/BATCH_SIZE, "seconds")
    
            if (off_chip_memory):
                network.write(0x0, 0x0)
    
            with open("overlay/results.txt", "w+") as fd:
                # total_energy = energy
                fd.write(f"Total time: {total_time:.2f} seconds\n")
                # fd.write("Total power: %f W" % mean_powe\nr)
                # fd.write("Total energy: %f J" % total_energ\ny)
                fd.write(f"Batch size {batch_size}\n")
                fd.write(f"Images nums: {batch_size * MAX_BATCH}\n")
                fd.write(f"Fps: {(batch_size * MAX_BATCH / total_time)}\n")
                fd.write(f"Accuracy: {(accuracy / (batch_size * MAX_BATCH))}\n")
    
                # total_energy = energy
            print("Total time:", total_time, "seconds")
            # print("Total energy:", mean_power , "W")
            # print("Total energy:", total_energy, "J")
            print("Batch size %d" % batch_size)
            print('images nums: {} .'.format(batch_size * MAX_BATCH))
            print('fps: {} .'.format(batch_size * MAX_BATCH / total_time))
            print('Accuracy: {} .'.format(accuracy / (batch_size * MAX_BATCH)))
        end_exe = time.time()
        print(f"Total execution time: {end_exe - start_exe:.2f} seconds")
        print(f"Total copy time: {time_to_copy:.2f} seconds")
        print(f"Total post processing time: {time_post_proc:.2f} seconds")
        print(f"Total enumeration time: {time_enum:.2f} seconds")

def ultra96_inference(
    test_loader,
    in_buffer,
    out_buffer,
    dma,
    batch_size,
    off_chip_memory,
    network,
    postprocess,
    uram_storage
):
    rails = pynq.get_rails()

    recorder = pynq.DataRecorder(rails["INT"].power)

    NUM_BATCH = len(test_loader)

    # Clocks.fclk0_mhz = 200
    if (off_chip_memory):
        network.write(0x0, 0x81)

    with recorder.record(0.05):        

        for i in range(1):
            interval_time = 0
            total_time = 0
            total_energy = 0
            result = list()
            accuracy = 0

            for batch, (features, results) in enumerate(test_loader):
                np_features = (np.asarray(torch.permute(features, (0, 2, 3, 1))).flatten()*255).astype(np.uint8)
                in_buffer[:] = np_features[:]

                # print("SENDING DATA")
                start = time.time()    
                dma.recvchannel.transfer(out_buffer)
                dma.sendchannel.transfer(in_buffer)

                dma.sendchannel.wait()

                dma.recvchannel.wait()

                end = time.time()
                batch_time = end - start
                total_time += batch_time

                predicted = np.argmax(np.asarray(out_buffer[:]), axis=-1)
                accuracy_batch = np.equal(predicted, results)
                accuracy_batch = accuracy_batch.sum()
                accuracy += accuracy_batch

                # print("Total time:", batch_time, "seconds")
                # print("Inference time:", batch_time/BATCH_SIZE, "seconds")
                # print("Batch accuracy:", accuracy_batch/BATCH_SIZE)
            if (off_chip_memory):
                network.write(0x0, 0x0)

            # # Energy measurements    
            mean_power = recorder.frame["INT_power"].mean()
            total_energy = mean_power * total_time
            # energy = 0

            with open("overlay/results.txt", "w+") as fd:
                # total_energy = energy
                fd.write("Total time: %f seconds" % total_time)
                fd.write("Total power: %f W" % mean_power)
                fd.write("Total energy: %f J" % total_energy)
                fd.write("Batch size %d" % batch_size)
                fd.write('images nums: {} .'.format(batch_size*NUM_BATCH))
                fd.write('fps: {} .'.format(batch_size * NUM_BATCH / total_time))
                fd.write('Accuracy: {} .'.format(accuracy / (batch_size * NUM_BATCH)))

                # total_energy = energy
                print("Total time:", total_time, "seconds")
                print("Total energy:", mean_power , "W")
                print("Total energy:", total_energy, "J")
                print("Batch size %d" % batch_size)
                print('images nums: {} .'.format(batch_size*NUM_BATCH))
                print('fps: {} .'.format(batch_size * NUM_BATCH / total_time))
                print('Accuracy: {} .'.format(accuracy / (batch_size * NUM_BATCH)))
