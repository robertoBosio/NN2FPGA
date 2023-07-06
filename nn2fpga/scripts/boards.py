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
            # print("Batch accuracy:", accuracy_batch/BATCH_SIZE)

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
