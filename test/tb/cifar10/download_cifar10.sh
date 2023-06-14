#!/bin/bash

# If not already downloaded
if [ ! -f ${PRJ_ROOT}/data/cifar-10-batches-bin/data_batch_1.bin ]; then
    # If the archive does not exist, download it
    if [ ! -f ${PRJ_ROOT}/data/cifar-10-binary.tar.gz ]; then
        mkdir -p ${PRJ_ROOT}/data/
        wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -O ${PRJ_ROOT}/data/cifar-10-binary.tar.gz --no-check-certificate
    fi

    # Extract all the files
    echo "Extracting the cifar10"
    tar xzf ${PRJ_ROOT}/data/cifar-10-binary.tar.gz -C ${PRJ_ROOT}/data/.
fi
