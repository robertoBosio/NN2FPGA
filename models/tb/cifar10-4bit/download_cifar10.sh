#!/bin/bash

# If not already downloaded
if [ ! -f ${WORK_ROOT}/data/cifar-10-batches-bin/data_batch_1.bin ]; then
    # If the archive does not exist, download it

    if [ ! -f ${WORK_ROOT}/data/cifar-10-binary.tar.gz ]; then
        mkdir -p ${WORK_ROOT}/data/
        wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -O ${WORK_ROOT}/data/cifar-10-binary.tar.gz --no-check-certificate
    fi

    # Extract all the files
    echo "Extracting the cifar10"
    tar xzf ${WORK_ROOT}/data/cifar-10-binary.tar.gz -C ${WORK_ROOT}/data/.
fi
