# NN2FPGA
## Introduction
NN2FPGA is a framework which generates quantized convolution neural networs accelerators in C++ for AMD FPGAs.
The main goal of this project is to provide a tool targeting embedded FPGAs keeping state-of-the-art performance.
Tha main addition to the state-of-the-art is the support for resnet-like models with specific optimizations for the skip connections and optimal resource allocation using BIP algorithm.

The project is completely open-source, and it is released under the MIT license.
We would be happy to receive contributions from the community.

## Installation
### Prerequisites
A machine with a Linux distribution (tested on Centos8 and Ubuntu 20.04) and a recent version of the Xilinx suite (tested with Vivado/Vitis HLS 2023.2) is required.
The python environment is managed with conda, so it is required to have a recent version of conda installed.
To perform the whole flow, it is required to have a Xilinx FPGA board with a Vitis license.
### Installation
To install the framework, it is required to clone the repository and to install the python environment.
```bash
cd NN2FPGA
conda env create -f nn2fpga.yml
conda activate nn2fpga
```
## Usage
### Quick start
To run the framework, it is required to have a trained model in the QONNX format.
The framework is able to convert models from the ONNX format to the C++ code.
To convert a model, it is required to run the following commands:
```bash
cd nn2fpga
make all TOP_NAME=${TOP_NAME} BOARD=${BOARD} ONNX_PATH=../test/onnx/${TOP_NAME}.onnx DATASET=cifar10 
```
The framework will generate the C++ code in the `work` folder, synthesize the `HLS` code, generate the block design and the bitstream.
To deploy the bitstream on the FPGA, it is required to run the following command:
```bash
make deploy TOP_NAME=${TOP_NAME} BOARD=${BOARD} ONNX_PATH=../test/onnx/${TOP_NAME}.onnx DATASET=cifar10 
```
Supported boards are `ULTRA96v2`,`KRIA KV260`and `ZCU102`.
Right now the framework is fully working for classification datasets, but we are extending it to object detection and segmentation.

