# This Makefile is used to calibrate and test the correctness
# of different model.
# The calibration is done by running the model on the test dataset.
# The following value should be defined:

export ROOT_DIR=./tmp
export DATA_DIR=./tmp/data
export LOG_NAME=resnetq_8w8f_cifar_fx
export PRETRAIN=True
export PRETRAIN_DIR=resnetq_8w8f_cifar_fx
export CIFAR=10
export DATASET=cifar10
export LR=0.01
export WD=1e-4
export TRAIN_BATCH_SIZE=256
export EVAL_BATCH_SIZE=100
export MAX_EPOCHS=0
export LOG_INTERVAL=40
export NUM_WORKERS=4
export WBITS=8
export ABITS=8
export PYTHON=python

# Run the calibration with batchnorm fuse
train:
	${PYTHON} py/train_fx.py

bnfuse:
	${PYTHON} py/export_withbnfuse_fx.py
