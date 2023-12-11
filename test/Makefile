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
export TRAIN_BATCH_SIZE=40
export EVAL_BATCH_SIZE=100
export MAX_EPOCHS=0
export LOG_INTERVAL=40
export NUM_WORKERS=4
export WBITS=8
export ABITS=8
export PYTHON=python
export ONNX_FILE=""

# Run the calibration with batchnorm fuse
train:
	${PYTHON} py/train_fx.py

bnfuse:
	${PYTHON} py/export_withbnfuse_fx.py

all:
	${PYTHON} py/train_fx.py
	make bnfuse PYTHON=${PYTHON} LOG_NAME=${LOG_NAME} DATASET=${DATASET} MAX_EPOCHS=5 BATCH_SIZE=32 PRETRAIN_FILE=tmp/ckpt/${LOG_NAME}/checkpoint_quant_fx.t7 PRETRAIN=True

test:
	${PYTHON} py/test_fx.py

export_qonnx:
	qonnx-cleanup ${ONNX_FILE}.onnx --out-file=${ONNX_FILE}_clean.onnx
	qonnx-convert ${ONNX_FILE}_clean.onnx --output-style quant --output-file=${ONNX_FILE}_clean_quant.onnx