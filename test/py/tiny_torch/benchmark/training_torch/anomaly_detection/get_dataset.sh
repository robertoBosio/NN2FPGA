#!/bin/sh

URL1="https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1"
ZIPFILE="dev_data_ToyCar.zip"

URL2="https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1"

DATASET_DIR=/home/datasets/tinyML/anomaly_detection
mkdir -p $DATASET_DIR

wget $URL1 -O $ZIPFILE
sudo unzip $ZIPFILE -d $DATASET_DIR
rm $ZIPFILE

wget $URL2 -O $ZIPFILE
sudo unzip $ZIPFILE -d $DATASET_DIR
rm $ZIPFILE
