#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset_path>"
    exit 1
fi

dataset="$1"
model="$2"
config="$3"
gpu="$4"

export CUDA_VISIBLE_DEVICES=$gpu

python train_binary.py \
    -s $dataset \
    -m $model \
    --configs $config

python render_binary.py \
    --model_path $model \
    --skip_train \
    --skip_test \
    --configs $config

