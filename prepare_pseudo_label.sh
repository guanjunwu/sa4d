#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset="$1"
gpu="$2"

export CUDA_VISIBLE_DEVICES=$gpu

dataset_folder=$dataset
if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi

# image_path="$dataset/cam03"
# python ./scripts/downsize_imgs.py --image_path $image_path

# 1. DEVA anything mask
cd Tracking-Anything-with-DEVA/

if [[ $dataset == *"hypernerf"* ]]; then
  img_path="../$dataset/rgb/2x"
  output_path="../$dataset/pseudo_label"
elif [[ $dataset == *"dynerf"* ]]; then
  img_path="../$dataset/cam15/images"
  output_path="../$dataset/cam15/pseudo_label"
else
  echo "Cannot recognize this dataset type!"
fi

# colored mask for visualization check
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path $img_path \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output $output_path \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7 \


mv $output_path/Annotations $output_path/Annotations_color

# gray mask for training
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path $img_path \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output $output_path \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  
# 2. copy gray mask to the correponding data path
# cp -r ./example/output_gaussian_dataset/${dataset_name}/Annotations ../data/${dataset_name}/object_mask
mv $output_path/Annotations $output_path/object_mask
cd ..