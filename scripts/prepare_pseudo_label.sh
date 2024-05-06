#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1"
gpu="$2"

export CUDA_VISIBLE_DEVICES=$gpu

dataset_folder="data/$dataset_name"
if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# 1. DEVA anything mask
cd Tracking-Anything-with-DEVA/

if [[ $dataset_name == *"$hypernerf"* ]] 
then
  img_path="../data/${dataset_name}/rgb/2x"
elif [[ $dataset_name == *"$hypernerf"* ]] 
then
  img_path="../data/${dataset_name}/rgb/2x"
else
  echo "Cannot recognize this dataset type!"
fi

# colored mask for visualization check
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "../data/${dataset_name}/pseudo_label" \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.88 \


mv ../data/${dataset_name}/pseudo_label/Annotations ../data/${dataset_name}/pseudo_label/Annotations_color

# gray mask for training
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "../data/${dataset_name}/pseudo_label/" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.88 \
  
# 2. copy gray mask to the correponding data path
# cp -r ./example/output_gaussian_dataset/${dataset_name}/Annotations ../data/${dataset_name}/object_mask
mv ../data/${dataset_name}/pseudo_label/Annotations ../data/${dataset_name}/pseudo_label/object_mask
cd ..