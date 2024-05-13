import os, sys
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    parser.add_argument("--image_path", default='./data/nerf_llff_data/horns/', type=str)
    args = parser.parse_args()
    
    IMAGE_DIR = os.path.join(args.image_path, 'images')
    # assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_path, 'images_2x')
    # print(OUTPUT_DIR)
    # sys.exit(0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_list = os.listdir(IMAGE_DIR)
    image_list.sort()
    
    for path in tqdm(image_list):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        img = cv2.resize(img, dsize=None,fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, path), img)
