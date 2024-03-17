import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
import json


if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    parser.add_argument("--image_root", default='./data/hypernerf/split-cookie', type=str)
    parser.add_argument("--sam_checkpoint_path", default="/data/sxj/dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--white_background", default=True, type=bool)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    if "dnerf" in args.image_root:
        IMAGE_DIR = os.path.join(args.image_root, 'train')
        assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
        OUTPUT_DIR = os.path.join(args.image_root, 'train_features')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        bg_color =  [1, 1, 1] if args.white_background else [0, 0, 0]
        
        print("Extracting features...")
        for path in tqdm(os.listdir(IMAGE_DIR)):
            name = path.split('.')[0]
            # img = cv2.imread(os.path.join(IMAGE_DIR, path))
            img = Image.open(os.path.join(IMAGE_DIR, path))
            im_data = np.array(img.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg_color * (1 - norm_data[:, :, 3:4])
            img = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img,dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            # print(img)
            # break
            predictor.set_image(img)
            features = predictor.features
            # print(features)
            # break
            torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))    
        
    elif "hypernerf" in args.image_root:
        IMAGE_DIR = os.path.join(args.image_root, 'rgb', "2x")
        assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
        OUTPUT_DIR = os.path.join(args.image_root, 'sam_features', "2x")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(f'{args.image_root}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        all_img_id = dataset_json['ids']
        val_id = dataset_json['val_ids']
        all_img = [f'{i}.png' for i in all_img_id]
        if len(val_id) == 0:
            train_img = [path for idx, path in enumerate(all_img) if idx % 4 == 0]
        else:
            # TODO
            # raise NotImplementedError
            train_id = dataset_json['train_ids']
            print(train_id)
            train_img = []
            for i in range(len(all_img_id)):
                id = all_img_id[i]
                if id in train_id:
                    train_img.append(all_img[i])
                
        print("Extracting features...")
        # print(len(train_img))
        for path in tqdm(train_img):
            name = path.split('.')[0]
            img = cv2.imread(os.path.join(IMAGE_DIR, path))
            img = cv2.resize(img, dsize=(1024,1024), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            # print(img)
            # break
            predictor.set_image(img)
            features = predictor.features
            # print(features.shape)
            # break
            torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))    
