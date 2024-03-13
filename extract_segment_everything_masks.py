import os
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
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--image_root", default='./data/hypernerf/split-cookie', type=str)
    parser.add_argument("--sam_checkpoint_path", default="/data/sxj/SegAnyGAussians/dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    # parser.add_argument("--downsample", default="4", type=str)
    parser.add_argument("--white_background", default=False, type=bool)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Default settings of SAM
    mask_generator = SamAutomaticMaskGenerator(
        sam, 
        pred_iou_thresh = 0.88, 
        stability_score_thresh = 0.95, 
        min_mask_region_area = 0
    )
    # Trex is hard to segment with the default setting
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.8,
    #     stability_score_thresh=0.8,
    #     crop_n_layers=0,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )
    # if args.downsample == "1":
    #     IMAGE_DIR = os.path.join(args.image_root, 'images')
    # else:
    #     IMAGE_DIR = os.path.join(args.image_root, 'images_'+args.downsample)
    if "dnerf" in args.image_root:
        IMAGE_DIR = os.path.join(args.image_root, "train")
        assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
        OUTPUT_DIR = os.path.join(args.image_root, 'train_sam_masks')
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        bg_color =  [1, 1, 1] if args.white_background else [0, 0, 0]

        print("Extracting SAM segment everything masks...")
        for path in tqdm(os.listdir(IMAGE_DIR)):
            # print(path)
            # break
            name = path.split('.')[0]
            # img = cv2.imread(os.path.join(IMAGE_DIR, path))
            img = Image.open(os.path.join(IMAGE_DIR, path))
            im_data = np.array(img.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg_color * (1 - norm_data[:, :, 3:4])
            img = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            masks = mask_generator.generate(img)

            mask_list = []
            for m in masks:
                m_score = torch.from_numpy(m['segmentation']).float()[None, None, :, :].to('cuda')
                m_score = torch.nn.functional.interpolate(m_score, size=(200,200) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                mask_list.append(m_score)
            masks = torch.stack(mask_list, dim=0)

            torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))
    
    elif "hypernerf" in args.image_root:
        IMAGE_DIR = os.path.join(args.image_root, 'rgb', "2x")
        assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
        OUTPUT_DIR = os.path.join(args.image_root, 'sam_masks', "2x")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(f'{args.image_root}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        all_img = dataset_json['ids']
        val_id = dataset_json['val_ids']
        all_img = [f'{i}.png' for i in all_img]
        if len(val_id) == 0:
            train_img = [path for idx, path in enumerate(all_img) if idx % 4 == 0]
        else:
            # TODO
            raise NotImplementedError
            # train_id = dataset_json['train_ids']
            # i_test = []
            # i_train = []
            # for i in range(len(all_img)):
            #     id = all_img[i]
            #     if id in val_id:
            #         i_test.append(i)
            #     if id in train_id:
            #         i_train.append(i)
                
        print("Extracting SAM segment everything masks...")
        # print(len(train_img))
        for path in tqdm(train_img):
            name = path.split('.')[0]
            img = cv2.imread(os.path.join(IMAGE_DIR, path))
            masks = mask_generator.generate(img)
            
            mask_list = []
            for m in masks:
                m_score = torch.from_numpy(m['segmentation']).float()[None, None, :, :].to('cuda')
                m_score = torch.nn.functional.interpolate(m_score, size=(200,200) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                mask_list.append(m_score)
            masks = torch.stack(mask_list, dim=0)

            torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))
