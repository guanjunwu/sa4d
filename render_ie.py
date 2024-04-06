# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import imageio


def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    # gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    # pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    # makedirs(colormask_path, exist_ok=True)
    # makedirs(gt_colormask_path, exist_ok=True)
    # makedirs(pred_obj_path, exist_ok=True)

    pred_obj_mask_list = []
    rgb_mask_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render_contrastive_feature(view, gaussians, pipeline, background)
        rendering_obj = results["render"]
        logits = gaussians._mlp(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        # rgb_mask = feature_to_rgb(rendering_obj)
        
        pred_obj_mask_list.append(pred_obj_mask)
        # rgb_mask_list.append(rgb_mask)

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_mask.mp4'), pred_obj_mask_list, fps=30)
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_pca.mp4'), rgb_mask_list, fps=30)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False)
        # cam_type = scene.dataset_type
        num_classes = 256
        print("Num classes: ",num_classes)

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", type=str, default="feature")
    args = get_combined_args(parser, target_cfg="feature")
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode)