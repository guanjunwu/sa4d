#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
# from gaussian_renderer import render, network_gui
from gaussian_renderer import render_contrastive_feature
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, get_combined_args
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
        
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def training(dataset, hyper, opt, pipe, mode="feature", testing_iterations=None, saving_iterations=None, checkpoint_iterations=None, checkpoint=None, debug_from=None):    
    dataset.need_features = True
    dataset.need_masks = True
    
    gaussians = GaussianModel(dataset.sh_degree, mode, hyper, dataset.feature_dim)
    scene = Scene(dataset, gaussians, mode=mode)
    gaussians.training_setup(opt)
    
    # background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")
    background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    final_iter = opt.feature_iterations
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    # video_cams = scene.getVideoCameras()
    # test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    
    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    
    # batch_size = opt.batch_size
    print("data loading done")
    # if opt.dataloader:
    #     viewpoint_stack = scene.getTrainCameras()
    #     if opt.custom_sampler is not None:
    #         sampler = FineSampler(viewpoint_stack)
    #         viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
    #         random_loader = False
    #     else:
    #         viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
    #         random_loader = True
    #     loader = iter(viewpoint_stack_loader)
    
    
    # print(len(train_cams)) # 150
    # print(train_cams[0].original_image.shape) # (3, 800, 800)
    # print(train_cams[0].original_sam_features.shape) # (1, 256, 64, 64)
    # print(train_cams[0].original_sam_masks.shape) # (9, 200, 200)
    # print(scene.feature_gaussians.get_sam_features.shape) # (27484, 32)
    # sys.exit(0)
    
    for iteration in range(first_iter, final_iter + 1):   
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack :
            viewpoint_stack =  temp_list.copy()
        viewpoint_cam = viewpoint_stack.pop()

        gaussians.update_learning_rate(iteration)

        sam_features = viewpoint_cam.original_sam_features.cuda() # (1, 256, 64, 64)
        H,W = sam_features.shape[-2:]

        # N_mask, H, W
        # (N_mask, 200, 200)
        sam_masks = viewpoint_cam.original_sam_masks.cuda()
        # (N_mask, 64, 64)
        sam_masks = torch.nn.functional.interpolate(sam_masks.unsqueeze(0), size=sam_features.shape[-2:] , mode='nearest').squeeze()
        nonzero_masks = sam_masks.sum(dim=(1,2)) > 0
        # (N_nonzero_mask, 64, 64)
        sam_masks = sam_masks[nonzero_masks, :, :]
        #  (N_nonzero_mask, 200, 200)
        full_resolution_sam_masks = viewpoint_cam.original_sam_masks.cuda()
        full_resolution_sam_masks = full_resolution_sam_masks[nonzero_masks, :, :]

        low_dim_sam_features = gaussians._sam_proj(
            sam_features.reshape(-1, H*W).permute([1,0])
        ).permute([1,0]).reshape(dataset.feature_dim, H, W)
        
        #! generate masked feature maps (N_masks, feature_dim, 64, 64)
        #! followed by average pooling
        # NHW, NCHW
        feature_query = (sam_masks.unsqueeze(1) * low_dim_sam_features).sum(dim = (2,3))
        feature_query /= sam_masks.sum(dim=(1,2)).unsqueeze(-1)

        #! render point features
        rendered_features = render_contrastive_feature(viewpoint_cam, gaussians, pipe, background)["render"]
        
        #! SAM-Guidance Loss
        pp = torch.einsum('NC, CHW -> NHW', feature_query, rendered_features)
        prob = torch.sigmoid(pp) # (N_masks, rendered_features.height, rendered_features.width)
        # if full_resolution_sam_masks.shape[-2:] != prob.shape[-2:]:
        #     full_resolution_sam_masks = torch.nn.functional.interpolate(full_resolution_sam_masks.unsqueeze(0), size=prob.shape[-2:] , mode='bilinear').squeeze()
        #     full_resolution_sam_masks[full_resolution_sam_masks <= 0.5] = 0
        #     full_resolution_sam_masks[full_resolution_sam_masks != 0] = 1
        bce_contrastive_loss = full_resolution_sam_masks * torch.log(prob + 1e-8) + (1 - full_resolution_sam_masks) * torch.log(1 - prob + 1e-8)
        bce_contrastive_loss = -bce_contrastive_loss.mean()

        rands = torch.rand(gaussians.get_sam_features.shape[0], device=prob.device)
        reg_loss = torch.relu(torch.einsum('NC,KC->NK', gaussians.get_sam_features[rands > 0.9, :], feature_query)).mean()
        loss = bce_contrastive_loss + 0.1 * reg_loss

        #! Correspondance Loss
        NHW = sam_masks
        N, H, W = NHW.shape
        NL = NHW.view(N,-1)
        intersection = torch.einsum('NL,NC->LC', NL, NL)
        union = NL.sum(dim = 0, keepdim = True) + NL.sum(dim = 0, keepdim = True).T - intersection
        similarity = intersection / (union + 1e-5)
        HWHW = similarity.view(H,W,H,W)
        HWHW[HWHW == 0] = -1
        norm_rendered_feature = torch.nn.functional.normalize(torch.nn.functional.interpolate(rendered_features.unsqueeze(0), (H,W), mode = 'bilinear').squeeze(), dim=0, p=2)
        correspondence = torch.relu(torch.einsum('CHW,CJK->HWJK', norm_rendered_feature, norm_rendered_feature))
        corr_loss = -HWHW * correspondence
        loss += corr_loss.mean()
        
        loss.backward()
        iter_end.record()
                
        with torch.no_grad():
            # prob[prob > 0.5] = 1.0
            # prob[prob != 1] = 0.0
            # # print(prob.shape)
            # # print(full_resolution_sam_masks.shape)
            # # sys.exit(0)
            # intersection = (prob * full_resolution_sam_masks).sum(dim=(1, 2))
            # union = (1 - (1 - prob) * (1 - full_resolution_sam_masks)).sum(dim=(1, 2))
            # mIoU = (intersection / union).mean()
            
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                                        "SAM Loss": f"{bce_contrastive_loss.item():.{3}f}",
                                        "Corr Loss": f"{corr_loss.mean().item():.{3}f}",
                                        # "mIoU": f"{mIoU:.{3}f}",
                                        })
                progress_bar.update(10)
                
            if iteration == final_iter:
                progress_bar.close()
                
        torch.cuda.empty_cache()
    
    scene.save(scene.loaded_iter)
        
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configs", type=str, default = "")
    # parser.add_argument("--load_iteration", default=None, type=int)
    parser.add_argument("--mode", type=str, default="feature")
    args = get_combined_args(parser)
    
    # args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    with open(os.path.join(args.model_path, "feature_cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.mode)

    # All done
    print("\nTraining complete.")
