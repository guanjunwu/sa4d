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
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, loss_cls_3d
# from gaussian_renderer import render, network_gui
from gaussian_renderer import render_contrastive_feature
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, get_combined_args
from torch.utils.data import DataLoader
import copy
        
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def training(dataset, hyper, opt, pipe, mode="feature", testing_iterations=None, saving_iterations=None, checkpoint_iterations=None, checkpoint=None, debug_from=None):    
    dataset.object_masks = True
    
    gaussians = GaussianModel(dataset.sh_degree, mode, hyper, dataset.feature_dim)
    scene = Scene(dataset, gaussians, mode=mode)
    gaussians.training_setup(opt)
    num_classes = 256
    print("Num classes: ",num_classes)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
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
    batch_size = 1
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=list)
        random_loader = True
        loader = iter(viewpoint_stack_loader)
        
    for iteration in range(first_iter, final_iter + 1):   
        iter_start.record()

        # Pick a random Camera
        # dynerf's branch
        if opt.dataloader:
            try:
                viewpoint_cam = next(loader)[0]
            except StopIteration:
                print("reset dataloader into random dataloader.")
                # if not random_loader:
                #     viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True, num_workers=32, collate_fn=list)
                #     random_loader = True
                loader = iter(viewpoint_stack_loader)
        else:
            # idx = 0
            # viewpoint_cams = []
            # while idx < batch_size :    
            #     viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
            #     if not viewpoint_stack :
            #         viewpoint_stack =  temp_list.copy()
            #     viewpoint_cams.append(viewpoint_cam)
            #     idx +=1
            # if len(viewpoint_cams) == 0:
            #     continue
            
            if not viewpoint_stack :
                viewpoint_stack =  temp_list.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gaussians.update_learning_rate(iteration)

        gt_obj = viewpoint_cam.objects.cuda().long()
        render_pkg = render_contrastive_feature(viewpoint_cam, gaussians, pipe, background)
        objects = render_pkg["render"]
        logits = gaussians._classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)
        
        loss_obj_3d = None
        # if iteration % opt.reg3d_interval == 0:
        # regularize at certain intervals
        identity_encoding = gaussians._mlp(gaussians.get_xyz, torch.tensor(viewpoint_cam.time).cuda().repeat(gaussians.get_xyz.shape[0], 1))
        logits3d = gaussians._classifier(identity_encoding.unsqueeze(1).permute(2, 0, 1))
        prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
        loss_obj_3d = loss_cls_3d(render_pkg["deformed_points"].detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
        loss = loss_obj + loss_obj_3d
        # else:
        #     loss = loss_obj
        
        loss.backward()
        iter_end.record()
                
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.{3}f}"})
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
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--mode", type=str, default="feature")
    # parser.add_argument("--gpu", type=int, default=1)
    args = get_combined_args(parser, target_cfg='scene')
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
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