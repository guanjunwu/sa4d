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

import os, sys
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.feature_gaussian_model import FeatureGaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel
    feature_gaussians : FeatureGaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, feature_gaussians: FeatureGaussianModel = None, 
                 load_iteration=None, feature_loaded_iteration=None,
                 target="scene", sample_rate=1.0, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.feature_loaded_iter = None
        self.gaussians = gaussians
        self.feature_gaussians = feature_gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained 4DGS model at iteration {}".format(self.loaded_iter))
            
        if feature_gaussians is not None and feature_loaded_iteration:
            if feature_loaded_iteration == -1:
                self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.feature_loaded_iter = feature_loaded_iteration
            print("Loading trained feature-4DGS model at iteration {}".format(self.feature_loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
                
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension, need_features = args.need_features, need_masks = args.need_masks, sample_rate = sample_rate)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval, need_features = args.need_features, need_masks = args.need_masks)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        else:
            assert False, "Could not recognize scene type!"
            
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        
        if args.add_points and target == "scene":
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
            
        # self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        self.feature_gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        
        # Load or initialize scene 4dgaussians
        if self.gaussians is not None:
            if self.loaded_iter:
                print("Loading pretrained 4dgs model")
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
                self.gaussians.load_model(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                    ))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
            
        # Load or initialize feature gaussians
        if self.feature_gaussians is not None:
            if self.feature_loaded_iter:
                print("Loading pretrained feature-4dgs model")
                self.feature_gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.feature_loaded_iter),
                                                ))
                self.feature_gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.feature_loaded_iter),
                                                    "contrastive_feature_point_cloud.ply"
                                                ))
            else:
                print("Iinitializing feature-4dgs model")
                self.feature_gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                ))
                self.feature_gaussians.load_ply_from_4dgs(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
                
    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    
    def save_features(self, iteration):
        assert self.feature_gaussians is not None
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.feature_gaussians.save_ply(os.path.join(point_cloud_path, "contrastive_feature_point_cloud.ply"))
        
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    
    def getVideoCameras(self, scale=1.0):
        return self.video_camera