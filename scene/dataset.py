from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
import sys

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        
    def __getitem__(self, index):
        # breakpoint()
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time, sam_features, sam_masks, gt_mask = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                sam_features = caminfo.sam_features
                sam_masks = caminfo.sam_masks
                gt_mask = caminfo.gt_mask
                mask = caminfo.mask
            
            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                        image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                        mask=mask, sam_features=sam_features, sam_masks=sam_masks, gt_mask=gt_mask)
        else:
            return self.dataset[index]
        
    def __len__(self):
        return len(self.dataset)
