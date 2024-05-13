import numpy as np
import torch

# egg to cut_roasted_beef
scales_bias1 = 0.2
rotation_bias1 = torch.tensor([-np.pi/4, -np.pi/32, 0])
motion_bias1 = torch.tensor([-1, 4.7, 8])

# cut_roasted_beef to drjohnson
# 固定drjohnson场景中的第81帧相机位姿，将cut_roasted_beef变换
scales_bias1 = 0.07
rotation_bias1 = torch.tensor([np.pi/32, 1.5*np.pi/8, 1.9*np.pi/4])
motion_bias1 = torch.tensor([0.9, 0.1, 1.5])

# drjohnson to cut_roasted_beef
# 按照cut_roasted_beef的val_poses，组合
scales_bias0 = 1
rotation_bias0 = torch.tensor([np.pi/32, -np.pi/32, 0])
motion_bias0 = torch.tensor([0, -1.5, 6])

scales_bias1 = 10
rotation_bias1 = torch.tensor([-np.pi/8, 0, -np.pi/2])
motion_bias1 = torch.tensor([0, 3, -15])

# falme_salmon to deleted flame_steak
