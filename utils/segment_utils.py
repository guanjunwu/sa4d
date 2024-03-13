import os
import time
import torch
import pytorch3d.ops
from plyfile import PlyData, PlyElement
import numpy as np
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render
from utils.sh_utils import SH2RGB

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def get_combined_args(parser : ArgumentParser, model_path, target_cfg_file = None):
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    
    if target_cfg_file is None:
        if args_cmdline.target == 'seg':
            target_cfg_file = "seg_cfg_args"
        elif args_cmdline.target == 'scene':
            target_cfg_file = "cfg_args"
        elif args_cmdline.target == 'feature' or args_cmdline.target == 'contrastive_feature' :
            target_cfg_file = "feature_cfg_args"

    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)

def load_point_colors_from_pcd(num_points, path):
    plydata = PlyData.read(path)

    features_dc = np.zeros((num_points, 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    colors = SH2RGB(features_dc)

    # N, 3
    return torch.clamp(torch.from_numpy(colors).squeeze().cuda(), 0.0, 1.0) * 255.

def write_ply(save_path, points, colors = None, normals = None, text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    assert colors is None or normals is None, "Cannot have both colors and normals"
    
    if colors is None and normals is None:
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    elif colors is not None:
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=dtype_full)
    else:
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('normal_x', 'f4'), ('normal_y', 'f4'), ('normal_z', 'f4')]
        points = [(points[i,0], points[i,1], points[i,2], normals[i,0], normals[i,1], normals[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=dtype_full)

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)
    
def write_ply_with_color(save_path, points, colors, text=True):
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=dtype_full)
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)
    
def postprocess_statistical_filtering(pcd, precomputed_mask = None, max_time = 5):
    
    if type(pcd) == np.ndarray:
        pcd = torch.from_numpy(pcd).cuda()
    else:
        pcd = pcd.cuda()

    num_points = pcd.shape[0]
    # (N, P1, K)

    std_nearest_k_distance = 10
    
    while std_nearest_k_distance > 0.1 and max_time > 0:
        nearest_k_distance = pytorch3d.ops.knn_points(
            pcd.unsqueeze(0),
            pcd.unsqueeze(0),
            K=int(num_points**0.5),
        ).dists
        mean_nearest_k_distance, std_nearest_k_distance = nearest_k_distance.mean(), nearest_k_distance.std()
        print(std_nearest_k_distance, "std_nearest_k_distance")

        mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + std_nearest_k_distance

        mask = mask.squeeze()

        pcd = pcd[mask,:]
        if precomputed_mask is not None:
            precomputed_mask[precomputed_mask != 0] = mask
        max_time -= 1
        
    return pcd.squeeze(), nearest_k_distance.mean(), precomputed_mask

def postprocess_grad_based_statistical_filtering(pcd, precomputed_mask, feature_gaussians, view, sam_mask, pipeline_args):
    start_time = time.time()
    
    background = torch.zeros(feature_gaussians.get_opacity.shape[0], 3, device = 'cuda')

    # project 2D mask onto the segmented Gaussians
    grad_catch_mask = torch.zeros(feature_gaussians.get_opacity.shape[0], 1, device = 'cuda')
    grad_catch_mask[precomputed_mask, :] = 1
    grad_catch_mask.requires_grad = True

    grad_catch_2dmask = render(
        view, 
        feature_gaussians, 
        pipeline_args, 
        background,
        filtered_mask=~precomputed_mask, 
        override_color=None, #torch.zeros(feature_gaussians.get_opacity.shape[0], 3, device = 'cuda'),
        override_mask=grad_catch_mask,
        )['mask']


    target_mask = torch.tensor(sam_mask, device=grad_catch_2dmask.device)
    target_mask = torch.nn.functional.interpolate(target_mask.unsqueeze(0).unsqueeze(0).float(), size=grad_catch_2dmask.shape[-2:] , mode='bilinear').squeeze(0).repeat([3,1,1])
    target_mask[target_mask > 0.5] = 1
    target_mask[target_mask != 1] = 0

    loss = -(target_mask * grad_catch_2dmask).sum() + 10 * ((1-target_mask)* grad_catch_2dmask).sum()
    loss.backward()

    grad_score = grad_catch_mask.grad[precomputed_mask != 0].clone().squeeze()
    grad_score = -grad_score
    
    pos_grad_score = grad_score.clone()
    pos_grad_score[pos_grad_score <= 0] = 0
    pos_grad_score[pos_grad_score <= pos_grad_score.mean() + pos_grad_score.std()] = 0
    pos_grad_score[pos_grad_score != 0] = 1

    confirmed_mask = pos_grad_score.bool()

    if type(pcd) == np.ndarray:
        pcd = torch.from_numpy(pcd).cuda()
    else:
        pcd = pcd.cuda()

    confirmed_point = pcd[confirmed_mask == 1]

    confirmed_point, _, _ = postprocess_statistical_filtering(confirmed_point, max_time=5)

    test_nearest_k_distance = pytorch3d.ops.knn_points(
        confirmed_point.unsqueeze(0),
        confirmed_point.unsqueeze(0),
        K=2,
    ).dists
    mean_nearest_k_distance, std_nearest_k_distance = test_nearest_k_distance[:,:,1:].mean(), test_nearest_k_distance[:,:,1:].std()
    test_threshold = torch.max(test_nearest_k_distance)
    print("test threshold", test_threshold)

    while True:

        nearest_k_distance = pytorch3d.ops.knn_points(
            pcd.unsqueeze(0),
            confirmed_point.unsqueeze(0),
            K=1,
        ).dists
        mask = nearest_k_distance.mean(dim = -1) <= test_threshold
        mask = mask.squeeze()
        true_mask = mask
        if torch.abs(true_mask.count_nonzero() - confirmed_point.shape[0]) / confirmed_point.shape[0] < 0.001:
            break

        confirmed_point = pcd[true_mask,:]

    precomputed_mask[precomputed_mask == 1] = true_mask
        
    # print(time.time() - start_time)
    return confirmed_point.squeeze().detach().cpu().numpy(), precomputed_mask, test_threshold
    
def postprocess_growing(original_pcd, point_colors, seed_pcd, seed_point_colors, thresh = 0.05, grow_iter = 1):
    s_time = time.time()
    min_x, min_y, min_z = seed_pcd[:,0].min(), seed_pcd[:,1].min(), seed_pcd[:,2].min()
    max_x, max_y, max_z = seed_pcd[:,0].max(), seed_pcd[:,1].max(), seed_pcd[:,2].max()

    lx, ly, lz = max_x - min_x, max_y - min_y, max_z - min_z
    min_x, min_y, min_z = min_x - lx*0.05, min_y - ly*0.05, min_z - lz*0.05
    max_x, max_y, max_z = max_x + lx*0.05, max_y + ly*0.05, max_z + lz*0.05

    cutout_mask = (original_pcd[:,0] < max_x) * (original_pcd[:,1] < max_y) * (original_pcd[:,2] < max_z)
    cutout_mask *= (original_pcd[:,0] > min_x) * (original_pcd[:,1] > min_y) * (original_pcd[:,2] > min_z)
    
    cutout_point_cloud = original_pcd[cutout_mask > 0]

    for i in range(grow_iter):
        num_points_in_seed = seed_pcd.shape[0]
        res = pytorch3d.ops.ball_query(
            cutout_point_cloud.unsqueeze(0), 
            seed_pcd.unsqueeze(0),
            K=1,
            radius=thresh,
            return_nn=False
        ).idx

        mask = (res != -1).sum(-1) != 0

        mask = mask.squeeze()

        seed_pcd = cutout_point_cloud[mask, :]
    
    final_mask = cutout_mask.clone()
    final_mask[final_mask != 0] = mask > 0

    # print(mask.count_nonzero())
    # print(time.time() - s_time)

    return seed_pcd, final_mask, None

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   