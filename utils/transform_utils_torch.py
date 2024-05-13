import numpy as np
import torch
import open3d as o3d
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.render_utils import get_state_at_time
import os
from arguments import ModelParams
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import Scene as DynamicScene
from static_scene import Scene as StaticScene
from scene.feature_gaussian_model import GaussianModel as DynamicGaussianModel
from static_scene. gaussian_model import GaussianModel as StaticGaussianModel
# from gaussian_renderer import render


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = torch.tensor([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]).to(R) / 3.0
    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def rx(theta):
    return torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta), torch.cos(theta)]])

def ry(theta):
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                        [0, 1, 0],
                        [-torch.sin(theta), 0, torch.cos(theta)]])

def rz(theta):
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                        [torch.sin(theta), torch.cos(theta), 0],
                        [0, 0, 1]])
    
def rescale(means3d, scales, scale_factor: float):
    means3d = means3d * scale_factor
    # scales += np.log(scale_factor)
    scales = scales * scale_factor
    # print("rescaled with factor {}".format(scale))
    return means3d, scales
        
def rotate_by_euler_angles(means3d, rotations, rotation_angles):
        """
        rotate in z-y-x order, radians as unit
        """
        x, y, z = rotation_angles

        if x == 0. and y == 0. and z == 0.:
            return means3d, rotations

        rotation_matrix = torch.tensor(rx(x) @ ry(y) @ rz(z), dtype=torch.float32).to(rotations)

        return rotate_by_matrix(means3d, rotations, rotation_matrix)
    
def rotate_by_matrix(means3d, rotations, rotation_matrix, keep_sh_degree: bool = True):
    # rotate xyz
    means3d = torch.tensor(torch.matmul(means3d, rotation_matrix.T))

    # rotate gaussian
    # rotate via quaternions
    def quat_multiply(quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.concatenate((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), dim=-1)

    # print(rotation_matrix.device)
    quaternions = rotmat2qvec(rotation_matrix)[None, ...]
    rotations_from_quats = quat_multiply(rotations, quaternions)
    rotations = rotations_from_quats / torch.linalg.norm(rotations_from_quats, dim=-1, keepdims=True)

    # rotate via rotation matrix
    # gaussian_rotation = build_rotation(torch.from_numpy(self.rotations)).cpu()
    # gaussian_rotation = torch.from_numpy(rotation_matrix) @ gaussian_rotation
    # xyzw_quaternions = R.from_matrix(gaussian_rotation.numpy()).as_quat(canonical=False)
    # wxyz_quaternions = xyzw_quaternions
    # wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
    # rotations_from_matrix = wxyz_quaternions
    # self.rotations = rotations_from_matrix

    # TODO: rotate shs
    if keep_sh_degree is False:
        print("set sh_degree=0 when rotation transform enabled")
        sh_degrees = 0
        
    return means3d, rotations

def translation(means3d, offsets):
    # x, y, z = offsets
    # if x == 0. and y == 0. and z == 0.:
    #     return

    means3d += offsets # np.asarray([x, y, z])
    # print("translation transform applied")
    
    return means3d
    
def transform(means3d, rotations, scales, scale_factor, offsets, rotation_angles):
    means3d, scales = rescale(means3d, scales, scale_factor)
    means3d, rotations = rotate_by_euler_angles(means3d, rotations, rotation_angles)
    means3d = translation(means3d, offsets)
    # scales = scales * scale_factor
    
    return means3d, rotations, scales
    

@torch.no_grad()
def get_state_at_time(pc, timestamp=None, seg=False, static=False):
    if static:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        return means3D, scales, rotations, opacity, shs
    else: 
        if seg:
            # nearest interpolate
            diff = torch.abs(pc._time_map - timestamp)
            index = torch.argmin(diff)
            mask = pc._mask_table[index].bool()     
            
            means3D = pc.get_xyz[mask]
            time = torch.tensor(timestamp).to(means3D.device).repeat(means3D.shape[0],1)
            opacity = pc._opacity[mask]
            shs = pc.get_features[mask]
            scales = pc._scaling[mask]
            rotations = pc._rotation[mask]
        else:
            means3D = pc.get_xyz
            time = torch.tensor(timestamp).to(means3D.device).repeat(means3D.shape[0],1)
            opacity = pc._opacity
            shs = pc.get_features
            scales = pc._scaling
            rotations = pc._rotation

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                    rotations, opacity, shs,
                                                                    time)
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)
        return means3D_final, scales_final, rotations_final, opacity, shs

@torch.no_grad()
def render(viewpoint_camera, timestamp, gaussians: list, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           motion_bias: list = [torch.tensor([0,0,0])], 
           rotation_bias: list = [torch.tensor([0,0])],
           scales_bias: list = [1],
           static: list = [False],
           seg: list = [False],
           bg = True):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = None
    for pc in gaussians:
        if screenspace_points is None:
            screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        else:
            screenspace_points1 = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            screenspace_points = torch.cat([screenspace_points,screenspace_points1],dim=0)
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=gaussians[0].active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = None, None, None, None, None
    for index, pc in enumerate(gaussians):
        # means3D_final1, scales_final1, rotations_final1, opacity_final1, shs_final1 = get_state_at_time(pc, viewpoint_camera)
        # scales_final1 = pc.scaling_activation(scales_final1)
        # rotations_final1 = pc.rotation_activation(rotations_final1)
        # opacity_final1 = pc.opacity_activation(opacity_final1)
        if index == 0:
            if bg:
                means3D_final, scales_final, rotations_final, opacity_final, shs_final = get_state_at_time(pc, timestamp=timestamp, seg=seg[index], static=static[index])

                # motion_bias_t = motion_bias[index].cpu().numpy()
                # rotation_bias_t = rotation_bias[index].cpu().numpy()
                # means3D_final = means3D_final.cpu().numpy()
                # rotations_final = rotations_final.cpu().numpy()
                # scales_final = scales_final.cpu().numpy()
                motion_bias_t = motion_bias[index].to('cuda')
                rotation_bias_t = rotation_bias[index].to('cuda')
                means3D_final, rotations_final, scales_final = transform(means3D_final, rotations_final, scales_final, scales_bias[index], motion_bias_t, rotation_bias_t)
                # means3D_final = torch.from_numpy(means3D_final).cuda().to(torch.float32)
                # scales_final = torch.from_numpy(scales_final).cuda().to(torch.float32)
                # rotations_final = torch.from_numpy(rotations_final).cuda().to(torch.float32)
            else:
                continue
        else:
            means3D_final1, scales_final1, rotations_final1, opacity_final1, shs_final1 = get_state_at_time(pc, timestamp=timestamp, seg=seg[index], static=static[index])
            
            # transform
            motion_bias_t = motion_bias[index].to('cuda')
            rotation_bias_t = rotation_bias[index].to('cuda')
            
            # means3D_final1 = means3D_final1.cpu().numpy()
            # rotations_final1 = rotations_final1.cpu().numpy()
            # scales_final1 = scales_final1.cpu().numpy()
            # motion_bias_t = motion_bias_t.cpu().numpy()
            # rotation_bias_t = rotation_bias_t.cpu().numpy()
            means3D_final1, rotations_final1, scales_final1 = transform(means3D_final1, rotations_final1, scales_final1, scales_bias[index], motion_bias_t, rotation_bias_t)
            # means3D_final1 = torch.from_numpy(means3D_final1).cuda().to(torch.float32)
            # scales_final1 = torch.from_numpy(scales_final1).cuda().to(torch.float32)
            # rotations_final1 = torch.from_numpy(rotations_final1).cuda().to(torch.float32)
            
            if bg:
                # merge 4dgs
                means3D_final = torch.cat([means3D_final, means3D_final1], dim=0)
                scales_final = torch.cat([scales_final, scales_final1], dim=0)
                rotations_final = torch.cat([rotations_final, rotations_final1], dim=0)
                opacity_final = torch.cat([opacity_final, opacity_final1], dim=0)
                shs_final = torch.cat([shs_final, shs_final1], dim=0)
            else:
                means3D_final = means3D_final1
                scales_final = scales_final1
                rotations_final = rotations_final1
                opacity_final = opacity_final1
                shs_final = shs_final1
            
    colors_precomp = None
    cov3D_precomp = None
    rendered_image, _, radii, _ = rasterizer(
        means3D = means3D_final,
        means2D = screenspace_points,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda"),
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def init_dynamic_gaussians(dataset : ModelParams, hyperparam, iteration : int, cam_view=None):
    with torch.no_grad():
        gaussians = DynamicGaussianModel(dataset.sh_degree, "scene", hyperparam)
        scene = DynamicScene(dataset, gaussians, load_iteration=iteration, mode='scene', shuffle=False, cam_view=cam_view)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f'Init {dataset.model_path} finished\n')
    return gaussians, scene, background

def init_static_gaussians(dataset : ModelParams, iteration : int):
    with torch.no_grad():
        gaussians = StaticGaussianModel(dataset.sh_degree)
        scene = StaticScene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f'Init {dataset.model_path} finished\n')
    return gaussians, scene, background

def save_point_cloud(points, model_path, timestamp):
    output_path = os.path.join(model_path,"point_pertimestamp")
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    points = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_path = os.path.join(output_path,f"points_{timestamp}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


# def rotate_point_cloud(point_cloud, displacement, rotation_angles, scales_bias):
#     # theta, phi, alpha = rotation_angles
#     theta, phi, _ = rotation_angles

#     rotation_matrix_z = torch.tensor([
#         [torch.cos(theta), -torch.sin(theta), 0],
#         [torch.sin(theta),  torch.cos(theta), 0],
#         [0,                0,               1]
#     ]).to(point_cloud)
    
#     rotation_matrix_x = torch.tensor([
#         [1, 0,                0],
#         [0, torch.cos(phi), -torch.sin(phi)],
#         [0, torch.sin(phi),  torch.cos(phi)]
#     ]).to(point_cloud)
    
#     # rotation_matrix_y = torch.tensor([
#     #     [torch.cos(alpha), 0, torch.sin(alpha)],
#     #     [0, 1, 0],
#     #     [-torch.sin(alpha), 0, torch.cos(alpha)]
#     # ]).to(point_cloud)
    
#     rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_x)
#     # rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_y)
#     # print(rotation_matrix)
    
#     point_cloud = point_cloud * scales_bias
#     rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix.T)
#     # rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix.t())
#     displaced_point_cloud = rotated_point_cloud + displacement

#     return displaced_point_cloud
