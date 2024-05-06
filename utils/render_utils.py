import torch
@torch.no_grad()
def get_state_at_time(pc, viewpoint_camera, seg=False):
    if seg:
        # nearest interpolate
        diff = torch.abs(pc._time_map - viewpoint_camera.time)
        index = torch.argmin(diff)
        mask = pc._mask_table[index].bool()     
           
        means3D = pc.get_xyz[mask]
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        opacity = pc._opacity[mask]
        shs = pc.get_features[mask]
        scales = pc._scaling[mask]
        rotations = pc._rotation[mask]
    else:
        means3D = pc.get_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        opacity = pc._opacity
        shs = pc.get_features
        scales = pc._scaling
        rotations = pc._rotation

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = pc._scaling
    # rotations = pc._rotation
    # cov3D_precomp = None

        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    return means3D_final, scales_final, rotations_final, opacity, shs