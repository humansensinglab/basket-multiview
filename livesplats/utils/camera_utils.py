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

from cameras.cameras import Camera, MiniCam
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, getProjectionMatrix, focal2fov
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def get_minicam(width, height, K, w2c, znear=0.01, zfar=100) -> MiniCam:
    
    fovx = focal2fov(K[0, 0], width)
    fovy = focal2fov(K[1, 1], height)
    
    if isinstance(w2c, np.ndarray):
        w2c = torch.tensor(w2c).float()
        
    cur_w2c = torch.vstack((w2c, torch.tensor([[0, 0, 0, 1]], device=w2c.device))).float()
    world_view_transform = cur_w2c.transpose(0, 1).float().cuda()
    
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).float().cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).float().squeeze(0)
    
    
    return MiniCam(
        width=width, 
        height=height, 
        fovy=fovy, 
        fovx=fovx,
        znear=znear, 
        zfar=zfar, 
        world_view_transform=world_view_transform, 
        full_proj_transform=full_proj_transform)
    


def create_list_minicam(batch_data):
    batch_size =  batch_data['cam_info']['K'].shape[0]
    list_cams = []
    for idx in range(batch_size): 
        cur_cam = get_minicam(
            batch_data['cam_info']['width'][idx],
            batch_data['cam_info']['height'][idx], 
            batch_data['cam_info']['K'][idx], 
            batch_data['cam_info']['w2c'][idx])
        
        list_cams.append(cur_cam)
        
    return list_cams