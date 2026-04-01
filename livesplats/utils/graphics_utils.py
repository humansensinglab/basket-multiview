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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def unreal2opencv(R, C):
    """Change the coordinate system from Unreal to OpenCV."""
    """Using: https://www.dariomazzanti.com/uncategorized/change-of-basis/"""
    R = R.copy()
    C = C.copy()

    B = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1],])
    
    Br = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0],])
    
    R = B @ R @ np.linalg.inv(B) @ np.linalg.inv(Br)
    C = B @ C

    return R, C


def invert(R, t):
    """ c2w <-> w2c """
    t = -R.T @ np.array(t).reshape(3, 1)
    R = R.T
    return R, t


def find_up(c2ws):
    """Find the up vector of the scene by averaging the up vector of cameras."""
    A = np.zeros((len(c2ws), 3))
    right_in_cam = np.array([1, 0, 0])
    for i, c2w in enumerate(c2ws):
        right_in_world = right_in_cam @ c2w[:3, :3].T
        A[i, :] = right_in_world
    V = np.linalg.svd(A)[-1]
    up = V[-1, :].reshape(1, 3)
    up_normed = up / np.linalg.norm(up)
    
    down_in_cam = np.array([0, 1, 0]).reshape(1, 3)
    down_in_world = down_in_cam @ c2ws[0][:3, :3].T
    if down_in_world @ up_normed.T > 0:
        up_normed = -up_normed
    
    return up_normed.reshape(-1)


def normalize(x, axis=-1, order=2):
    l2 = np.linalg.norm(x, order, axis)
    l2 = np.expand_dims(l2, axis)
    l2[l2 == 0] = 1
    return x / l2

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin. The up direction is assumed to be z+. Here we use the OpenCV 
    camera convention: x: right, y: down, z: forward. The outputs are w2cs.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 0, 1) in default
    """

    if at is None:
        at = np.zeros_like(camera_position)
    else:
        at = np.array(at)
    if up is None:
        up = np.zeros_like(camera_position)
        up[3] = 1
    else:
        up = np.array(up)
    camera_position = np.array(camera_position)
    
    
    z_axis = normalize(at - camera_position)
    x_axis = normalize(np.cross(z_axis, up))
    y_axis = normalize(np.cross(z_axis, x_axis))

    w2cs = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return w2cs


def quasi_rect_position(theta, a=12, b=20, p=4):
    """
        quasi rectangular trajectory: (x/a)^p + (y/b)^p = 1
        input: 0 <= theta < 2*pi 
    """
    theta = theta % (2 * np.pi)
    if 0 <= theta < np.pi / 2:
        sign = [1, 1]
        theta = theta
    elif np.pi/2 <= theta < np.pi:
        sign = [-1, 1]
        theta = np.pi - theta
    elif np.pi <= theta < 3 * np.pi / 2:
        sign = [-1, -1]
        theta = theta - np.pi
    else:
        sign = [1, -1]
        theta = 2 * np.pi - theta

    k = np.tan(theta)
    x = 1 / (((1/a)**p + (k/b)**p)**(1/p) + 1e-15)
    y = k * x

    x = x * sign[0]
    y = y * sign[1]

    return x, y


def get_camera_trajectory(num_frames=50, world_up=[0,0,1]):
    camera_position = []
    at = []
    for theta in np.linspace(0, 2*np.pi, num_frames):
        x, y = quasi_rect_position(theta)
        camera_position.append([x, y, 2])
        at.append([0,0,1])
    w2cs = look_at_rotation(camera_position=camera_position, up=world_up, at=at)
    trajectory_poses = []
    for i in range(w2cs.shape[0]):
        w2c = w2cs[i,...]
        c2w = np.linalg.inv(w2c)
        t = np.array(camera_position[i]).reshape(3, 1)
        P = np.hstack([c2w, t])
        trajectory_poses.append(P)
    
    return trajectory_poses