import csv
import json
import os
import shutil as sh
import uuid
from argparse import Namespace

import numpy as np
import torch
import time
import torch.nn as nn
from torchvision.utils import save_image
import glob

from losses import l1_loss, fast_ssim, ssim
from scene_hs.dataset_readers import CameraInfo
from utils.camera_utils import camera_to_JSON, get_minicam
from utils.graphics_utils import focal2fov
from utils.image_utils import psnr
from scipy.spatial.transform import Rotation as R
import concurrent.futures
import roma
from typing import List
import torch.optim as optim
from scene_hs.skeleton import Skeleton
from scene_hs.skeleton_global import SkeletonGlobal
import nvtx
from bone_bone import bk_gather


def thaw_parameters(optimizer, params_to_thaw, all_lrs):
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_thaw:
            param_group['lr'] = all_lrs[param_group["name"]]

def freeze_parameters(optimizer, params_to_freeze):
    all_lrs = {}
    for param_group in optimizer.param_groups:
        all_lrs[param_group["name"]] = param_group['lr']

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_freeze:
            param_group['lr'] = 0.0

    return all_lrs

def create_output(frame_idx, model_params, dataset, gaussian_model):

    path_to_output = os.path.join(model_params.model_path, str(frame_idx))
    # if os.path.exists(path_to_output):
    #     sh.rmtree(path_to_output)
    # os.mkdir(path_to_output)
    os.makedirs(path_to_output, exist_ok=True)

    # path_to_point_clouds_folder = os.path.join(path_to_output, 'point_cloud')
    # os.mkdir(path_to_point_clouds_folder)

    # Copy cfg_args
    src_file_args = os.path.join(model_params.model_path, "cfg_args")
    dst_file_args = os.path.join(path_to_output, "cfg_args")
    sh.copyfile(src_file_args, dst_file_args)

    # initial point cloud
    # init_model_path = os.path.join(path_to_output, 'input.ply')
    # gaussian_model.save_ply(init_model_path)

    json_cams = []
    num_cams = dataset.__len__()
    for idx in range(num_cams):
        cur_data = dataset.__getitem__(idx)
        R = np.transpose(cur_data["cam_info"]["w2c"][:, 0:3])
        T = cur_data["cam_info"]["w2c"][:, 3]
        fovx = focal2fov(
            cur_data["cam_info"]["K"][0, 0], cur_data["cam_info"]["width"]
        )
        fovy = focal2fov(
            cur_data["cam_info"]["K"][1, 1], cur_data["cam_info"]["height"]
        )

        cur_cam = CameraInfo(
            uid=cur_data["id"],
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=None,
            image_path=cur_data["id"],
            image_name=cur_data["id"],
            width=cur_data["cam_info"]["width"],
            height=cur_data["cam_info"]["height"],
        )

        json_cams.append(camera_to_JSON(id=cur_data["id"], camera=cur_cam))

    with open(os.path.join(path_to_output, "cameras.json"), "w") as file:
        json.dump(json_cams, file)

    return path_to_output


def get_timestamps(source_path, subsample_step=1):

    timestamps = []
    path_to_obj_timestamps = os.path.join(
        source_path, "objects", "timestamps.txt"
    )
    if os.path.exists(path_to_obj_timestamps):
        # object timestamps
        timestamps = np.loadtxt(path_to_obj_timestamps)
    else:
        # stadium timestamps
        path_to_cam_poses = os.path.join(
            source_path, "cams", "1", "CameraPoses.csv"
        )
        if os.path.exists(path_to_cam_poses):
            with open(path_to_cam_poses, "r") as f:
                data = list(csv.reader(f, delimiter=","))
            timestamps = [float(x[-1]) for x in data[1:]]
        elif os.path.exists(os.path.join(source_path, "rgb")):
            file_list = sorted(glob.glob(os.path.join(source_path, "rgb", "cam_0000/*.png")))
            timestamps = [int(os.path.basename(f).split(".")[0]) for f in file_list]
        else:
            file_list = sorted(glob.glob(os.path.join(source_path, "*/*.png")))
            timestamps = [int(os.path.basename(f).split(".")[0]) for f in file_list]

    print("Found {} timestamps".format(len(timestamps)))

    if subsample_step > 1:
        timestamps = timestamps[::subsample_step]
        print("Subsampled to {} timestamps".format(len(timestamps)))

    return timestamps


def send_data_to_device(data_dict, device="cuda:0"):

    for cur_key, cur_val in data_dict.items():
        if torch.is_tensor(cur_val):
            data_dict[cur_key] = cur_val.to(device)
        elif isinstance(cur_val, np.ndarray):
            data_dict[cur_key] = torch.from_numpy(np.copy(cur_val)).to(device)
        else:
            if isinstance(cur_val, dict):
                send_data_to_device(data_dict=cur_val, device=device)
            else:
                continue


def transform_batch_format(src_batch, device="cuda:0"):

    # src_batch = { 'id', 'intrinsics', 'extrinsics', 'class2color', 'img_data'}
    # dst_batch = {'cam': cam, 'im': im, 'seg': seg_col, 'id': c}
    batch_size = src_batch["intrinsics"].shape[0]
    width = src_batch["img_data"]["rgb"].shape[-1]
    height = src_batch["img_data"]["rgb"].shape[-2]
    K_list = src_batch["intrinsics"]
    w2c_list = src_batch["extrinsics"]["w2c"]

    list_cams = []
    list_imgs = []
    list_seg = []
    list_id = []
    hair_mask = [0, 0, 0]

    for idx in range(batch_size):
        cur_cam = get_minicam(width, height, K_list[idx], w2c_list[idx])
        list_cams.append(cur_cam)

        if "class2color" in src_batch:
            cur_head_mask = [
                float(src_batch["class2color"]["Head"][0][idx]),
                float(src_batch["class2color"]["Head"][1][idx]),
                float(src_batch["class2color"]["Head"][2][idx]),
            ]
            mask_data = [hair_mask, cur_head_mask]
            cur_im, cur_seg = create_img_and_mask(
                img_rgb=src_batch["img_data"]["rgb"][idx],
                img_seg=src_batch["img_data"]["semantic"][idx],
                mask_data=mask_data,
            )
        else:
            cur_im, cur_seg = create_img_and_mask(
                img_rgb=src_batch["img_data"]["rgb"][idx],
                img_seg=None,
                mask_data=None,
            )

        list_imgs.append(torch.squeeze(cur_im))
        list_seg.append(cur_seg)
        list_id.append(int(src_batch["id"][idx]))

    seg_tensor = torch.stack(list_seg, axis=0).to(device)
    imgs_tensor = torch.stack(list_imgs, axis=0).to(device)

    dst_batch = {
        "cam": list_cams,
        "im": imgs_tensor,
        "seg": seg_tensor,
        "id": list_id,
    }

    return dst_batch


def get_loss_batch(batch_data, gauss_params, variables, is_initial_t_step):

    batch_size = len(batch_data["id"])
    loss_accum = 0

    for idx in range(batch_size):

        cur_img_data = {
            "cam": batch_data["cam"][idx],
            "im": batch_data["im"][idx],
            "seg": batch_data["seg"][idx],
            "id": batch_data["id"][idx],
        }

        cur_loss, variables = get_loss(
            gauss_params, cur_img_data, variables, is_initial_t_step
        )

        loss_accum = loss_accum + cur_loss

    return loss_accum, variables


def get_loss(gaussians_model, cur_img_data, variables, frame_idx):
    pass


def render_samples(
    gaussian_model,
    cur_dataset,
    render_func,
    pipeline,
    frame_idx,
    output_folder,
    max_workers=4,
    show_render=True,
    show_gt=True,
    show_vs=True,
):
    render_dir = os.path.join(output_folder, "sample_renders")

    os.makedirs(render_dir, exist_ok=True)
    
    num_cams = cur_dataset.__len__()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in range(0, num_cams, 10):
            futures.append(executor.submit(render_sample_atomic,
                cur_dataset[idx], render_func, gaussian_model, pipeline, render_dir, idx, show_render, show_gt, show_vs))
        concurrent.futures.wait(futures)
    render_sample_atomic(cur_dataset[0], render_func, gaussian_model, pipeline, render_dir, 0, show_render, show_gt, show_vs)
        
def render_sample_atomic(cur_data, render_func, gaussian_model, pipeline, render_dir, idx, show_render, show_gt, show_vs):
    send_data_to_device(cur_data)

    if show_render or show_vs:
        image = torch.clamp(
            render_func(cur_data, gaussian_model, pipeline)["render"], 0.0, 1.0
        )

    if show_render:
        save_image(
            image,
            os.path.join(render_dir, f"render_{idx}.png"),
        )
    if show_gt:
        save_image(
            cur_data["im"],
            os.path.join(render_dir, f"gt_{idx}.png"),
        )
    if show_vs:
        image_gt_rendered = torch.hstack((cur_data["im"], image))
        save_image(
            image_gt_rendered,
            os.path.join(render_dir, f"gt_vs_render_{idx}.png"),
        )


def training_report(
    gaussian_model,
    iteration,
    cur_dataset,
    testing_iterations,
    renderFunc,
    pipeline,
):

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        l1_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0

        num_images = cur_dataset.__len__()
        for idx in range(num_images):
            cur_data = cur_dataset.__getitem__(idx)
            image = torch.clamp(
                renderFunc(cur_data, gaussian_model, pipeline)["render"],
                0.0,
                1.0,
            )
            gt_image = torch.clamp(cur_data["im"].to("cuda"), 0.0, 1.0)
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += fast_ssim(image, gt_image).mean().double()
        l1_test /= len(cur_dataset)
        psnr_test /= len(cur_dataset)
        ssim_test /= len(cur_dataset)
        print(
            "\n[ITER {}] Evaluating: L1 {} PSNR {} SSIM {}".format(
                iteration, l1_test, psnr_test, ssim_test
            )
        )
        torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    return True

def quaternion_to_rotation_matrix(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotation_matrix_to_quaternion(R):
    q = torch.zeros((R.size(0), 4), device='cuda')
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    max_trace = trace > 0
    
    if max_trace.any():
        s = torch.sqrt(trace[max_trace] + 1.0) * 2
        q[max_trace, 0] = 0.25 * s
        q[max_trace, 1] = (R[max_trace, 2, 1] - R[max_trace, 1, 2]) / s
        q[max_trace, 2] = (R[max_trace, 0, 2] - R[max_trace, 2, 0]) / s
        q[max_trace, 3] = (R[max_trace, 1, 0] - R[max_trace, 0, 1]) / s

    not_max_trace = ~max_trace
    if not_max_trace.any():
        max_diag = torch.argmax(torch.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], dim=-1), dim=-1)
        
        i = torch.logical_and(max_diag == 0, not_max_trace)
        j = torch.logical_and(max_diag == 1, not_max_trace)
        k = torch.logical_and(max_diag == 2, not_max_trace)

        if i.any():
            s = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
            q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (R[i, 0, 1] + R[i, 1, 0]) / s
            q[i, 3] = (R[i, 0, 2] + R[i, 2, 0]) / s

        if j.any():
            s = torch.sqrt(1.0 + R[j, 1, 1] - R[j, 0, 0] - R[j, 2, 2]) * 2
            q[j, 0] = (R[j, 0, 2] - R[j, 2, 0]) / s
            q[j, 1] = (R[j, 0, 1] + R[j, 1, 0]) / s
            q[j, 2] = 0.25 * s
            q[j, 3] = (R[j, 1, 2] + R[j, 2, 1]) / s

        if k.any():
            s = torch.sqrt(1.0 + R[k, 2, 2] - R[k, 0, 0] - R[k, 1, 1]) * 2
            q[k, 0] = (R[k, 1, 0] - R[k, 0, 1]) / s
            q[k, 1] = (R[k, 0, 2] + R[k, 2, 0]) / s
            q[k, 2] = (R[k, 1, 2] + R[k, 2, 1]) / s
            q[k, 3] = 0.25 * s

    return q

def rotmat2qvec(R):
    batch_size = R.shape[0]
    
    # Extract individual elements of the rotation matrices
    Rxx, Ryx, Rzx = R[:, 0, 0], R[:, 1, 0], R[:, 2, 0]
    Rxy, Ryy, Rzy = R[:, 0, 1], R[:, 1, 1], R[:, 2, 1]
    Rxz, Ryz, Rzz = R[:, 0, 2], R[:, 1, 2], R[:, 2, 2]
    
    K = torch.stack([
        torch.stack([Rxx - Ryy - Rzz, Ryx + Rxy, Rzx + Rxz, Ryz - Rzy], dim=1),
        torch.stack([Ryx + Rxy, Ryy - Rxx - Rzz, Rzy + Ryz, Rzx - Rxz], dim=1),
        torch.stack([Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, Rxy - Ryx], dim=1),
        torch.stack([Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz], dim=1)
    ], dim=1) / 3.0

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(K)
    
    # Extract the eigenvector corresponding to the largest eigenvalue
    max_eigval_indices = torch.argmax(eigvals, dim=1)
    qvec = torch.stack([eigvecs[i, :, max_eigval_indices[i]] for i in range(batch_size)], dim=0)
    
    # Reorder to match [w, x, y, z] convention
    qvec = qvec[:, [3, 0, 1, 2]]
    
    # Ensure the scalar part of the quaternion is positive
    qvec = torch.where(qvec[:, 0].unsqueeze(1) < 0, -qvec, qvec)
    
    return qvec

# def ensure_right_handed2(rotations):
#     # Compute the determinant of each rotation matrix
#     det = torch.det(rotations)
    
#     # Create a tensor of the same shape as rotations filled with ones
#     # except for the last element of the last dimension which will be -1
#     correction = torch.ones_like(rotations)
#     correction[:, 2, 2] = -1
    
#     # Apply correction where the determinant is negative
#     corrected_rotations = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, rotations * correction, rotations)
    
#     return corrected_rotations

def ensure_right_handed(rotations):
    crosses = torch.cross(rotations[:,0,:].squeeze(1), rotations[:,1,:].squeeze(1))
    dots = torch.bmm(crosses.unsqueeze(2).transpose(1,2), rotations[:,2,:].squeeze(1).unsqueeze(2)).flatten()
    
    corrected_rotations = rotations
    corrected_rotations[dots < 0, 2] *= -1
    
    return corrected_rotations


def ensure_right_handed2(rotations, sizes):
    crosses = torch.cross(rotations[:,:,0], rotations[:,:,1])
    dots = torch.bmm(crosses.unsqueeze(2).transpose(1,2), rotations[:,:,2].unsqueeze(2)).flatten()
    
    corrected_rotations = rotations
    corrected_sizes = sizes
    w = dots < 0
    
    (corrected_rotations[w,:,2], corrected_rotations[w,:,1]) = (corrected_rotations[w,:,1], corrected_rotations[w,:,2])
    (corrected_sizes[w, 2], corrected_sizes[w, 1]) = (corrected_sizes[w, 1], corrected_sizes[w, 2])
    return corrected_rotations, corrected_sizes

def ensure_left_handed(rotation_matrices):
    # Assumes input is of shape (batch_size, 3, 3)
    batch_size = rotation_matrices.size(0)

    # Compute the determinant of each rotation matrix
    det = torch.det(rotation_matrices)
    
    # Create a mask for matrices with positive determinants (right-handed)
    mask = det > 0
    
    # To convert right-handed to left-handed, we can negate the last column (or row)
    # Here we negate the last column
    adjustment = torch.eye(3, device=rotation_matrices.device).repeat(batch_size, 1, 1)
    adjustment[:, :, 2] = -1

    # Apply the adjustment where the mask is True
    adjusted_matrices = rotation_matrices * mask.unsqueeze(-1).unsqueeze(-1) + rotation_matrices * (~mask).unsqueeze(-1).unsqueeze(-1)
    
    return adjusted_matrices

def apply_combined_rotations(transformations, quaternions, sizes):
    # Extract the rotation part (upper-left 3x3 submatrix of the transformation matrix)
    rotations = transformations[:, :3, :3].contiguous()  # Nx3x3
    
    # Convert quaternions to rotation matrices
    quaternion_rotations = quaternion_to_rotation_matrix(quaternions).contiguous()
    
    # Combine the rotations by matrix multiplication
    combined_rotations, corrected_sizes = ensure_right_handed2(torch.bmm(rotations, quaternion_rotations).contiguous(), sizes)
    
    # Convert the combined rotation matrices back to quaternions
    combined_quaternions = rotation_matrix_to_quaternion(combined_rotations)

    return combined_quaternions.data, corrected_sizes.data

def apply_transformation(transformations, base_verts, diffs, spatial_scale_factor):

    # Extract the translation part (last column of the transformation matrix)
    translations = transformations[:, :3, 3] # Nx3
    rotations = transformations[:, :3, :3]  # Nx3x3

    # Apply the translation to the vertices
    translated_vertices = (base_verts + translations * spatial_scale_factor) + torch.bmm(rotations, diffs.unsqueeze(2)).squeeze(2)

    return translated_vertices.data


def indices_from_file(file_path):
    with open(file_path, 'r') as f:
        indices = [int(line.strip()) for line in f]
    return torch.tensor(indices, dtype=torch.long)


def read_skin_file(skn_path, skl_path="data/running_black/static/skls/0000.skl", return_dict=False, device='cuda'):
    """
    Args:
        skn_path (str): The path to the skinning weight file. 
        skl_path (str): The path to a skeleton file. This is just used to get the index of each joint.
    Return:
        skn_weights (torch.Tensor N x max_num_joints): max_num_joints is the maximum number of 
                                                    joints that a joint may depend on. 
        skn_weights_inds (torch.Tensor N x max_num_joints): The joint index of each entry in skn_weights
        name2idx (dict): {joint_name : joint_idx}
    """
    skn_weights_dict = {}
    with open(skn_path, 'r') as file:
        for line in file:
            # Only process lines that start with 'vi ' (vertices)
            if line.startswith('vi '):
                parts = line.split()
                vert_idx = int(parts[1])
                v_value = {}
                i = 2
                while i < len(parts):
                    v_value[parts[i]] = float(parts[i+1])
                    i += 2
                skn_weights_dict[vert_idx] = v_value
    if return_dict:
        return skn_weights_dict
    
    name2idx = Skeleton(skl_path, 'cpu').name2idx
    
    num_vertices = len(skn_weights_dict)
    max_num_joints = max([len(x) for x in skn_weights_dict.values()])
    skn_weights = torch.zeros((num_vertices, max_num_joints), dtype=torch.float32, device="cpu")
    skn_weights_inds = torch.zeros((num_vertices, max_num_joints), dtype=torch.int32, device="cpu")
    for vert_idx, vert_skn_weights in skn_weights_dict.items():
        for neighbor_idx, (neighbor_name, neighbor_weight) in enumerate(vert_skn_weights.items()):
            skn_weights[vert_idx, neighbor_idx] = neighbor_weight
            skn_weights_inds[vert_idx, neighbor_idx] = name2idx[neighbor_name]
    
    return skn_weights.to(device), skn_weights_inds.to(device)

def start_range(name):
    torch.cuda.synchronize()
    g = nvtx.start_range(name)
    return g

def end_range(g):
    torch.cuda.synchronize()
    nvtx.end_range(g)
    
    
ran = None
def startran(name):
    global ran
    ran = start_range(name)
    
def endran():
    global ran
    end_range(ran)

def transform_gaussians_by_skeleton(gaussian_model, 
                                    binding_skeleton: SkeletonGlobal, 
                                    animated_skeleton: SkeletonGlobal, 
                                    gs_skn_weights, 
                                    gs_skn_weights_inds,
                                    chains):
    """
        n: number of gaussians
        m: number of vertices
        k: number of joints
        r: max number of joints that each vertex may depend on 
        
        gaussian_model.get_xyz [n x 3]
        gaussian_model.get_rotation [n x 4] in (w, x, y, z) format
        gs2vert [n]
        skeleton.get_xyz [k x 3]
        skeleton.get_global_rotmats [k x 3 x 3]
        skeleton.get_global_rotmats [k x 3 x 3]
        gs_skn_weights [n x r]
        gs_skn_weights_inds [n x r]
    
    """
    # Archive the gaussian binding position and rotation at the beginning
    if not hasattr(gaussian_model, "binding_xyz"):
        gaussian_model.binding_xyz = gaussian_model.get_xyz.clone().detach()
    if not hasattr(gaussian_model, "binding_rotation"):
        gaussian_model.binding_rotation = gaussian_model.get_rotation.clone().detach()

    # g = start_range('drive')
    animated_skeleton.drive_with_chains(chains)
    # end_range(g)

    r = gs_skn_weights.shape[1]

    gs_rotquats = gaussian_model.binding_rotation[:, None, [1,2,3,0]].repeat(1, r, 1) # [n x r x 4] in (x, y, z, w) format
    gs_xyz = gaussian_model.binding_xyz[:, None, :] # [n x 1 x 3]
    binding_skl_xyz = binding_skeleton.get_xyz[gs_skn_weights_inds] # [n x r x 3]
    binding_skl_global_rotquats = binding_skeleton.get_global_rotquats_norm[gs_skn_weights_inds] # [n x r x 4]
    animated_skl_xyz = bk_gather(animated_skeleton.get_xyz, gs_skn_weights_inds)
    animated_skl_global_rotquats = bk_gather(animated_skeleton.get_global_rotquats_norm, gs_skn_weights_inds)
    
    rel_rotquats = roma.quat_product(animated_skl_global_rotquats, roma.quat_conjugation(binding_skl_global_rotquats)) # [n x r x 4]
    rotated_xyz = roma.quat_action(rel_rotquats, (gs_xyz - binding_skl_xyz), True).squeeze() + animated_skl_xyz # [n x r x 3] 
    rotated_rotquats = roma.quat_product(rel_rotquats, gs_rotquats) # [n x r x 4]
    averaged_xyz = torch.sum(rotated_xyz * gs_skn_weights.unsqueeze(-1), dim=1, keepdim=False) # [n x 3]
    averaged_rotquats = torch.sum(rotated_rotquats * gs_skn_weights.unsqueeze(-1), dim=1, keepdim=False)[:, [3,0,1,2]] # [n x 4] in (w, x, y, z) format

    gaussian_model._xyz = averaged_xyz
    gaussian_model._rotation = averaged_rotquats


C0 = torch.tensor(0.28209479177387814)
C1 = torch.tensor(0.4886025119029199)

def weighted_least_squares_batch(A, W, b):
    # Ensure W is a diagonal matrix
    W = torch.diag_embed(W)
    torch.clamp_min_(W, 1e-8)
    
    # Compute A^T W A for each batch
    AT = A.transpose(1, 2)
    AT_W = torch.bmm(AT, W)
    AT_W_A = torch.bmm(AT_W, A)
    
    # Compute A^T W b for each batch
    AT_W_b = torch.bmm(AT_W, b)
    
    # Solve the system (A^T W A) x = A^T W b using batched inverse
    x = torch.inverse(AT_W_A) @ AT_W_b
    
    return x

def read_obj_file(filename):
    """
    Reads the vertices (3D points) from an OBJ file.
    Parameters:
    - filename: str, the path to the OBJ file.
    Returns:
    - points: list of lists, each containing a 3D point [x, y, z]
    """
    points = []
    with open(filename, 'r') as file:
        for line in file:
            # Only process lines that start with 'v ' (vertices)
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
    return points