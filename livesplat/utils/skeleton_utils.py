import torch
import roma
from scene_hs.skeleton import Skeleton


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
    skn_weights = torch.zeros((num_vertices, max_num_joints), dtype=torch.float32, device=device)
    skn_weights_inds = torch.zeros((num_vertices, max_num_joints), dtype=torch.int32, device=device)
    for vert_idx, vert_skn_weights in skn_weights_dict.items():
        for neighbor_idx, (neighbor_name, neighbor_weight) in enumerate(vert_skn_weights.items()):
            skn_weights[vert_idx, neighbor_idx] = neighbor_weight
            skn_weights_inds[vert_idx, neighbor_idx] = name2idx[neighbor_name]
    
    return skn_weights, skn_weights_inds


def transform_gaussians_by_skeleton(gaussian_model, 
                                    binding_skeleton: Skeleton, 
                                    animated_skeleton: Skeleton, 
                                    gs_skn_weights, 
                                    gs_skn_weights_inds):
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

    animated_skeleton.drive()
    r = gs_skn_weights.shape[1]
    
    gs_xyz = gaussian_model.binding_xyz[:, None, :] # [n x 1 x 3]
    binding_skl_xyz = binding_skeleton.get_xyz[gs_skn_weights_inds] # [n x r x 3]
    animated_skl_xyz = animated_skeleton.get_xyz[gs_skn_weights_inds] # [n x r x 3]
    binding_skl_global_rotmats = binding_skeleton.get_global_rotmats[gs_skn_weights_inds] # [n x r x 3 x 3]
    animated_skl_global_rotmats = animated_skeleton.get_global_rotmats[gs_skn_weights_inds] # [n x r x 3 x 3]
    rel_rotmats = torch.matmul(animated_skl_global_rotmats, binding_skl_global_rotmats.mT) # [n x r x 3 x 3]
    rotated_xyz = torch.matmul(rel_rotmats, (gs_xyz - binding_skl_xyz)[..., None]).squeeze()  + animated_skl_xyz # [n x r x 3] 
    averaged_xyz = torch.sum(rotated_xyz * gs_skn_weights.unsqueeze(-1), dim=1, keepdim=False) # [n x 3]
    
    # TODO: need a better way to compute the weighted average of quaternions
    gs_rotquats = gaussian_model.binding_rotation[:, None, [1,2,3,0]].repeat(1, r, 1) # [n x r x 4] in (x, y, z, w) format
    binding_skl_global_rotquats = binding_skeleton.get_global_rotquats[gs_skn_weights_inds] # [n x r x 4]
    animated_skl_global_rotquats = animated_skeleton.get_global_rotquats[gs_skn_weights_inds] # [n x r x 4]
    rel_rotquats = roma.quat_product(animated_skl_global_rotquats, roma.quat_conjugation(binding_skl_global_rotquats)) # [n x r x 4]
    rotated_rotquats = roma.quat_product(rel_rotquats, gs_rotquats) # [n x r x 4]
    averaged_rotquats = torch.sum(rotated_rotquats * gs_skn_weights.unsqueeze(-1), dim=1, keepdim=False)[:, [3,0,1,2]] # [n x 4] in (w, x, y, z) format
    
    gaussian_model._xyz = averaged_xyz
    gaussian_model._rotation = averaged_rotquats