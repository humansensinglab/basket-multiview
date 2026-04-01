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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from diff_gaussian_rasterization import SparseGaussianAdam
from tqdm import tqdm
import torch.nn.functional as F

class GaussianModel:
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_params, frame_idx):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        # opt_dict, 
        self.spatial_lr_scale) = model_args

        cur_opt_params = training_params.optimization.get_opt_params(frame_idx=frame_idx)
        # self.training_setup(
        #     optimization_params=cur_opt_params,
        #     scheduler_params=training_params.optimization.scheduler_conf,
        #     densification_params=training_params.densification)
        
        
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def make_parameters_leaf_tensors(self):
        self._xyz = nn.Parameter(self._xyz.clone().detach().requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc.clone().detach().requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest.clone().detach().requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.clone().detach().requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.clone().detach().requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.clone().detach().requires_grad_(True))

    def training_setup(self, optimization_params, scheduler_params, densification_params):
        self.percent_dense = densification_params.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        params_list = []
        for cur_param in optimization_params.parameters_to_optimize:
            cur_dict = {}
            if cur_param.name == 'xyz':
                cur_dict['params'] = self._xyz
                cur_dict['lr'] = cur_param.lr * self.spatial_lr_scale
                cur_dict['name'] = cur_param.name
            else: 
                if cur_param.name == 'f_dc':
                    cur_dict['params'] = self._features_dc
                elif cur_param.name == 'f_rest':
                    cur_dict['params'] = self._features_rest
                elif cur_param.name == 'opacity':
                    cur_dict['params'] = self._opacity
                elif cur_param.name == 'scaling':
                    cur_dict['params'] = self._scaling
                elif cur_param.name == 'rotation':
                    cur_dict['params'] = self._rotation
                else:
                    print("Paramenter name: " + cur_param.name + " not recognized")
                    exit(-1)

                cur_dict['lr'] = cur_param.lr
                cur_dict['name'] = cur_param.name
            params_list.append(cur_dict)

        self.optimizer = torch.optim.Adam(params_list, lr=0.0, eps=1e-15)
        # self.optimizer = SparseGaussianAdam(params_list, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=scheduler_params.position_lr_init * self.spatial_lr_scale,
            lr_final=scheduler_params.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=scheduler_params.position_lr_delay_mult,
            max_steps=optimization_params.iterations)
        
    def training_setup2(self, optimization_params, scheduler_params, densification_params):
        self.percent_dense = densification_params.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_scaling.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_scaling.shape[0], 1), device="cuda")
        
        params_list = []
        for cur_param in optimization_params.parameters_to_optimize:
            cur_dict = {}
            if cur_param.name == 'xyz':
                cur_dict['params'] = self._mesh
                cur_dict['lr'] = cur_param.lr * self.spatial_lr_scale
                cur_dict['name'] = cur_param.name
            else: 
                if cur_param.name == 'f_dc':
                    cur_dict['params'] = self._features_dc
                elif cur_param.name == 'f_rest':
                    cur_dict['params'] = self._features_rest
                elif cur_param.name == 'opacity':
                    cur_dict['params'] = self._opacity
                elif cur_param.name == 'scaling':
                    cur_dict['params'] = self._scaling
                elif cur_param.name == 'rotation':
                    cur_dict['params'] = self._rotation
                else:
                    print("Paramenter name: " + cur_param.name + " not recognized")
                    exit(-1)

                cur_dict['lr'] = 0#cur_param.lr
                cur_dict['name'] = cur_param.name
            params_list.append(cur_dict)

        self.optimizer = torch.optim.Adam(params_list, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=scheduler_params.position_lr_init * self.spatial_lr_scale,
            lr_final=scheduler_params.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=scheduler_params.position_lr_delay_mult,
            max_steps=optimization_params.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normal = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    
    def save_params_sequence(self, path, frame_idx):
        
        cur_params = { 
            "xyz" : self._xyz.detach().cpu().unsqueeze(0).numpy(),
            "normal" : np.zeros_like(self._xyz.detach().cpu().numpy()),
            "f_dc" : self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().unsqueeze(0).numpy(),
            "f_rest" : self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().unsqueeze(0).numpy(),
            "opacity" : self._opacity.detach().cpu().unsqueeze(0).numpy(),
            "scale" : self._scaling.detach().cpu().unsqueeze(0).numpy(),
            "rotation" : self._rotation.detach().cpu().unsqueeze(0).numpy(),
        }
        
        if frame_idx == 0 or not os.path.isfile(path):
            np.savez(path, **cur_params)
        else:
            params_series = dict(np.load(path))
            for key, val in cur_params.items():
                params_series[key] = np.concatenate((params_series[key], val), axis=0)
            np.savez(path, **params_series)
    
    def load_from_params_sequence(self, params_sequence, frame_idx):
        
        n = params_sequence['xyz'].shape[1]

        self._xyz = nn.Parameter(params_sequence['xyz'][frame_idx].float().cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(params_sequence['f_dc'][frame_idx].unsqueeze(2).transpose(1, 2).float().cuda().contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(params_sequence['f_rest'][frame_idx].reshape(n, 3, -1).transpose(1, 2).float().cuda().contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(params_sequence['opacity'][frame_idx].float().cuda().requires_grad_(True))
        self._scaling = nn.Parameter(params_sequence['scale'][frame_idx].float().cuda().requires_grad_(True))
        self._rotation = nn.Parameter(params_sequence['rotation'][frame_idx].float().cuda().requires_grad_(True))

        
        self.active_sh_degree = self.max_sh_degree
        

    def save_params(self, path):
        
        cur_params = { 
            "xyz" : self._xyz.detach().cpu().unsqueeze(0).numpy(),
            "normal" : np.zeros_like(self._xyz.detach().cpu().numpy()),
            "f_dc" : self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().unsqueeze(0).numpy(),
            "f_rest" : self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().unsqueeze(0).numpy(),
            "opacity" : self._opacity.detach().cpu().unsqueeze(0).numpy(),
            "scale" : self._scaling.detach().cpu().unsqueeze(0).numpy(),
            "rotation" : self._rotation.detach().cpu().unsqueeze(0).numpy(),
        }
        
        np.savez(path, **cur_params)
       
    def get_params_detached(self):
        
        cur_params = { 
            "xyz" : self._xyz.detach().clone(),
            "f_dc" : self._features_dc.detach().clone(),
            "f_rest" : self._features_rest.detach().clone(),
            "opacity" : self._opacity.detach().clone(),
            "scale" : self._scaling.detach().clone(),
            "rotation" : self.rotation_activation(self._rotation.detach()).detach().clone()
        }
        
        return cur_params

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def load_ply_pcd(self, pcld, spatial_lr_scale, scene_bbox=None, spatial_scale=0.01):
        plydata = PlyData.read(pcld)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"] * spatial_scale),
                np.asarray(plydata.elements[0]["y"] * spatial_scale),
                np.asarray(plydata.elements[0]["z"] * spatial_scale),
            ),
            axis=1,
        )
        if scene_bbox:
            xmin, xmax, ymin, ymax, zmin, zmax = scene_bbox
            mask = (xmin < xyz[:, 0]) * (xyz[:, 0] < xmax) * \
                    (ymin < xyz[:, 1]) * (xyz[:, 1] < ymax) * \
                    (zmin < xyz[:, 2]) * (xyz[:, 2] < zmax)
            xyz = xyz[mask, :]
        # num_points = plydata.elements[0]["x"].shape[0]
        num_points = xyz.shape[0]
        shs = np.random.random((num_points, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_points, 3))
        )
        self.create_from_pcd(pcd=pcd, spatial_lr_scale=spatial_lr_scale)

    def load_ply(self, path, spatial_lr_scale : float, scale : float = 1.0):
        plydata = PlyData.read(path)

        # 1. Apply scale to the XYZ positions
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1) * scale
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if scale != 1.0:
            scales += np.log(scale)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def load_ply_sh0(self, path, spatial_lr_scale : float, scale : float = 1.0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]) * scale,
                        np.asarray(plydata.elements[0]["y"]) * scale,
                        np.asarray(plydata.elements[0]["z"]) * scale),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]) * scale

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def load_ply2(self, path, spatial_lr_scale : float, vertices, inds, warped_vertices):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        #self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._dibbledi = torch.from_numpy(xyz).cuda() - vertices[inds]
        self._mesh = nn.Parameter(warped_vertices.requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_scaling.shape[0]), device="cuda")
        
    def load_obj(self, path, spatial_lr_scale : float, down_factor : int = 1, scale : float = 1.0):
        points = []
        with open(path, 'r') as file:
            for line in file:
                # Only process lines that start with 'v ' (vertices)
                if line.startswith('v '):
                    parts = line.split()
                    x, y, z = -float(parts[1]), -float(parts[2]), float(parts[3])
                    points.append([x, y, z])
        points = np.array(points)[::down_factor]
        self.create_from_mesh(points, spatial_lr_scale, scale=scale)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True).to(group["params"][0].device))
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        
    def create_random(self, center, lenght, spatial_lr_scale, num_points):
        xyz = (np.random.random((num_points, 3)) - 0.5) * lenght + center
        shs = np.random.random((num_points, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_points, 3)))
        self.create_from_pcd(pcd=pcd, spatial_lr_scale=spatial_lr_scale)
    
    def create_from_mesh(self, vertices, spatial_lr_scale, scale=1.0):
        xyz = vertices * scale
        num_points = len(xyz)
        shs = np.random.random((num_points, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_points, 3)))
        self.create_from_pcd(pcd=pcd, spatial_lr_scale=spatial_lr_scale)
        
    def update_param(self, type, value):
        
        if type == 'xyz':
            self._xyz = nn.Parameter(value.clone().detach().requires_grad_(True))
        elif type == 'rotation':
            self._rotation = nn.Parameter(
                self.rotation_activation(value.clone().detach().requires_grad_(True))
                )
    
    def filter_by_2d_masks(self, dataset):
        points = self.get_xyz.detach()
        device = points.device
        num_points = len(points)
        
        pgbr = tqdm(total=len(dataset), desc="Pruning gaussians with masks")
        prune_mask = torch.zeros(num_points, device=device, dtype=bool)
        
        for i in range(len(dataset)):
            datum = dataset[i]
            homo_points_world = torch.cat([points, torch.ones((num_points, 1), device=device)], dim=-1)
            points_cam = homo_points_world @ datum['cam_info']['w2c'].T.clone().detach().to(torch.float32)
            homo_points_img = points_cam @ datum['cam_info']['K'].T.clone().detach().to(torch.float32)
            points_img = homo_points_img[:, :2] / homo_points_img[:, 2:]
            points_img = points_img.to(torch.int32)
            
            points_img = torch.cat([points_img[:, 1:], points_img[:, :1]], dim=1) # swap x and y
            points_img[:, 0] = torch.clamp(points_img[:, 0], 0, datum['cam_info']['height']-1)
            points_img[:, 1] = torch.clamp(points_img[:, 1], 0, datum['cam_info']['width']-1)

            dilate_pix = 2
            seg = datum['seg'].to(device).to(torch.float32)
            kernel = torch.ones((1, 1, dilate_pix*2+1, dilate_pix*2+1), dtype=torch.float32, device=device)
            dilated_seg = F.conv2d(seg.unsqueeze(0), kernel, padding=dilate_pix) > 0
            cur_prune_mask = dilated_seg[0, points_img[:, 0], points_img[:, 1]] == 0
            prune_mask = torch.logical_or(prune_mask, cur_prune_mask)
            
            pgbr.update(1)
        pgbr.close()

        self.prune_points(prune_mask)
        
        print(f"{sum(prune_mask)} / {len(prune_mask)} gaussians have been pruned.")
    
    @torch.no_grad
    def makeup_SH_features(self):
        # makeup SH features from degree 1 to degree 3
        n, _, _ = self._features_rest.shape
        features_rest = nn.Parameter(torch.zeros((n, 15, 3), dtype=torch.float32, device='cuda').contiguous().requires_grad_(True))
        features_rest[:, :3, :] = self._features_rest
        self._features_rest = features_rest
    
    @torch.no_grad
    def get_gaussians_in_bboxes(self, bboxes):
        total_mask = torch.zeros(self._xyz.shape[0], dtype=bool, device=self._xyz.device)
        for bbox in bboxes:
            xmin, xmax, ymin, ymax, zmin, zmax = bbox
            cur_mask = (self._xyz[:, 0] > xmin) * (self._xyz[:, 0] < xmax) * \
                        (self._xyz[:, 1] > ymin) * (self._xyz[:, 1] < ymax) * \
                        (self._xyz[:, 2] > zmin) * (self._xyz[:, 2] < zmax)
            total_mask = torch.logical_or(total_mask, cur_mask)
        return total_mask
    
    @torch.no_grad
    def trim_gaussians_with_bboxes(self, boxes):
        mask = self.get_gaussians_in_bboxes(boxes)
        self._xyz = self._xyz[~mask, ...]
        self._opacity = self._opacity[~mask, ...]
        self._scaling = self._scaling[~mask, ...]
        self._rotation = self._rotation[~mask, ...]
        self._features_dc = self._features_dc[~mask, ...]
        self._features_rest = self._features_rest[~mask, ...]
    
    def add_gaussians(self, gaussian_model, mask):
        self._xyz = torch.concat([self._xyz, gaussian_model._xyz[mask, ...]], axis=0)
        self._opacity = torch.concat([self._opacity, gaussian_model._opacity[mask, ...]], axis=0)
        self._scaling = torch.concat([self._scaling, gaussian_model._scaling[mask, ...]], axis=0)
        self._rotation = torch.concat([self._rotation, gaussian_model._rotation[mask, ...]], axis=0)
        self._features_dc = torch.concat([self._features_dc, gaussian_model._features_dc[mask, ...]], axis=0)
        self._features_rest = torch.concat([self._features_rest, gaussian_model._features_rest[mask, ...]], axis=0)
        
        
                                     