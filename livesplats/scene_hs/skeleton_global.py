import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import roma
import time
from bone_bone import drive


IGNORED_JOINT_NAMES = set(
    ['FACIAL_L_ForeheadMid2', 'FACIAL_L_ForeheadMid1', 'FACIAL_C_Skull', 'FACIAL_R_LipLower2', 
     'FACIAL_R_LipLower3', 'FACIAL_R_LipLower1', 'FACIAL_L_LipLowerOuter3', 'FACIAL_L_LipLowerOuter1', 
     'FACIAL_L_LipLowerOuter2', 'FACIAL_R_LipLowerOuter1', 'FACIAL_R_LipLowerOuter2', 
     'FACIAL_R_LipLowerOuter3', 'FACIAL_L_LipLower2', 'FACIAL_L_LipLower3', 'FACIAL_L_LipLower1', 
     'FACIAL_C_LipLower3', 'FACIAL_C_LipLower2', 'FACIAL_C_LipLower1', 'FACIAL_C_LipLowerSkin', 
     'FACIAL_R_LipLowerSkin', 'FACIAL_L_LipLowerSkin', 'FACIAL_L_LipLowerOuterSkin', 
     'FACIAL_R_LipLowerOuterSkin', 'FACIAL_L_Chin2', 'FACIAL_L_Chin3', 'FACIAL_L_Chin1', 
     'FACIAL_R_ChinSide', 'FACIAL_R_Chin2', 'FACIAL_L_Jawline1', 'FACIAL_L_Jawline2', 'FACIAL_C_Jawline', 
     'FACIAL_R_Jawline2', 'FACIAL_R_Jawline1', 'FACIAL_L_ChinSide', 'FACIAL_C_Chin', 'FACIAL_R_Chin1', 
     'FACIAL_R_Chin3', 'FACIAL_C_Chin1', 'FACIAL_C_Chin3', 'FACIAL_C_Chin2', 'FACIAL_R_ForeheadInA1', 
     'FACIAL_R_ForeheadInA3', 'FACIAL_R_ForeheadInA2', 'FACIAL_R_ForeheadInB1', 'FACIAL_R_ForeheadInB2', 
     'FACIAL_L_LipUpper1', 'FACIAL_L_LipUpper3', 'FACIAL_L_LipUpper2', 'FACIAL_C_LipUpper2', 
     'FACIAL_C_LipUpper3', 'FACIAL_C_LipUpper1', 'FACIAL_R_LipUpperOuter1', 'FACIAL_R_LipUpperOuter3', 
     'FACIAL_R_LipUpperOuter2', 'FACIAL_R_LipUpper3', 'FACIAL_R_LipUpper2', 'FACIAL_R_LipUpper1', 
     'FACIAL_L_LipUpperOuter1', 'FACIAL_L_LipUpperOuter3', 'FACIAL_L_LipUpperOuter2', 'FACIAL_R_Forehead1', 
     'FACIAL_R_Forehead2', 'FACIAL_R_Forehead3', 'FACIAL_L_Forehead3', 'FACIAL_C_Forehead1', 
     'FACIAL_C_Forehead3', 'FACIAL_C_Forehead2', 'FACIAL_L_Forehead2', 'FACIAL_L_Forehead1', 
     'FACIAL_R_ForeheadOutA1', 'FACIAL_R_ForeheadOutA2', 'FACIAL_R_ForeheadOutB2', 'FACIAL_R_ForeheadOutB1', 
     'FACIAL_R_NasolabialBulge1', 'FACIAL_R_NasolabialBulge2', 'FACIAL_R_NasolabialBulge3', 
     'FACIAL_L_NasolabialBulge2', 'FACIAL_L_NasolabialBulge3', 'FACIAL_L_Temple', 'FACIAL_L_CheekLower2', 
     'FACIAL_L_CheekLower1', 'FACIAL_L_CheekOuter4', 'FACIAL_L_ForeheadOutA2', 'FACIAL_L_ForeheadOutB1', 
     'FACIAL_L_ForeheadOutB2', 'FACIAL_L_ForeheadOutA1', 'FACIAL_R_LipUpperOuterSkin', 
     'FACIAL_L_NasolabialBulge1', 'FACIAL_R_EyesackLower2', 'FACIAL_R_EyesackLower1', 'FACIAL_C_NoseBridge', 
     'FACIAL_R_CheekLower1', 'FACIAL_R_CheekLower2', 'FACIAL_L_NasolabialFurrow', 'FACIAL_L_LipUpperSkin', 
     'FACIAL_C_ForeheadSkin', 'FACIAL_L_EyesackLower2', 'FACIAL_L_EyesackLower1', 'FACIAL_L_ForeheadInB2', 
     'FACIAL_L_ForeheadInB1', 'FACIAL_L_ForeheadInA3', 'FACIAL_L_ForeheadInA2', 'FACIAL_L_ForeheadInA1', 
     'FACIAL_R_ForeheadMid1', 'FACIAL_R_ForeheadMid2', 'FACIAL_L_ForeheadInSkin', 'FACIAL_R_UnderChin', 
     'FACIAL_C_UnderChin', 'FACIAL_C_LipUpperSkin', 'FACIAL_R_NasolabialFurrow', 'FACIAL_L_LipUpperOuterSkin', 
     'FACIAL_R_CheekOuter4', 'FACIAL_R_NoseUpper', 'FACIAL_L_NoseUpper', 'FACIAL_C_Hair1', 'FACIAL_C_NoseUpper', 
     'FACIAL_R_LipUpperSkin', 'FACIAL_R_Temple', 'FACIAL_R_ForeheadInSkin', 'FACIAL_L_UnderChin', 'FACIAL_R_NeckBackA', 
     'FACIAL_L_NeckBackA', 'FACIAL_C_NeckBackA', 'FACIAL_C_AdamsApple', 'FACIAL_L_NeckA1', 'FACIAL_L_NeckA3', 
     'FACIAL_L_NeckA2', 'FACIAL_R_NeckA3', 'FACIAL_R_NeckA2', 'FACIAL_R_NeckA1', 'FACIAL_R_NeckBackB', 'FACIAL_L_NeckBackB', 
     'FACIAL_C_NeckBackB', 'FACIAL_L_NeckB1', 'FACIAL_L_NeckB2', 'FACIAL_R_NeckB2', 'FACIAL_R_NeckB1', 'FACIAL_C_NeckB'
])

max_depth = 0

class SkeletonNodeGlobal:

    def __init__(self, skeleton, joint_name, joint_idx, children_inds, device):
        self.skeleton : SkeletonGlobal = skeleton
        self.name = joint_name
        self.idx = joint_idx
        self.children_inds = torch.tensor(children_inds, dtype=torch.int32, device=device)
        self.device = device
        
    
    @property
    def all_children_inds(self):
        inds = [self.children_inds] + [x.all_children_inds for x in self.children]
        return torch.concat(inds, dim=0)
    
    @property
    def children(self):
        return [self.skeleton.joints[self.skeleton.idx2name[i]] for i in self.children_inds]
    
    @property
    def parent_idx(self):
        return self.skeleton.parent_inds[self.idx]
    
    @property
    def parent(self):
        return self.skeleton.joints[self.skeleton.idx2name[self.parent_idx]]

    @property
    def get_xyz(self):
        return self.skeleton._xyz[self.idx]
    
    @property
    def get_local_rotquat(self):
        return self.skeleton.get_local_rotquats[self.idx]
    
    @property
    def get_global_rotquat(self):
        return self.skeleton.get_global_rotquats[self.idx]
    
    @property
    def get_local_rotmat(self):
        return self.skeleton.get_local_rotmats[self.idx]
    
    @property
    def get_zero_bone(self):
        return self.skeleton._zero_bones[self.idx]

    # @property
    # def get_bone_length(self):
    #     return self.skeleton._bone_lengths[self.idx]
    
    @property
    def get_global_rotmat(self):
        return self.skeleton.get_global_rotmats[self.idx]
    
    def compute_children_global_rotation(self):
        if len(self.children) == 0:
            return
        children_quats = self.skeleton.get_local_rotquats[self.children_inds]
        parent_quats = self.skeleton._global_rotquats[self.idx][None, :].repeat(len(children_quats), 1)
        self.skeleton._global_rotquats[self.children_inds] = roma.quat_product(parent_quats, children_quats)
        for child in self.children:
            child.compute_children_global_rotation()
    
    def compute_children_bone_length(self):
        if len(self.children) == 0:
            return
        children_xyz = self.skeleton.get_xyz[self.children_inds]
        parent_xyz = self.get_xyz[None, :]
        self.skeleton._bone_lengths[self.children_inds] = torch.cdist(parent_xyz, children_xyz).squeeze()
        for child in self.children:
            child.compute_children_bone_length()
    
    def compute_xyz(self):
        # Do global rotation from parents to children
        # with torch.no_grad():
        #     self.skeleton._xyz[self.idx] = self.parent.get_xyz
        parent_xyz = self.skeleton.get_xyz[self.parent_idx]
        parent_global_rotmat = self.skeleton.get_global_rotmats[self.parent_idx]
        self.skeleton._xyz[self.idx] = parent_xyz + (parent_global_rotmat @ self.get_zero_bone[:, None]).squeeze()
        for child in self.children:
            child.compute_xyz()

    def trace_children(self, chains, t, depth):
        global max_depth        
        if depth > max_depth:
            max_depth = depth

        if chains != None:
            chains[self.idx,:depth] = t.clone()
            t = torch.cat((t, torch.LongTensor([self.idx])))
        
        for child in self.children:
            child.trace_children(chains, t, depth + 1)
    
    def shift_children(self):

        if len(self.children) == 0:
            return
        
        self.skeleton._xyz[self.children_inds] += self.get_xyz[None, :]
        for child in self.children:
            child.shift_children()
    
    def rotate_all_children(self, origin, rot_mat):
        """Rotate all the children with the given origin and rotation
        Args:
            origin: (3)
            rot_mat: (3 x 3)
        """
        if len(origin.shape) == 1:
            origin = origin.unsqueeze(0)
        inds = self.all_children_inds
        if len(inds) > 0:
            self.skeleton._xyz[inds] = (self.skeleton._xyz[inds] - origin) @ rot_mat.T + origin
    
    # def translate_all_children(self, translation):
    #     """Translate all the children with the given translation
    #     Args:
    #         translation: (3)
    #     """
    #     if len(translation.shape) == 1:
    #         translation = translation.unsqueeze(0)
    #     inds = self.all_children_inds
    #     if len(inds) > 0:
    #         self.skeleton._xyz[inds] += translation
    
    def unrotate_tree(self):
        # Do local rotation from parents to children
        self.rotate_all_children(self.get_xyz, self.get_local_rotmat.T)
        for child in self.children:
            child.unrotate_tree()
    
    def rotate_tree(self):
        # Do local rotation from children to parents
        for child in self.children:
            child.rotate_tree()
        self.rotate_all_children(self.get_xyz, self.get_local_rotmat)


class SkeletonGlobal():
    def __init__(self, skl_path, device='cpu', requires_grad=True, skel_scale=1.0):
        self.device = device
        self.requires_grad = requires_grad
        joint_infos = self.read_skl_file(skl_path)
        
        self.name2idx = {}
        self.idx2name = []
        for joint_idx, joint_name in enumerate(joint_infos.keys()):
            self.name2idx[joint_name] = joint_idx
            self.idx2name.append(joint_name)
        
        self.joints = {}
        xyz_list = []
        local_rotquat_list = []
        parent_idx_list = []
        for joint_name, joint_info in joint_infos.items():
            joint_idx = self.name2idx[joint_name]
            parent_idx = self.name2idx[joint_info['parent_name']]
            children_inds = [self.name2idx[x] for x in joint_info['children_names']]
            joint = SkeletonNodeGlobal(self, joint_name, joint_idx, children_inds, device)
            self.joints[joint_name] = joint
            xyz_list.append(joint_info['xyz'])
            local_rotquat_list.append(joint_info['local_rotquat'])
            parent_idx_list.append(parent_idx)
        self._xyz = torch.tensor(xyz_list, dtype=torch.float32, device=device) * skel_scale # will be updated with zero grad
        self._local_rotquats = torch.tensor(local_rotquat_list, dtype=torch.float32, device=device) # will never be used or updated
        self.parent_inds = torch.tensor(parent_idx_list, dtype=torch.int32, device=device)
        self._zero_bones = nn.Parameter(self.compute_zero_bones().clone().detach(), requires_grad=self.requires_grad)
        self._global_rotquats = nn.Parameter(self.compute_global_rotation().clone().detach(), requires_grad=self.requires_grad)
        
        # atc1_close
        self.global_rotquats_lr = 4e-3
        self.zero_bones_lr = 1e-3
        
        if 'upperarm_l' in self.name2idx:
            self.loss_joints = [
                'upperarm_l', 'lowerarm_l', 'hand_l', 
                'upperarm_r', 'lowerarm_r', 'hand_r',
                'thigh_l', 'calf_l', 'foot_l',
                'thigh_r', 'calf_r', 'foot_r',
            ]
            
            self.moving_joints = {'thigh_l': 0.01, 'thigh_r': 0.01, 'calf_l': 0.01, 'calf_r': 0.01, 
                                'upperarm_l': 0.01, 'upperarm_r': 0.01, 'lowerarm_l': 0.01, 'lowerarm_r': 0.01}
        elif 'left_shoulder' in self.name2idx:
            self.loss_joints = [
                'left_shoulder', 'left_elbow', 'left_wrist', 
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle',
            ]
            
            self.moving_joints = {'left_hip': 1.0, 'right_hip': 1.0, 
                                'left_knee': 1.0, 'right_knee': 1.0, 
                                'left_shoulder': 1.0, 'right_shoulder': 1.0, 
                                'left_elbow': 1.0, 'right_elbow': 1.0,}
        else:
            raise NotImplementedError(f"Unknown skeleton with joints: {self.idx2name}")
        self.moving_bones = []
        
        self.loss_joints_inds = torch.tensor([self.name2idx[x] for x in self.loss_joints], dtype=torch.int32, device=device)
        self.moving_joint_weights = torch.zeros(len(self._xyz), dtype=torch.float32, device=device)
        for joint_name, weight in self.moving_joints.items():
            self.moving_joint_weights[self.name2idx[joint_name]] = weight
        self.moving_bones_inds = torch.tensor([self.name2idx[x] for x in self.moving_bones], dtype=torch.int32, device=device)
        self.moving_bones_mask = torch.zeros(len(self._xyz), dtype=bool, device=device)
        self.moving_bones_mask[self.moving_bones_inds] = 1
        
        if self.requires_grad:
            self.init_optimizer_scheduler()
        
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_local_rotquats(self):
        return roma.quat_normalize(self._local_rotquats)

    @property
    def get_global_rotquats(self):
        return roma.quat_normalize(self._global_rotquats)
    
    @property
    def get_global_rotquats_norm(self):
        return self._global_rotquats

    @property
    def get_local_rotmats(self):
        return roma.unitquat_to_rotmat(self.get_local_rotquats)
    
    @property
    def get_global_rotmats(self):
        return roma.unitquat_to_rotmat(self.get_global_rotquats)
    
    @property
    def get_global_rotmats_norm(self):
        return roma.unitquat_to_rotmat(self.get_global_rotquats_norm)
    
    def init_optimizer_scheduler(self):
        # black bg
        
        
        self.optimizer = optim.Adam([{'params': self._global_rotquats, 'lr': self.global_rotquats_lr},
                                {'params': self._zero_bones, 'lr': self.zero_bones_lr}], betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=[self.global_rotquats_lr, self.zero_bones_lr], total_steps=100, pct_start=0.1, final_div_factor=0.15)
        # self.scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=60, power=1, )
    
    @torch.no_grad()
    def set_local_rotquats(self, inds, quats):
        """
        Args:
            inds (torch.Tensor M)
            quats (torch.Tensor M x 4)
        """
        assert len(inds) == len(quats)
        self._local_rotquats[inds] = quats
        self.global_rotquats_up_to_date = False
    
    @torch.no_grad()
    def compute_global_rotation(self):
        self._global_rotquats = torch.zeros_like(self._local_rotquats)
        self._global_rotquats[:, -1] = 1
        self._global_rotquats[0, :] = self._local_rotquats[0, :]
        self.joints['root'].compute_children_global_rotation()
        return self._global_rotquats
    
    @torch.no_grad()
    def compute_zero_bones(self):
        self.joints['root'].unrotate_tree()
        parent_xyz = torch.stack([self._xyz[x.parent_idx] for x in self.joints.values()])
        zero_bones = self._xyz - parent_xyz
        self.joints['root'].rotate_tree()
        return zero_bones
        
    def drive(self):
        self._xyz = self._xyz.detach()
        self._xyz[1:] = torch.bmm(self.get_global_rotmats[self.parent_inds], self._zero_bones[..., None]).squeeze()[1:]
        
        # with torch.no_grad(): # independent bone update
        self.joints['root'].shift_children()

    def drive_with_chains(self, chains, cpu=True):
        self._xyz = self._xyz.detach()
        self._xyz[1:] = torch.bmm(self.get_global_rotmats[self.parent_inds], self._zero_bones[..., None]).squeeze()[1:]

        if cpu:
            a = self._xyz.cpu()
            b = chains.cpu()
            c = drive(a, b)
            self._xyz = c.cuda()
        else:
            print("Unsupported")


    def precomp_chains(self):
        global max_depth
        max_depth = 0
        self.joints['root'].trace_children(None, None, 0)

        chains = torch.zeros((self._xyz.size(0), max_depth)).long() - 1
        trace = torch.empty((0))
        self.joints['root'].trace_children(chains, trace, 0)
        return chains
         
    
    def optimize_step(self):
        with torch.no_grad():
            if self._global_rotquats.grad is None:
                return
            self._global_rotquats.grad *= self.moving_joint_weights.unsqueeze(1)
            self._zero_bones.grad[~self.moving_bones_mask] = 0
        # breakpoint()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        # breakpoint()
        # self.global_rotquats_up_to_date = False
    
    @torch.no_grad()
    def perturb(self, 
                max_deg=5
                ):
        perturbation_dict = {
            'thigh_l': None, # half range in degree
            'thigh_r': None,
            'calf_l': [0, 0, 1],
            'calf_r': [0, 0, 1],
            'upperarm_l': None,
            'upperarm_r': None,
            'lowerarm_l': [0, 0, 1],
            'lowerarm_r': [0, 0, 1],
        }
        
        # update local rotation
        inds = []
        quats = []
        for joint_name, axis in perturbation_dict.items():
            max_rad = np.deg2rad(max_deg)
            if axis is None:
                axis = np.random.uniform(low=-1.0, high=1.0, size=3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(0, max_rad)
            rotvec = torch.from_numpy(axis * angle).to(self.device)
            quat = roma.rotvec_to_unitquat(rotvec)
            inds.append(self.name2idx[joint_name])
            quats.append(quat)
        inds = torch.tensor(inds, dtype=torch.int32, device=self.device)
        quats = torch.stack(quats).to(torch.float32).to(self.device)
        self.set_local_rotquats(inds, roma.quat_composition([quats, self.get_local_rotquats[inds]], normalize=True))
        
        # update global rotation
        self._global_rotquats = nn.Parameter(self.compute_global_rotation().clone().detach(), requires_grad=self.requires_grad)
        if self.requires_grad:
            self.init_optimizer_scheduler() # re-init the optimizer and scheduler
        
        # update xyz
        self.drive() 
        
        return self
    
    def read_skl_file(self, filename):
        joint_infos = {}
        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                elements = line.strip().split(' ')
                child_name, parent_name = elements[0], elements[1]
                if child_name in IGNORED_JOINT_NAMES:
                    continue
                if parent_name in joint_infos:
                    joint_infos[parent_name]['children_names'].append(child_name)
                x, y, z = float(elements[2]), float(elements[3]), float(elements[4])
                rot_x, rot_y, rot_z, rot_w = float(elements[5]), float(elements[6]), float(elements[7]), float(elements[8])
                joint_infos[child_name] = {
                    "parent_name": parent_name,
                    "xyz": [x, y, z],
                    "local_rotquat": [rot_x, rot_y, rot_z, rot_w],
                    "children_names": [],
                }
        return joint_infos
        
    def save_skl_file(self, filename):
        with open(filename, 'w') as f:
            f.write('name parent pos_x pos_y pos_z rot_x rot_y rot_z rot_w\n')
            for joint in self.joints.values():
                xyz = np.array(joint.get_xyz.detach().cpu())
                rot = np.array(joint.get_local_rotquat.detach().cpu())
                parent_name = self.idx2name[joint.parent_idx] if joint.parent_idx is not None else joint.name
                line = f"{joint.name} {parent_name} {xyz[0]} {xyz[1]} {xyz[2]} {rot[0]} {rot[1]} {rot[2]} {rot[3]}\n"
                f.write(line)

def skeleton_error(skeleton_1: SkeletonGlobal, skeleton_2: SkeletonGlobal):
    eval_joint_names = [
                        # 'pelvis', 'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'neck_01', 'neck_02', 'head',
                        'upperarm_l', 'lowerarm_l', 'hand_l',
                        'upperarm_r', 'lowerarm_r', 'hand_r',
                        'thigh_l', 'calf_l', 'foot_l',
                        'thigh_r', 'calf_l', 'foot_r'
                        ]
    eval_joint_inds = torch.tensor([skeleton_1.name2idx[x] for x in eval_joint_names]).to(skeleton_1.device)
    
    dist = ((skeleton_1.get_xyz[eval_joint_inds] - skeleton_2.get_xyz[eval_joint_inds])**2).sum(dim=1).sqrt().mean()
    return dist