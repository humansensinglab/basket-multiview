import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import roma


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


class SkeletonNode:
    def __init__(self, skeleton, joint_name, joint_idx, parent_idx, children_inds, device):
        self.skeleton = skeleton
        self.name = joint_name
        self.idx = joint_idx
        self.parent_idx = parent_idx if parent_idx != joint_idx else None
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
    
    def translate_all_children(self, translation):
        """Translate all the children with the given translation
        Args:
            translation: (3)
        """
        if len(translation.shape) == 1:
            translation = translation.unsqueeze(0)
        inds = self.all_children_inds
        if len(inds) > 0:
            self.skeleton._xyz[inds] += translation
    
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


class Skeleton():
    def __init__(self, skl_path, device='cpu', requires_grad=False):
        joint_infos = self.read_skl_file(skl_path)
        
        self.name2idx = {}
        self.idx2name = []
        for joint_idx, joint_name in enumerate(joint_infos.keys()):
            self.name2idx[joint_name] = joint_idx
            self.idx2name.append(joint_name)
        
        xyz_list = []
        local_rotquat_list = []
        self.joints = {}
        for joint_name, joint_info in joint_infos.items():
            joint_idx = self.name2idx[joint_name]
            parent_idx = self.name2idx[joint_info['parent_name']]
            children_inds = [self.name2idx[x] for x in joint_info['children_names']]
            joint = SkeletonNode(self, joint_name, joint_idx, parent_idx, children_inds, device)
            self.joints[joint_name] = joint
            xyz_list.append(joint_info['xyz'])
            local_rotquat_list.append(joint_info['local_rotquat'])
        self._xyz = torch.tensor(xyz_list, dtype=torch.float32, device=device)
        self._local_rotquats = nn.Parameter(torch.tensor(local_rotquat_list, dtype=torch.float32, device=device), requires_grad=requires_grad)
        self.compute_global_rotation()
        
        with torch.no_grad():
            self.joints['root'].unrotate_tree()
            self._xyz_zerorot = self._xyz.clone().detach()
            self.joints['root'].rotate_tree() 
        
        if 'upperarm_l' in self.name2idx:
            self.loss_joints = [
                'upperarm_l', 'lowerarm_l', 'hand_l', 
                'upperarm_r', 'lowerarm_r', 'hand_r',
                'thigh_l', 'calf_l', 'foot_l',
                'thigh_r', 'calf_r', 'foot_r',
            ]
            self.moving_joints = ['thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r']
        elif 'left_shoulder' in self.name2idx:
            self.loss_joints = [
                'left_shoulder', 'left_elbow', 'left_wrist', 
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle',
            ]
            self.moving_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']
        else:
            raise NotImplementedError("Unknown skeleton format.")

        self.loss_joints_inds = torch.tensor([self.name2idx[x] for x in self.loss_joints], dtype=torch.int32, device=device)
        self.moving_joints_inds = torch.tensor([self.name2idx[x] for x in self.moving_joints], dtype=torch.int32, device=device)
        self.moving_joints_mask = torch.zeros(len(self._xyz), dtype=bool, device=device)
        self.moving_joints_mask[self.moving_joints_inds] = 1
        
        starting_lr = 5e-3
        self.optimizer = optim.Adam([{'params': self._local_rotquats}], lr=starting_lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=starting_lr, total_steps=60,
                                                       pct_start=0.1, final_div_factor=0.15)
        self.device = device
        
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_local_rotquats(self):
        return roma.quat_normalize(self._local_rotquats)

    @property
    def get_global_rotquats(self):
        if not self.global_rotquats_up_to_date:
            self.compute_global_rotation()
        return roma.quat_normalize(self._global_rotquats)
    
    @property
    def get_local_rotmats(self):
        return roma.unitquat_to_rotmat(self.get_local_rotquats)
    
    @property
    def get_global_rotmats(self):
        return roma.unitquat_to_rotmat(self.get_global_rotquats)
    
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
    
    def compute_global_rotation(self):
        self._global_rotquats = torch.zeros_like(self._local_rotquats)
        self._global_rotquats[:, -1] = 1
        self.joints['root'].compute_children_global_rotation()
        self.global_rotquats_up_to_date = True
    
    @torch.no_grad()
    def perturb(self, hr=5):
        perturbation_dict = {
            'thigh_l': (hr, hr, hr), # half range in degree
            'thigh_r': (hr, hr, hr),
            'calf_l': (0, 0, 2*hr),
            'calf_r': (0, 0, 2*hr),
            'upperarm_l': (hr, hr, hr),
            'upperarm_r': (hr, hr, hr),
            'lowerarm_l': (0, 0, 2*hr),
            'lowerarm_r': (0, 0, 2*hr),
        }
        
        self.joints['root'].unrotate_tree()
        inds = []
        quats = []
        for joint_name, (x_hr_deg, y_hr_deg, z_hr_deg) in perturbation_dict.items():
            x_hr_rad, y_hr_rad, z_hr_rad = np.deg2rad(x_hr_deg), np.deg2rad(y_hr_deg), np.deg2rad(z_hr_deg)
            roll = np.random.uniform(-x_hr_rad, x_hr_rad)
            pitch = np.random.uniform(-y_hr_rad, y_hr_rad)
            yaw = np.random.uniform(-z_hr_rad, z_hr_rad)
            quat = roma.euler_to_unitquat("xyz", [roll, pitch, yaw]).to(self.device)
            inds.append(self.name2idx[joint_name])
            quats.append(quat)
        inds = torch.tensor(inds, dtype=torch.int32, device=self.device)
        quats = torch.stack(quats).to(torch.float32).to(self.device)
        self.set_local_rotquats(inds, roma.quat_composition([quats, self.get_local_rotquats[inds]], normalize=True))
        self.joints['root'].rotate_tree()
        return self
        
    def drive(self):
        self._xyz = self._xyz_zerorot.clone().detach()
        self.joints['root'].rotate_tree()
    
    def optimize_step(self):
        with torch.no_grad():
            self._local_rotquats.grad[~self.moving_joints_mask] = 0
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_rotquats_up_to_date = False
    
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