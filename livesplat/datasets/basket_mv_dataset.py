import torch
import numpy as np
import os.path as osp
import glob
import cv2
import json
from tqdm import trange
from torchvision import transforms
from scene_hs.colmap_loader import qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from scene_hs.dataset_readers import getNerfppNorm
from utils.graphics_utils import getWorld2View2

COLOR_MAP = {
    'player_0':  [64, 192, 64],
    'player_1':  [192, 192, 64],
    'player_2':  [64, 64, 192],
    'player_3':  [192, 64, 192],
    'player_4':  [64, 192, 192],
    'player_5':  [192, 192, 192],
    'player_6':  [160, 96, 96],
    'player_7':  [255, 96, 96],
    'player_8':  [128, 192, 64],
    'player_9':  [255, 64, 64],
    'player_10': [160, 32, 64],
    'ball':      [160, 64, 192],
    'background': [0, 0, 0]
}

PALETTE = torch.tensor(list(COLOR_MAP.values()), dtype=torch.long)
KEYS = list(COLOR_MAP.keys())

class BasketMVCamDataset():
    def __init__(self, conf:dict):
        self.conf = conf
        self.data_path = self.conf['data_path']
        self.spatial_scale_factor = self.conf['spatial_scale_factor']
        self.img_scale_factor = self.conf['img_scale_factor']
        self.verbose = self.conf['verbose']
        self.require_all_img_data_available = self.conf['require_all_img_data_available']
        self.first_k_cams = self.conf['first_k_cams']
        self.timestamp = self.conf['timestamp']
        self.caching = self.conf['dataset_caching']
        self.rgb_source_path = self.conf['rgb_source_path'] if 'rgb_source_path' in self.conf else ''
        self.object_key = self.conf['object_key'] if 'object_key' in self.conf else None
        self.erosion_kernel_size = self.conf['erosion_kernel_size'] if 'erosion_kernel_size' in self.conf else 0
        
        self.data_type = {
            'rgb': {
                'type': 'ColorImage',
                'path': osp.join(self.data_path, 'rgb') if self.rgb_source_path == '' else self.rgb_source_path,
                'imread_flag': cv2.IMREAD_COLOR,
                'cvt_color_flag': cv2.COLOR_BGR2RGB
                },
            'depth': {
                'type': 'DepthImage',
                'path': osp.join(self.data_path, 'true_depth') if osp.exists(osp.join(self.data_path, 'true_depth')) else osp.join(self.data_path, 'depth'),
                'imread_flag': cv2.IMREAD_ANYDEPTH,
                'cvt_color_flag': None
                },
            'normal': {
                'type': 'NormalImage',
                'path': osp.join(self.data_path, 'normals'),
                'imread_flag': cv2.IMREAD_COLOR,
                'cvt_color_flag': cv2.COLOR_BGR2RGB
                },
            'semantic': {
                'type': 'SemanticImage',
                'path': osp.join(self.data_path, 'masks'),
                'imread_flag': cv2.IMREAD_COLOR,
                'cvt_color_flag': cv2.COLOR_BGR2RGB
                },
        }
        
        if not self.conf['load_depth']:
            del self.data_type['depth']
        
        self.img_transforms = transforms.Compose([transforms.ToTensor()])
        self.data = self.load_data()
        # self.t_idx = self.conf['t_idx']
        
        self.cached_data = {}        
        
        if self.caching:
            num_elements = self.__len__()
            for idx in trange(num_elements):
                dst_batch = self.__getitem__(idx)        
                self.cached_data[idx] = dst_batch  
    
        
    def load_data(self):
        data = []
        cameras_extrinsic_file = osp.join(self.data_path, 'cameras', 'images.bin')
        cameras_intrinsic_file = osp.join(self.data_path, 'cameras', 'cameras.bin')
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        
        depth_params_file = osp.join(self.data_path, "cameras", "depth_params.json")
        if osp.exists(depth_params_file):
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            self.depth_params = depths_params
        else:
            self.depth_params = None

        ids = sorted(cam_extrinsics.keys(), key=int)
        if self.first_k_cams > 0:
            ids = ids[:self.first_k_cams]
        
        for id in ids:
            extr = cam_extrinsics[id]
            intr = cam_intrinsics[extr.camera_id]
            cam_name = extr.camera_id
            cam = {"id" : cam_name}
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            T = self.spatial_scale_factor * T
            
            w2c_aug = torch.tensor(getWorld2View2(R, T))
            c2w = np.linalg.inv(w2c_aug)[0:3,:]
            
            cam['extrinsics'] = {
                'R': R,
                'C': T,
                'c2w': c2w,
                'w2c': w2c_aug[0:3,:].numpy()
                }
            intrinsic = np.array([[intr.params[0], 0, intr.params[2]],
                                  [0, intr.params[1], intr.params[3]],
                                  [0, 0, 1]])
            img_scaling_matrix = np.array(
                [[self.img_scale_factor, 0 , 0],
                [0, self.img_scale_factor, 0],
                [0, 0, 1]]) 
            cam['intrinsics'] = img_scaling_matrix @ intrinsic

            img_data_paths = {}
            for cur_key, cur_val in self.data_type.items(): 
                if cur_key == 'semantic':
                    img_data_paths[cur_key] = glob.glob(osp.join(cur_val['path'], f"cam_{cam_name:04d}", f"{self.timestamp:04d}*.png"))[0]
                else:
                    img_data_paths[cur_key] = glob.glob(osp.join(cur_val['path'], f"cam_{cam_name:04d}", f"{self.timestamp:04d}*"))[0]
            cam['img_data_paths'] = img_data_paths
        
            data.append(cam)
        
        self.nerfpp_norm = getNerfppNorm(data)
        return data

    def erode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Erode a 2D binary mask (H, W) using a square kernel of size erosion_kernel_size."""
        if self.erosion_kernel_size <= 0:
            return mask
        kernel = np.ones((self.erosion_kernel_size, self.erosion_kernel_size), dtype=np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        return torch.from_numpy(eroded).to(mask.device, dtype=mask.dtype)

    def rgb2semantic(self, img, key=None):
        image_int = (img * 255).round().long().permute(1, 2, 0)
    
        if key is not None:
            if key not in COLOR_MAP:
                raise ValueError(f"Key '{key}' not found in COLOR_MAP.")
                
            target_color = torch.tensor(COLOR_MAP[key], device=img.device)
            return (image_int == target_color).all(dim=2)

        else:
            mask = img.any(dim=0)
            seg_img = mask.to(dtype=torch.long)
            return seg_img
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if idx in self.cached_data:
            dst_batch = self.cached_data[idx]
        else:
            cur_cam_data = self.data[idx]
            img_data = {}
            for cur_key, cur_val in self.data_type.items():
                img_path = cur_cam_data['img_data_paths'][cur_key]
                if cur_key == 'depth':
                    true_depth = np.load(img_path)['arr_0']
                    epsilon = 1e-8
                    img = np.where( true_depth > epsilon, 1.0 / true_depth, 0.0).astype(np.float32)
                else:
                    img = cv2.imread(img_path, cur_val['imread_flag'])
                    if cur_val['cvt_color_flag'] is not None:
                        img = cv2.cvtColor(img, cur_val['cvt_color_flag'])
                    img = cv2.resize(img, (0,0), fx=self.img_scale_factor, fy=self.img_scale_factor, interpolation=cv2.INTER_AREA)
                img_data[cur_key] = self.img_transforms(img)
            
            width = img_data['rgb'].shape[-1]
            height = img_data['rgb'].shape[-2]
            
            cam_info = {
                "width": width,
                "height": height,
                "K": cur_cam_data['intrinsics'],
                "w2c": cur_cam_data['extrinsics']['w2c']
            }
            alpha = (self.rgb2semantic(img_data['semantic'], key=self.object_key) > 0).to(torch.float32)
            alpha = self.erode_mask(alpha)

            dst_batch = { 
                "cam_info": cam_info,
                "im": img_data['rgb'],
                "inv_depth": img_data['depth'] if 'depth' in img_data else None,
                "normal": img_data['normal'],
                "seg": alpha,
                "id": int(cur_cam_data['id']),
                "alpha": alpha,
            }
            
            if self.caching:
                self.cached_data[idx] = dst_batch 
                
        return dst_batch
    
