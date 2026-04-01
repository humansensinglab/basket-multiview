import torch
import numpy as np
from post_proc._utils import BasePostProc
from post_proc._utils import o3d_knn

class DynamicPostProc(BasePostProc):
    
    def __init__(self, conf: dict):
        super().__init__(conf)
        self.conf = conf
    
    def get_neighbors(self, gauss_model):    
        
        pts = gauss_model.get_xyz.detach().cpu().numpy()
        num_knn=20
        neighbor_sq_dist, neighbor_indices = o3d_knn(pts=pts, num_knn=num_knn)
        neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
        neighbor_dist = np.sqrt(neighbor_sq_dist)
        
        neighbor_data = {
            'indices': torch.tensor(neighbor_indices).cuda().long().contiguous(),
            'weight': torch.tensor(neighbor_weight).cuda().float().contiguous(),
            'dist': torch.tensor(neighbor_dist).cuda().float().contiguous()
        }

        return neighbor_data
    
    def process(self, last_gauss_model, motion_info, frame_idx) -> dict:
        # Get the neighbor data
        # there_is_prev_info = True
        if motion_info is None:
            motion_info = {
                'neighbor_data': self.get_neighbors(gauss_model=last_gauss_model)
            }
            # there_is_prev_info = False
            
        cur_rotation = last_gauss_model.get_rotation.detach().clone()
        inv_rotation = cur_rotation.clone()
        inv_rotation[:, 1:] = -1 * inv_rotation[:, 1:] 
        cur_xyz = last_gauss_model.get_xyz.detach().clone()
        offset = cur_xyz[motion_info['neighbor_data']['indices']] - cur_xyz[:, None]
            
        motion_info['rigid_info'] = {
            'prev_inv_rotation': inv_rotation,
            'prev_offset': offset
        }
            
        motion_info['prev_model'] = last_gauss_model.get_params_detached()
            
        # if there_is_prev_info: 
    
        #     trans_delta = cur_xyz - motion_info['prev_model']['xyz']
        #     new_xyz =  cur_xyz + trans_delta
            
        #     rot_delta = cur_rotation - motion_info['prev_model']['rotation']
        #     new_rot = cur_rotation + rot_delta
            
        #     # Updating positions and rotations with a linear approximation x + delta 
        #     last_gauss_model.update_param(type='xyz', value=new_xyz)
        #     last_gauss_model.update_param(type='rotation', value=new_rot)

    
        return motion_info