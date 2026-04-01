from losses._utils import BaseLoss, l1_loss, ssim, fast_ssim
from utils.image_utils import psnr
from scene_hs import GaussianModel
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
# from utils.dynamic_utils import get_joint_scores

def get_joint_scores(points, skl_joints):
    """
    Args:
        points (Nx3): torch.tensor
        skl_joints: a list of joints on the skeleton
    
    Returns:
        joints_score (N): torch.tensor, a list indicating the joint score of each point
    """
    if 'upperarm_l' in skl_joints.keys():
        selected_joints = {
            'calf_l': 1.3, 
            'calf_r': 1.3, 
            'foot_l': 0.5, 
            'foot_r': 0.5,
            'lowerarm_l': 1.0, 
            'lowerarm_r': 1.0,
            'hand_l': 0.7, 
            'hand_r': 0.7,
        }
    elif 'left_shoulder' in skl_joints.keys():
        selected_joints = {
            'left_knee': 1.3, 
            'right_knee': 1.3, 
            'left_ankle': 0.5, 
            'right_ankle': 0.5,
            'left_elbow': 1.0, 
            'right_elbow': 1.0,
            'left_wrist': 0.7, 
            'right_wrist': 0.7,            
        }
    else:
        raise NotImplementedError("Unknown skeleton format.")
    
    def bump_function(x):
        c = 0.04
        func_res = torch.exp(1 - 1 / (1 - (c * x) ** 2))
        mask = torch.abs(c * x) < 1 - 1e-5
        final_res = func_res * mask
        final_res[torch.isnan(final_res)] = 0
        return final_res

    with torch.no_grad():
        joint_scores = torch.zeros((len(selected_joints), len(points)), dtype=torch.float32, device=points.device)
        for joint_idx, (joint_name, joint_weight) in enumerate(selected_joints.items()):
            joint_position = skl_joints[joint_name].get_xyz.to(points.device)
            dists = torch.linalg.norm(points - joint_position.unsqueeze(0), dim=1)
            joint_scores[joint_idx] = bump_function(dists) * joint_weight
        joint_scores = torch.max(joint_scores, dim=0)[0]
    
    # print(f"joint_scores min: {joint_scores.min()}, max: {joint_scores.max()}")
    
    return joint_scores

class DefaultLoss(BaseLoss):
    
    def __init__(self, conf:dict):
        self.conf = conf
        super().__init__(conf=conf)
    
    def comp_img_based_loss(self, pred_img, gt_img):
        Ll1 = l1_loss(gt_img, pred_img)
        ssim_score = fast_ssim(gt_img, pred_img).item()
        psnr_score = psnr(gt_img, pred_img).mean().item()
        loss = (1.0 - self.conf['lambda_dssim']) * Ll1 + \
            self.conf['lambda_dssim'] * (1.0 - ssim_score)
        
        return {
            "total_loss": loss,
            "l1": Ll1.item(),
            "ssim": ssim_score,
            "psnr": psnr_score,
        }
        
    def compute_loss(self, 
        gt_dict:dict, 
        pred_dict:dict, 
        gaussian_model:GaussianModel=None, 
        motion_info:dict=None,
        mask:torch.Tensor=None,
        skl_joints:dict=None,
        bg_color=None) -> dict:
        
        loss_dict = {}

        gt = gt_dict['im']
        alpha = gt_dict['alpha']
        mask = alpha > 0.5
        
        render = pred_dict['render']
        
        _, h, w = gt.shape
        bg = bg_color.T.unsqueeze(2).repeat(1, h, w)
        gt_new = gt * mask + (~mask) * bg
        
        # Ll1 = l1_loss(gt_new, render)
        ssim_score = ssim(gt_new, render)
        loss_dict = self.comp_img_based_loss(gt_new, pred_dict['render'])
        
        if skl_joints is not None:
            scales = gaussian_model.get_scaling
            ratios = scales.max(dim=1)[0] / (scales.min(dim=1)[0] + 1e-7)
            gs_joint_scores = get_joint_scores(gaussian_model.get_xyz, skl_joints)
            loss_dict['scale'] = torch.mean(ratios * (ratios > 3) * gs_joint_scores)
            loss_dict['total_loss'] += loss_dict['scale']

        return loss_dict
        
    def backward(self, loss):
        loss['total_loss'].backward()
        