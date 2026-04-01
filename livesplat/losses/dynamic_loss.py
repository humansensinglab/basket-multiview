import torch
from losses._utils import BaseLoss, l1_loss, ssim, fast_ssim
from utils.image_utils import psnr
from scene_hs import GaussianModel
from losses._utils import l1_loss


class DynamicLoss(BaseLoss):
    
    def __init__(self, conf:dict):
        self.conf = conf
        super().__init__(conf=conf)
        self.weights = {
            'img': 1.0,
            'seg': 1.0,
            'rigid_offset': 4.0,
            'rigid_rot': 4.0, 
            'iso': 2.0,
            'soft_color_constraint': 0.01
        }
    
    def comp_img_based_loss(self, pred_img, gt_img):
        Ll1 = l1_loss(gt_img, pred_img)
        ssim_score = fast_ssim(gt_img, pred_img)
        psnr_score = psnr(gt_img, pred_img).mean().item()
        loss = (1.0 - self.conf['lambda_dssim']) * Ll1 + \
            self.conf['lambda_dssim'] * (1.0 - ssim_score)
        
        return {
            "loss": loss,
            "ssim": ssim_score,
            "psnr": psnr_score,
        }
    
    def compute_loss(self,
        gt_dict:dict, 
        pred_dict:dict, 
        gaussian_model:GaussianModel, 
        motion_info:dict,
        masked=False,
        reg=False,
        bg_color=None) -> dict:
        
        loss_dict = {}
        
        if masked:
            render = pred_dict['render']
            gt = gt_dict['im']
            alpha = gt_dict['alpha']
            fg_mask = alpha > 0
            _, h, w = gt.shape
            bg = bg_color.T.unsqueeze(2).repeat(1, h, w)
            
            gt_valid = (gt * alpha + (1 - alpha) * bg) * fg_mask
            render_valid = render * fg_mask
            
            img_loss_dict = self.comp_img_based_loss(render_valid, gt_valid)
        else:
            img_loss_dict = self.comp_img_based_loss(pred_dict['render'], gt_dict['im'])
        loss_dict['img'] = img_loss_dict['loss']
        loss_dict['ssim'] = img_loss_dict['ssim']
        loss_dict['psnr'] = img_loss_dict['psnr']
            
        total_loss = sum([self.weights[loss_type] * loss_val 
                          for loss_type, loss_val in loss_dict.items()
                          if loss_type in self.weights])
        
        if reg == True:
            if hasattr(gaussian_model, '_xyz_reg'):
                total_loss += l1_loss(gaussian_model._xyz, gaussian_model._xyz_reg) * 0.005
            if hasattr(gaussian_model, '_scaling_reg'):
                total_loss += l1_loss(gaussian_model.get_scaling, gaussian_model._scaling_reg) * 0.005
        
        loss_dict['total_loss'] = total_loss

        
        return loss_dict
        
    def backward(self, loss):
        loss['total_loss'].backward()
        