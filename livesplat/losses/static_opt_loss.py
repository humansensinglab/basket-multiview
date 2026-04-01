from losses._utils import BaseLoss, l1_loss, ssim
from scene_hs import GaussianModel
import torch
from torchvision.utils import save_image


class StaticOptLoss(BaseLoss):

    def __init__(self, conf: dict):
        self.conf = conf
        super().__init__(conf=conf)

    def compute_loss(
        self,
        gt_dict: dict,
        pred: dict,
        gaussian_model: GaussianModel = None,
        motion_info: dict = None,
        frameID=None,
        depth_r=None,
        depth_gt=None,
        scale_reg: float = 0,
    ) -> dict:
        
        alpha = gt_dict['alpha']
        mask = alpha > 0.5
        Ll1 = l1_loss(gt_dict["im"] * mask, pred * mask)
        ssim_score = []
        if frameID == 0:
            ssim_score = ssim(
                gt_dict["im"] * mask, pred * mask
            )
            # ssim_score = ssim(gt_dict['im'], pred_dict)
            total_loss = (1.0 - self.conf["lambda_dssim"]) * Ll1 + self.conf[
                "lambda_dssim"
            ] * (1.0 - ssim_score)
            if depth_r is not None:
                total_loss += (
                    0.004 * torch.sum((depth_r - depth_gt) ** 2) / depth_r.numel()
                )
            total_loss += self.quantile_ratio_penalty(
                gaussian_model._scaling, reg_lambda=0.1
            )
        else:
            # Ll1 = l1_loss(gt_dict['im'], pred_dict)
            total_loss = Ll1

        loss_dict = {
            "l1_im": Ll1,
            "ssim_score": ssim_score if frameID == 0 else 0,
            "total_loss": total_loss,
        }
        return loss_dict

    def quantile_ratio_penalty(self, scales, lower_q=0.3, upper_q=0.7, reg_lambda=1.0):
        scales_norm = torch.norm(scales, dim=1, p=2)  # shape [N]
        q_vals = torch.quantile(
            scales_norm, torch.tensor([lower_q, upper_q], device=scales_norm.device)
        )
        q_low, q_high = q_vals[0], q_vals[1]
        eps = 1e-8
        ratio = q_high / (q_low + eps)
        penalty = (ratio - 1.0) ** 2
        penalty = reg_lambda * penalty

        return penalty

    def backward(self, loss):
        loss["total_loss"].backward()
