import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from scene_hs import GaussianModel

from fused_ssim import fusedssim, fusedssim_backward, fused_ssim

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2, True)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

class BaseLoss(Module):
    def __init__(self, conf:dict):
        self.conf = conf
        
    def compute_loss(self, 
        gt_dict:dict, 
        pred_dict:dict, 
        gaussian_model:GaussianModel,
        motion_info:dict) -> dict:
        
        return None
    
    def backward(self, loss):
        loss.backward()
        
     
def l1_loss(pred, gt):
    return torch.abs((pred - gt)).mean()

def l1_loss_v2(pred, gt):
    return torch.abs((pred - gt).sum(-1)).mean()

def l2_loss(pred, gt):
    return ((pred - gt) ** 2).mean()

def weighted_l2_loss(pred, gt, weight):
    return torch.sqrt(((pred - gt) ** 2).sum(-1) * weight + 1e-20).mean()

def weighted_l2_loss_v1(pred, gt, weight):
    return torch.sqrt(((pred - gt) ** 2) * weight + 1e-20).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def fast_ssim(img1, img2):
    return fused_ssim(img1.unsqueeze(0), img2.unsqueeze(0), train=True)

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_stretch_loss(cov3D_upper, thres):
    cov3D_real = torch.hstack([cov3D_upper[:,0:3], cov3D_upper[:,1::6], cov3D_upper[:,3:5], 
                               cov3D_upper[:,2::6], cov3D_upper[:,4:6]]).reshape(-1, 3, 3)
    eigenvalues, _ = torch.linalg.eigh(cov3D_real)
    ratios = eigenvalues[:, -1] / (eigenvalues[:, -2]+1e-8)

    loss = torch.nn.functional.elu(ratios - thres, alpha=1).mean()

    return loss