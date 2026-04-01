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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_masked(img1, img2, mask):
    """
    Args:
        img1: (c x h x w)
        img2: (c x h x w)
        mask: (1 x h x w)
    """
    c, h, w = img1.shape
    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    se = (img1 - img2) ** 2 # (c x h x w)
    se_flatten = se.view(c, -1) # (c x hw)
    mask_flatten = mask.flatten() # (hw)
    se_flatten_masked = se_flatten[..., mask_flatten]
    mse = se_flatten_masked.mean(axis=1, keepdim=True)
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    if torch.isnan(psnr).any():
        print(f"mse={mse}, psnr={psnr}")
        breakpoint()
    return psnr
