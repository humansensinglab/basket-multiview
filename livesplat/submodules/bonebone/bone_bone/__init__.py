from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import time

def drive(
    xyz,
    chains
):
    return _Drive.apply(
        xyz,
        chains
    )

class _Drive(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz,
        chains
    ):

        args = (
            xyz,
            chains,
            True
        )

        new_xyz = _C.drive(*args)

        # Keep relevant tensors for backward
        ctx.save_for_backward(chains)

        return new_xyz

    @staticmethod
    def backward(ctx, grad_new_xyz):

        # Restore necessary values from context
        chains, = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            grad_new_xyz, 
            chains,
            True)

        grad_rot_xyz = _C.drive_backward(*args)

        grads = (
            grad_rot_xyz,
            None
        )

        return grads
    
def bk_gather(
    values,
    indices
):
    return _Gather.apply(
        values,
        indices
    )

class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values,
        indices
    ):

        args = (
            values,
            indices
        )

        gathered = _C.bk_gather(*args)


        # Keep relevant tensors for backward
        ctx.save_for_backward(indices)
        ctx.dim = values.size(0)

        return gathered

    @staticmethod
    def backward(ctx, gathered_grad):

        # Restore necessary values from context
        indices, = ctx.saved_tensors
        D = ctx.dim

        # Restructure args as C++ method expects them
        args = (
            D,
            gathered_grad.cpu(), 
            indices.cpu()
            )

        value_grads = _C.bk_gather_backward(*args).cuda()

        grads = (
            value_grads,
            None
        )

        return grads

# class Driver(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, means3D, means2D, opacities, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
#         raster_settings = self.raster_settings

#         if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
#             raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
#         if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
#             raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
#         if dc is None:
#             dc = torch.Tensor([])
#         if shs is None:
#             shs = torch.Tensor([])
#         if colors_precomp is None:
#             colors_precomp = torch.Tensor([])

#         if scales is None:
#             scales = torch.Tensor([])
#         if rotations is None:
#             rotations = torch.Tensor([])
#         if cov3D_precomp is None:
#             cov3D_precomp = torch.Tensor([])

#         # Invoke C++/CUDA rasterization routine
#         return rasterize_gaussians(
#             means3D,
#             means2D,
#             dc,
#             shs,
#             colors_precomp,
#             opacities,
#             scales, 
#             rotations,
#             cov3D_precomp,
#             raster_settings
#         )

# class SparseGaussianAdam(torch.optim.Adam):
#     def __init__(self, params, lr, eps):
#         super().__init__(params=params, lr=lr, eps=eps)
    
#     @torch.no_grad()
#     def step(self, visibility, N):
#         for group in self.param_groups:
#             lr = group["lr"]
#             eps = group["eps"]

#             assert len(group["params"]) == 1, "more than one tensor in group"
#             param = group["params"][0]
#             if param.grad is None:
#                 continue

#             # Lazy state initialization
#             state = self.state[param]
#             if len(state) == 0:
#                 state['step'] = torch.tensor(0.0, dtype=torch.float32)
#                 state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
#                 state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


#             stored_state = self.state.get(param, None)
#             exp_avg = stored_state["exp_avg"]
#             exp_avg_sq = stored_state["exp_avg_sq"]
#             M = param.numel() // N
#             _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)