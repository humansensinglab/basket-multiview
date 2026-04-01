#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.store_ordering,
            raster_settings.using_precomp,
            raster_settings.gaussian_list,
            raster_settings.ranges,
            raster_settings.num_buckets,
            raster_settings.num_rendered,
            raster_settings.use_cuda_graph,
            raster_settings.img_buffer,
            raster_settings.geom_buffer,
            raster_settings.sample_buffer,
            raster_settings.tiles_id,
            raster_settings.num_tiles,
            raster_settings.pixel_tr,
            raster_settings.output_img,
            raster_settings.use_load_balancing,
        )

        if not raster_settings.using_precomp and not raster_settings.store_ordering:
            assert (
                not raster_settings.use_cuda_graph
                and not raster_settings.use_load_balancing
            ), "use_cuda_graph and load_balancing require using_precomp to be enabled"

        if raster_settings.using_precomp:

            (
                num_rendered,
                num_buckets,
                ranges,
                gaussian_list,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                sampleBuffer,
                tiles_id,
                num_tiles,
                pixel_t,
                lb_order,
            ) = _C.rasterize_gaussians(*args)

            ctx.raster_settings = raster_settings
            ctx.num_rendered = num_rendered
            ctx.num_buckets = num_buckets
            ctx.num_tiles = raster_settings.num_tiles

            ctx.save_for_backward(
                colors_precomp,
                means3D,
                scales,
                rotations,
                cov3Ds_precomp,
                radii,
                dc,
                sh,
                opacities,
                raster_settings.geom_buffer,
                binningBuffer,
                raster_settings.img_buffer,
                raster_settings.sample_buffer,
                raster_settings.tiles_id,
                raster_settings.pixel_tr,
                lb_order,
            )

            return (
                color,
                None,
                None,
                None,
                None,
                num_rendered,
                0,
                0,
                0,
                0,
                None,
                0,
                None,
            )
        else:

            (
                num_rendered,
                num_buckets,
                ranges,
                gaussian_list,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                sampleBuffer,
                tiles_id,
                num_tiles,
                pixel_t,
                lb_out,
            ) = _C.rasterize_gaussians(*args)

            # Keep relevant tensors for backward
            ctx.raster_settings = raster_settings
            ctx.num_rendered = num_rendered
            ctx.num_buckets = num_buckets
            ctx.num_tiles = raster_settings.num_tiles

            ctx.save_for_backward(
                colors_precomp,
                means3D,
                scales,
                rotations,
                cov3Ds_precomp,
                radii,
                dc,
                sh,
                opacities,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                sampleBuffer,
                raster_settings.tiles_id,
                raster_settings.pixel_tr,
                lb_out,
            )

            if raster_settings.store_ordering:
                return (
                    color,
                    radii,
                    gaussian_list,
                    ranges,
                    None,
                    num_rendered,
                    num_buckets,
                    imgBuffer.size(),
                    geomBuffer.size(),
                    sampleBuffer.size(),
                    tiles_id,
                    num_tiles,
                    pixel_t,
                )
            return (
                color,
                radii,
                None,
                None,
                None,
                num_rendered,
                0,
                0,
                0,
                0,
                None,
                0,
                None,
            )

    @staticmethod
    def backward(ctx, grad_out_color, *_):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        num_tiles = ctx.num_tiles
        raster_settings = ctx.raster_settings

        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            dc,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sampleBuffer,
            tiles_id,
            pixel_tr,
            lb_out,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            num_buckets,
            sampleBuffer,
            raster_settings.debug,
            raster_settings.using_precomp,
            raster_settings.use_cuda_graph,
            raster_settings.gaussian_list,
            raster_settings.ranges,
            raster_settings.dl_ddc,
            raster_settings.dl_dsh,
            raster_settings.dl_dcolors,
            tiles_id,
            num_tiles,
            pixel_tr,
            raster_settings.use_load_balancing,
            False,
            lb_out,
        )

        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_dc,
            grad_sh,
            grad_scales,
            grad_rotations,
        ) = _C.rasterize_gaussians_backward(*args)

        if raster_settings.using_precomp:

            if not raster_settings.use_cuda_graph:

                grads = (
                    None,
                    None,
                    grad_dc,
                    grad_sh,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            else:

                grads = (
                    None,
                    None,
                    raster_settings.dl_ddc,
                    raster_settings.dl_dsh,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        else:

            grads = (
                grad_means3D,
                grad_means2D,
                grad_dc,
                grad_sh,
                grad_colors_precomp,
                grad_opacities,
                grad_scales,
                grad_rotations,
                grad_cov3Ds_precomp,
                None,
            )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    store_ordering: bool
    using_precomp: bool
    gaussian_list: torch.Tensor
    ranges: torch.Tensor
    num_buckets: int
    num_rendered: int
    use_cuda_graph: bool
    img_buffer: torch.Tensor
    geom_buffer: torch.Tensor
    sample_buffer: torch.Tensor
    dl_dsh: torch.Tensor
    dl_ddc: torch.Tensor
    dl_dcolors: torch.Tensor
    tiles_id: torch.Tensor
    num_tiles: int
    pixel_tr: torch.Tensor
    output_img: torch.Tensor
    use_load_balancing: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        dc=None,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
